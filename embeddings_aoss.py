import glob
import json
import fitz  # PyMuPDF
import boto3
import os
from typing import List
from langchain_aws import BedrockEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import OpenSearchVectorSearch
from opensearchpy import RequestsHttpConnection, OpenSearch
from requests_aws4auth import AWS4Auth
from dotenv import load_dotenv

load_dotenv()

class AOSSEmbeddings:

    def __init__(self, model_id: str = "amazon.titan-embed-text-v1"):
        self.boto_session = boto3.session.Session(
            aws_access_key_id=os.environ["AWS_ACCESS_KEY"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"]
        )
        self.bedrock_runtime = self.boto_session.client(
            service_name="bedrock-runtime",
            region_name="us-east-1",
        )

        self.credentials = self.boto_session.get_credentials()
        self.model_id = model_id
        self.bedrock_embeddings = BedrockEmbeddings(client=self.bedrock_runtime, model_id=self.model_id)

        self.awsauth = AWS4Auth(self.credentials.access_key, self.credentials.secret_key, "us-east-1", "aoss",
                                session_token=self.credentials.token)

        self.opensearch_domain_endpoint = os.environ['OPEN_SEARCH_ENDPOINT']
        self.opensearch_index = os.environ['OPEN_SEARCH_INDEX']

        self.vector = OpenSearchVectorSearch(
            embedding_function=self.bedrock_embeddings,
            index_name=self.opensearch_index,
            http_auth=self.awsauth,
            use_ssl=True,
            verify_certs=True,
            http_compress=True,
            connection_class=RequestsHttpConnection,
            opensearch_url=self.opensearch_domain_endpoint
        )

    def load_pdf(self, path: str) -> List[Document]:
        """
        Lee un PDF y lo divide en secciones según títulos y subtítulos.
        """
        try:
            doc = fitz.open(path)
            sections = []  # Almacena las secciones como Document
            current_section = []  # Almacena las líneas de la sección actual

            # Iterar sobre cada página del PDF
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                blocks = page.get_text("dict")["blocks"]

                # Recorrer los bloques y detectar títulos por tamaño de fuente
                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text = span["text"].strip()
                                font_size = span["size"]

                                # Si detectamos un título, guardamos la sección anterior
                                if font_size > 8 and current_section:
                                    sections.append(
                                        Document(page_content="\n".join(current_section))
                                    )
                                    current_section = []

                                # Agregamos el texto a la sección actual
                                current_section.append(text)

            # Agregar la última sección si no está vacía
            if current_section:
                sections.append(Document(page_content="\n".join(current_section)))

            print(f"PDF '{path}' cargado con {len(sections)} secciones.")
            return sections

        except Exception as e:
            raise ValueError(f"Error loading PDF '{path}': {e}")

    def load_pdfs_from_dir(self, path: str) -> List[Document]:
        """
        Lee todos los PDFs de un directorio y los divide en secciones según títulos y subtítulos.
        """
        try:
            pdf_files = glob.glob(os.path.join(path, '**', '*.pdf'), recursive=True)
            print(f"Archivos encontrados: {pdf_files}")

            all_documents = []
            for pdf in pdf_files:
                documents = self.load_pdf(pdf)  # Carga las secciones del PDF
                all_documents.extend(documents)  # Añade las secciones a la lista general

            print(f"Total de secciones cargadas: {len(all_documents)}")
            return all_documents

        except Exception as e:
            raise RuntimeError(f"Error al cargar los PDFs del directorio '{path}': {e}")

    def store_data(self, pdf_dir: str) -> None:
        # Cargar PDFs y dividirlos en oraciones con ventana de contexto
        data = self.load_pdfs_from_dir(pdf_dir)
        print(data)
        print('*' * 20)
        print('SPLITS')

        # Implementación de Sentence Window Retrieval
        splits = []
        for doc in data:
            sentences = doc.page_content.split('\n')
            for i, sentence in enumerate(sentences):
                # Añadir ventana de contexto (2 oraciones antes y después)
                context = sentences[max(0, i-2):i+3]
                paragraph = "\n".join(context)
                splits.append(Document(page_content=paragraph, metadata=doc.metadata))
        print(splits)
        # Almacenar los fragmentos en OpenSearch
        self.vector.add_documents(documents=splits, vector_field="rag_vector", bulk_size=3000)

        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        # splits = text_splitter.split_documents(data)
        # self.vector.add_documents(
        #     documents=splits,
        #     vector_field="rag_vector",
        #     bulk_size=3000
        # )

    def query(self, question: str, k: int = 5):

        results = self.vector.similarity_search(
            question,
            vector_field="rag_vector",
            text_field="text",
            metadata_field="metadata",
            k=k,
        )
        print(results)
        rr = [{"page_content": r.page_content, "metadata": r.metadata} for r in results]
        data = []

        for doc in rr:
            data.append(f"{doc['page_content']}")

        print('#' * 20)
        print(data)
        print('#' * 20)

        return data


if __name__ == "__main__":
    aoss = AOSSEmbeddings()

    aoss.store_data('./resources/')
    aoss.query(question='Alianzas Estratégicas e Innovación', k=2)