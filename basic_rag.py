import glob
import json
import os
from typing import List
import faiss
import boto3
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyMuPDFLoader
# from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv

load_dotenv()


def load_pdf(path: str) -> List[Document]:
    try:
        loader = PyMuPDFLoader(path)  # Usa el loader para PDFs
        documents = loader.load()  # Carga todas las páginas

        print(f"PDF '{path}' cargado con {len(documents)} páginas.")
        return documents  # Devuelve la lista completa de documentos

    except Exception as e:
        raise ValueError(f"Error loading PDF '{path}': {e}")


def load_pdfs_from_dir(path: str) -> List[Document]:
    try:
        pdf_files = glob.glob(os.path.join(path, '**', '*.pdf'), recursive=True)
        print(f"Archivos encontrados: {pdf_files}")

        all_documents = []
        for pdf in pdf_files:
            documents = load_pdf(pdf)  # Carga todas las páginas del PDF
            all_documents.extend(documents)  # Añade todas las páginas a la lista general

        print(f"Total de documentos cargados: {len(all_documents)}")
        return all_documents

    except Exception as e:
        raise RuntimeError(f"Error al cargar los PDFs del directorio '{path}': {e}")

def prepare_tfidf_index(documents: List[Document]):
    texts = [doc.page_content for doc in documents]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    return vectorizer, tfidf_matrix, texts


def retrieve_top_k_tfidf(query: str, vectorizer, tfidf_matrix, texts, k=2):
    query_vec = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_k_indices = np.argsort(cosine_similarities)[::-1][:k]
    return [texts[i] for i in top_k_indices]


def exercise1(query: str = ""):
    pdf_docs = load_pdfs_from_dir('./resources/')
    vectorizer, tfidf_matrix, texts = prepare_tfidf_index(pdf_docs)
    return retrieve_top_k_tfidf(query, vectorizer, tfidf_matrix, texts, k=1)


def get_single_response_openai(prompt: str = "", embeddings: List[str] = []):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"],)
    embs = '\n'.join(embeddings)
    prompt_plus = f"""
    Responde la siguiente pregunta:
    {prompt}
    
    Usando la siguiente información:
    {embs}
    """
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Eres un asistente para responder preguntas usando la informacion adjunta"},
            {
                "role": "user",
                "content": prompt_plus
            }
        ]
    )

    return completion.choices[0].message.content


def get_single_response_bedrock(prompt: str) -> str:
    try:
        # Configuración de la sesión de boto3
        boto_session = boto3.session.Session(
            aws_access_key_id=os.environ["AWS_ACCESS_KEY"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"]
        )

        # Cliente de Amazon Bedrock
        bedrock_runtime = boto_session.client(
            service_name="bedrock-runtime",
            region_name="us-east-1",
        )


        body = json.dumps({
            "prompt": prompt,
            "max_tokens_to_sample": 500,
            "temperature": 0.1,
            "top_k": 250,
            "top_p": 0.9,
        })

        modelId = "anthropic.claude-3-sonnet-20240229-v1:0"
        accept = 'application/json'
        contentType = 'application/json'

        response = bedrock_runtime.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)

        response_body = json.loads(response.get('body').read())
        print(response_body)
        return response_body

    except Exception as e:
        print(f"Error al obtener la respuesta de Bedrock: {e}")
        return None


def save_faiss(documents: List[Document], index_path: str = ""):
    texts = [doc.page_content for doc in documents]
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generar los embeddings para los documentos
    doc_embeddings = model.encode(texts, convert_to_tensor=False)

    # Crear el índice FAISS (usando un índice plano L2)
    dimension = doc_embeddings.shape[1]  # Dimensión de los embeddings
    index = faiss.IndexFlatL2(dimension)

    # Añadir los embeddings al índice
    index.add(np.array(doc_embeddings, dtype=np.float32))

    # Guardar el índice FAISS en un archivo
    faiss.write_index(index, index_path)

    return model, texts


def load_faiss(index_path: str):
    index = faiss.read_index(index_path)
    return index


def top_k_from_faiss(query: str, model, index, texts: List[str], k=2):
    query_embedding = model.encode(query, convert_to_tensor=False).reshape(1, -1)

    # Realizar la búsqueda en el índice FAISS
    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), k)

    # Devolver los documentos más relevantes
    # distances[0][j]
    return [texts[i] for j, i in enumerate(indices[0])]


if __name__ == "__main__":
    print('############# EXERCISE 1 ############')
    test_question = "¿Cuáles son las Alianzas Estratégicas de HistoriaCard?"
    embeddings = exercise1(query=test_question)
    print(embeddings)
    print('#'*30)

    print('############# EXERCISE 2 ############')
    resp = get_single_response_openai(prompt=test_question, embeddings=embeddings)
    print(resp)
    print('#' * 30)

    print('############# EXERCISE 3 ############')
    pdf_docs = load_pdfs_from_dir('./resources/')
    index_path = 'pdfs.faiis'
    model, texts = save_faiss(documents=pdf_docs, index_path=index_path)
    index = load_faiss(index_path)
    emb_faiis = top_k_from_faiss(test_question, model, index, texts, k=1)
    # print(emb_faiis)
    resp = get_single_response_openai(prompt=test_question, embeddings=emb_faiis)
    print(resp)
    print('#' * 30)

