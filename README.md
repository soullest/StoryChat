# How to run

It is recommended to use python 3.10 or higher, as well as the virtual machine of your choice, after which you will only need to install the dependencies with the following command

```
pip install -r requirements.txt
```
After installing the dependencies the system is ready to run in a local environment using the following instruction:

```
streamlit run storychat.py
```

## Project Description

To meet the requested requirements, the project is divided into several parts:

1) basic_rag.py. This file contains the exercises described on the first page of the project and does not contain a graphical interface, only the code to execute specific RAG functions using OpenAI, TfidfVectorizer and SentenceTransformer, as well as the faiss library. It should be noted that since it is an exercise and not a real project, the vectors are calculated locally and saved in the pdf.faiss file.

2) storychat.py. This code solves the exercises on the second page of the specifications file, contains the graphical interface and the communication methods with the user programmed in Streamlit.

3) aoss_embeddings. It makes more professional use of embeddings through an OpenSearch database implemented in AWS, with read permissions for the Role specified in the .env file; It has been indexed in faiss and optimized for Euclidean distance. The amazon.titan-embed-text-v1 model has been used because it is a dense and semantic model, although a local model could be used, this one has been chosen for demonstration purposes. It also makes use of Sentence Augmented Retrival as an advanced RAG technique

4) metrics.py. This code meets the last requirement and creates a series of metrics in a report: Metrics_results.xlsx

## Credentials

The .env file contains access credentials for a private connection to AWS, these credentials are personal and the user assigned to these credentials has been created temporarily and with limited permissions, only for the recruitment process with STORI, please make responsible use of these credentials without incurring any abuse.

## Contact

Dr. Alberto Beltrán Herrera:

[Linkedin](https://www.linkedin.com/in/dr-alberto-beltran/)



