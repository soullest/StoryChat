import pandas as pd
import numpy as np
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# Cargar el modelo de embeddings
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def get_embedding(text: str) -> np.ndarray:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def calculate_faithfulness(response: str, sources: List[str]) -> float:
    response_embedding = get_embedding(response)
    source_embeddings = [get_embedding(source) for source in sources]
    similarities = [cosine_similarity(response_embedding, source_embedding)[0][0] for source_embedding in source_embeddings]
    return max(similarities)

def calculate_relevance(query: str, response: str) -> float:
    query_embedding = get_embedding(query)
    response_embedding = get_embedding(response)
    similarity = cosine_similarity(query_embedding, response_embedding)[0][0]
    return similarity

def calculate_correctness(response: str, expected_answer: str) -> str:
    response_embedding = get_embedding(response)
    expected_embedding = get_embedding(expected_answer)
    similarity = cosine_similarity(response_embedding, expected_embedding)[0][0]
    return "SI" if similarity >= 0.6 else "NO"

def evaluate_response(query: str, response: str, expected_answer: str, sources: List[str]) -> Tuple[float, float, str]:

    faithfulness = calculate_faithfulness(response, sources)
    relevance = calculate_relevance(query, response)
    correctness = calculate_correctness(response, expected_answer)
    return faithfulness, relevance, correctness


def evaluate_chatbot_responses(df: pd.DataFrame) -> pd.DataFrame:
    results = {
        "Prompt": [],
        "Response": [],
        "Expected Answer": [],
        "Faithfulness": [],
        "Relevance": [],
        "Correctness": []
    }

    for idx, row in df.iterrows():
        query = row["Prompt"]
        expected_answer = row["Expected Answer"]
        response = row["Response"]
        sources = [row["Source_1"], row["Source_2"], row["Source_3"]]

        # Evaluar las métricas
        faithfulness, relevance, correctness = evaluate_response(query, response, expected_answer, sources)

        # Almacenar los resultados
        results["Prompt"].append(query)
        results["Response"].append(response)
        results["Expected Answer"].append(expected_answer)
        results["Faithfulness"].append(faithfulness)
        results["Relevance"].append(relevance)
        results["Correctness"].append(correctness)

    # Crear DataFrame de resultados
    results_df = pd.DataFrame(results)
    return results_df


def run_tests():
    file_path = "metrics.xlsx"
    df = pd.read_excel(file_path)
    results_df = evaluate_chatbot_responses(df)
    print(results_df)
    results_df.to_excel('Metrics_Results.xlsx', index=False, engine="openpyxl")


if __name__ == "__main__":
    run_tests()
