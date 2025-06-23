import requests
import os


LLM_MODEL_EMBEDDING_NAME: str = os.getenv("LLM_MODEL_EMBEDDING_NAME")


def get_embedding(text, llm_local_endpoint: str):
    url = f"{llm_local_endpoint}/embeddings"
    headers = {"Content-Type": "application/json"}
    data = {"model": LLM_MODEL_EMBEDDING_NAME, "input": text}
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]
