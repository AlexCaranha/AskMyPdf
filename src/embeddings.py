import requests


def get_embedding(text, llm_local_endpoint: str):
    url = f"{llm_local_endpoint}/embeddings"
    headers = {"Content-Type": "application/json"}
    data = {"model": "text-embedding-nomic-embed-text-v1.5", "input": text}
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]
