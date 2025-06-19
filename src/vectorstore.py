import faiss
import numpy as np

from langchain.docstore import InMemoryDocstore
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from src.embeddings import get_embedding


def create_vectorstore(chunks, llm_local_endpoint):
    valid_texts = [
        doc.page_content
        for doc in chunks
        if hasattr(doc, "page_content")
        and isinstance(doc.page_content, str)
        and doc.page_content.strip()
    ]
    # print(f"[DEBUG] Number of valid texts: {len(valid_texts)}")
    if not valid_texts:
        raise ValueError("No valid text found for indexing.")

    # Manually generate embeddings
    embeddings = [get_embedding(text, llm_local_endpoint) for text in valid_texts]
    # print(f"[DEBUG] Embedding shape: {len(embeddings[0])}")

    # Manually create FAISS index
    embedding_dim = len(embeddings[0])
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(np.array(embeddings).astype("float32"))

    # Create Document objects
    docs = [Document(page_content=text) for text in valid_texts]

    # Create docstore and index_to_docstore_id
    docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(docs)})
    index_to_docstore_id = {i: str(i) for i in range(len(docs))}

    # Dummy embedding function (will not be used, since we already have the embeddings)
    def dummy_embedding(x):
        return [0.0] * embedding_dim

    # Create the LangChain FAISS vectorstore
    faiss_index = FAISS(
        dummy_embedding,  # embedding_function
        index,
        docstore,
        index_to_docstore_id,
        normalize_L2=False,
    )
    return faiss_index
