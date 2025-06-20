import faiss
import numpy as np

from langchain.docstore import InMemoryDocstore
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from src.embeddings import get_embedding


class CustomEmbeddings(Embeddings):
    def __init__(self, llm_local_endpoint, embedding_dim):
        self.llm_local_endpoint = llm_local_endpoint
        self.embedding_dim = embedding_dim

    def embed_documents(self, texts):
        return [get_embedding(text, self.llm_local_endpoint) for text in texts]

    def embed_query(self, text):
        return get_embedding(text, self.llm_local_endpoint)


def create_vectorstore(chunks, llm_local_endpoint):
    valid_texts = [
        doc.page_content
        for doc in chunks
        if hasattr(doc, "page_content")
        and isinstance(doc.page_content, str)
        and doc.page_content.strip()
    ]
    if not valid_texts:
        raise ValueError("No valid text found for indexing.")

    # Manually generate embeddings
    embeddings = [get_embedding(text, llm_local_endpoint) for text in valid_texts]

    # Manually create FAISS index
    embedding_dim = len(embeddings[0])
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(np.array(embeddings).astype("float32"))

    # Create Document objects
    docs = [Document(page_content=text) for text in valid_texts]

    # Create docstore and index_to_docstore_id
    docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(docs)})
    index_to_docstore_id = {i: str(i) for i in range(len(docs))}

    # Use a proper Embeddings object
    embedding_obj = CustomEmbeddings(llm_local_endpoint, embedding_dim)

    # Create the LangChain FAISS vectorstore
    faiss_index = FAISS(
        embedding_obj,  # embedding_function as Embeddings object
        index,
        docstore,
        index_to_docstore_id,
        normalize_L2=False,
    )
    return faiss_index
