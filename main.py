import os
import faiss
import requests
import numpy as np
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore

# Local LLM configuration (via LM Studio)
LLM_LOCAL_ENDPOINT = "http://127.0.0.1:1234/v1"  # adjust if necessary
LLM_MODEL_NAME = "fastllama-3.2-1b-instruct"  # or another name visible in LM Studio

# Dummy API key (LM Studio ignores)
os.environ["OPENAI_API_KEY"] = "lmstudio"


def load_pdf(file_path):
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)


def get_embedding(text):
    url = f"{LLM_LOCAL_ENDPOINT}/embeddings"
    headers = {"Content-Type": "application/json"}
    data = {"model": "text-embedding-nomic-embed-text-v1.5", "input": text}
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]


def create_vectorstore(chunks):
    valid_texts = [
        doc.page_content
        for doc in chunks
        if hasattr(doc, "page_content")
        and isinstance(doc.page_content, str)
        and doc.page_content.strip()
    ]
    print(f"[DEBUG] Number of valid texts: {len(valid_texts)}")
    if not valid_texts:
        raise ValueError("No valid text found for indexing.")

    # Manually generate embeddings
    embeddings = [get_embedding(text) for text in valid_texts]
    print(f"[DEBUG] Embedding shape: {len(embeddings[0])}")

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


def build_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(
        temperature=0,
        model_name=LLM_MODEL_NAME,
        openai_api_key="lmstudio",
        openai_api_base=LLM_LOCAL_ENDPOINT,
    )
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    return qa


if __name__ == "__main__":
    pdf_path = "The-Skalunda-Giant.pdf"

    print("[INFO] Loading PDF...")
    docs = load_pdf(pdf_path)

    print("[INFO] Splitting into chunks...")
    chunks = split_documents(docs)

    print("[INFO] Creating vector store...")
    vectorstore = create_vectorstore(chunks)

    print("[INFO] Initializing chatbot...")
    qa_chain = build_qa_chain(vectorstore)

    print("\nType your question (or 'exit' to exit):")
    while True:
        question = input("Ask something: ")
        if question.lower() in ["exit", "quit"]:
            break

        answer = qa_chain.run(question)
        print(f"Answer: {answer}\n")
