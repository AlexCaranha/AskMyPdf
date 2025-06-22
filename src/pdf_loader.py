import string
import os

from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


LLM_LOCAL_ENDPOINT: str = os.getenv("LLM_LOCAL_ENDPOINT")
LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME")
pdf_path: str = os.getenv("PDF_PATH")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def load_pdf(file_path):
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)


def get_full_text_from_pdf_file(documents):
    if not documents:
        return ""

    full_text = "\n".join(doc.page_content for doc in documents)
    full_text = "".join(c for c in full_text if c in string.printable)
    full_text = full_text.replace(" \n", " ")  # Remove quebras de linha

    return full_text
