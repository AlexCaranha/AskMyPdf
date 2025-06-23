import string
import os

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_pdfs(pdf_dir: str):
    if not pdf_dir or not os.path.isdir(pdf_dir):
        raise FileNotFoundError(
            f"Invalid PDF_PATH: '{pdf_dir}'. Provide a valid directory containing PDF files."
        )

    pdf_files = [
        os.path.join(pdf_dir, f)
        for f in os.listdir(pdf_dir)
        if f.lower().endswith(".pdf")
    ]
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in directory '{pdf_dir}'.")

    documents = []
    for pdf_file in pdf_files:
        print(f"Reading PDF file: {pdf_file}")
        loader = PyMuPDFLoader(pdf_file)
        documents.extend(loader.load())

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
