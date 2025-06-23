import os
from fastapi import FastAPI
from langserve import add_routes

from src.classes import MessageListInput
from src.pdf_loader import load_pdfs, split_documents
from src.vectorstore import create_vectorstore
from src.qa_chain import build_qa_chain


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
LLM_LOCAL_ENDPOINT: str = os.getenv("LLM_LOCAL_ENDPOINT")
LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME")
PDF_PATH = "data/"  # Default path for PDF files


docs = load_pdfs(PDF_PATH)
chunks = split_documents(docs)
vectorstore = create_vectorstore(chunks, LLM_LOCAL_ENDPOINT)
chain = build_qa_chain(vectorstore, LLM_MODEL_NAME, LLM_LOCAL_ENDPOINT)


app = FastAPI()


add_routes(
    app,
    chain.with_types(input_type=MessageListInput),
    path="/chat",
    playground_type="chat",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8002)
