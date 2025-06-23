import os
from dotenv import load_dotenv
from langserve import add_routes

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from src.classes import MessageListInput
from src.pdf_loader import load_pdf, split_documents
from src.vectorstore import create_vectorstore
from src.qa_chain import build_qa_chain


load_dotenv()


LLM_LOCAL_ENDPOINT: str = os.getenv("LLM_LOCAL_ENDPOINT")
LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME")
LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME")
PDF_PATH: str = os.getenv("PDF_PATH")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


docs = load_pdf(PDF_PATH)
chunks = split_documents(docs)
vectorstore = create_vectorstore(chunks, LLM_LOCAL_ENDPOINT)
chain = build_qa_chain(vectorstore, LLM_MODEL_NAME, LLM_LOCAL_ENDPOINT)


add_routes(
    app,
    chain.with_types(input_type=MessageListInput),
    path="/chat",
    playground_type="chat"
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8002)
