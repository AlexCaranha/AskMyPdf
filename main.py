from fastapi import FastAPI, UploadFile, File, Form
from langchain_core.runnables import RunnableLambda
from typing_extensions import Annotated
from langserve import add_routes
from src.pdf_loader import load_pdf, split_documents
from src.vectorstore import create_vectorstore
from src.qa_chain import build_qa_chain
import os
import tempfile


app = FastAPI()


# Estado global para armazenar o QA chain por sessão (simples para exemplo)
qa_chain = None


@app.post("/upload_pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    llm_model_name: str = Form(...),
    llm_local_endpoint: str = Form(...),
):
    global qa_chain
    # Salva PDF temporariamente
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    docs = load_pdf(tmp_path)
    chunks = split_documents(docs)
    vectorstore = create_vectorstore(chunks, llm_local_endpoint)
    qa_chain = build_qa_chain(vectorstore, llm_model_name, llm_local_endpoint)
    os.remove(tmp_path)
    return {"message": "PDF uploaded and vectorstore created."}


# Função para responder perguntas usando o QA chain
def ask_question(question: str) -> str:
    global qa_chain
    if qa_chain is None:
        return "PDF not uploaded yet."
    answer = qa_chain.invoke(question)
    return answer["result"]


chain = RunnableLambda(ask_question)


add_routes(app, chain, path="/AskMyPdf")


@app.get("/")
def root():
    return {"message": "LangServe server is running!"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8002)
