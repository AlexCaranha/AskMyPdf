import base64
import os
import tempfile


from fastapi import FastAPI
from langchain_core.runnables import RunnableLambda
from langserve import add_routes
from src.classes import PDFUploadRequest

from src.pdf_loader import load_pdf, split_documents
from src.vectorstore import create_vectorstore
from src.qa_chain import build_qa_chain

from dotenv import load_dotenv


load_dotenv()

LLM_LOCAL_ENDPOINT: str = os.getenv("LLM_LOCAL_ENDPOINT")
LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = (
    "Answer briefly and directly. "
    "Only provide essential information. "
    "Add a short justification after your answer."
)


app = FastAPI()
qa_chain = None


def ask_question(question: str) -> str:
    global qa_chain
    if qa_chain is None:
        return "PDF not uploaded yet."

    full_prompt = f"{SYSTEM_PROMPT}\n\n{question}"
    answer = qa_chain.invoke(full_prompt)

    print(f"Question: {question}")
    return answer["result"]


def process_pdf_upload(request: PDFUploadRequest) -> str:
    print("Recebendo PDF via playground...")
    content = base64.b64decode(request.file.encode("utf-8"))

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    docs = load_pdf(tmp_path)

    os.remove(tmp_path)

    chunks = split_documents(docs)
    vectorstore = create_vectorstore(chunks, LLM_LOCAL_ENDPOINT)
    global qa_chain
    qa_chain = build_qa_chain(vectorstore, LLM_MODEL_NAME, LLM_LOCAL_ENDPOINT)

    return "PDF uploaded and vectorstore created."


chain = RunnableLambda(ask_question)
pdf_importer = RunnableLambda(process_pdf_upload).with_types(input_type=PDFUploadRequest)

add_routes(app, chain, path="/AskMyPdf-Chat")
add_routes(app, pdf_importer, path="/AskMyPdf-Upload")

@app.get("/")
def root():
    return {"message": "LangServe server is running!"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8002)
