import os
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Configuração do LLM local (via LM Studio)
LLM_LOCAL_ENDPOINT = "http://localhost:1234/v1"  # ajuste se necessário
LLM_MODEL_NAME = "fastllama-3.2-1b-instruct"  # ou outro nome visível no LM Studio

# Dummy API key (LM Studio ignora)
os.environ["OPENAI_API_KEY"] = "lmstudio"


def load_pdf(file_path):
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)


def create_vectorstore(chunks):
    embeddings = OpenAIEmbeddings(
        openai_api_key="lmstudio",
        openai_api_base=LLM_LOCAL_ENDPOINT,
        model="text-embedding-nomic-embed-text-v1.5",  # modelo de embeddings
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


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

    print("[INFO] Carregando PDF...")
    docs = load_pdf(pdf_path)

    print("[INFO] Dividindo em chunks...")
    chunks = split_documents(docs)

    print("[INFO] Criando base vetorial...")
    vectorstore = create_vectorstore(chunks)

    print("[INFO] Inicializando chatbot...")
    qa_chain = build_qa_chain(vectorstore)

    print("\nDigite sua pergunta (ou 'sair' para encerrar):")
    while True:
        pergunta = input("Pergunta: ")
        if pergunta.lower() in ["sair", "exit", "quit"]:
            break

        resposta = qa_chain.run(pergunta)
        print(f"Resposta: {resposta}\n")
