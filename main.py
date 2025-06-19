import os

from src.pdf_loader import load_pdf, split_documents
from src.qa_chain import build_qa_chain
from src.tts import speak_text
from src.vectorstore import create_vectorstore


# Local LLM configuration (via LM Studio)
LLM_LOCAL_ENDPOINT = "http://127.0.0.1:1234/v1"  # adjust if necessary
LLM_MODEL_NAME = "fastllama-3.2-1b-instruct"  # or another name visible in LM Studio
pdf_path = "data\The-Skalunda-Giant.pdf"

# Dummy API key (LM Studio ignores)
os.environ["OPENAI_API_KEY"] = "lmstudio"

if __name__ == "__main__":
    print("[INFO] Loading PDF...")
    docs = load_pdf(pdf_path)

    print("[INFO] Splitting into chunks...")
    chunks = split_documents(docs)

    print("[INFO] Creating vector store...")
    vectorstore = create_vectorstore(chunks, LLM_LOCAL_ENDPOINT)

    print("[INFO] Initializing chatbot...")
    qa_chain = build_qa_chain(vectorstore, LLM_MODEL_NAME, LLM_LOCAL_ENDPOINT)

    print("\nType your question (or 'exit' to exit):")
    while True:
        question = input("Ask something: ")
        if question.lower() in ["exit", "quit"]:
            break

        answer = qa_chain.run(question)
        print(f"Answer: {answer}\n")
        speak_text(answer, lang="en")
