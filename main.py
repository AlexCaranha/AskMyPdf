import os
from langdetect import detect

from dotenv import load_dotenv
from src.pdf_loader import load_pdf, split_documents
from src.qa_chain import build_qa_chain

from src.stt import get_audio_input
from src.translate import translate_to_english
from src.vectorstore import create_vectorstore

from src.util import Category, print_rich


load_dotenv()


LLM_LOCAL_ENDPOINT: str = os.getenv("LLM_LOCAL_ENDPOINT")
LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME")
pdf_path: str = os.getenv("PDF_PATH")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

SYSTEM_PROMPT = (
    "Respond in a polite, concise, direct, and brief manner. "
    "Provide only the information needed to answer the user's question, "
    "avoiding lengthy explanations or unnecessary details. "
    "After your answer, include a brief explanation to justify or clarify your response."
)


def text_or_audio_input():
    while True:
        user_input = input(
            "\033[94mAsk something (or write 'mic' for microphone input): \033[0m"
        )
        user_input = user_input.strip().lower()

        if user_input == "" or user_input == None:
            continue

        if user_input == "exit" or user_input == "quit":
            return "exit"

        # Detect if input is not English and translate
        if user_input != "mic":
            try:
                lang = detect(user_input)
            except Exception as e:
                print_rich(category=Category.SYSTEM, text=e)
                lang = "en"

            if lang != "en":
                translated = translate_to_english(user_input)
                print_rich(category=Category.TRANSLATOR, text=translated)
                continue

        if user_input == "mic":
            user_input = get_audio_input()
            print_rich(category=Category.USER, text=user_input)

            if user_input == "exit" or user_input == "quit":
                return "exit"

        if user_input is not None:
            break

    return user_input


def run():

    print_rich(category=Category.NONE, text=f"Loading PDF: {pdf_path}")
    docs = load_pdf(pdf_path)

    print_rich(category=Category.NONE, text="Splitting into chunks")
    chunks = split_documents(docs)

    print_rich(category=Category.NONE, text="Creating vector store")
    vectorstore = create_vectorstore(chunks, LLM_LOCAL_ENDPOINT)

    print_rich(category=Category.NONE, text="Initializing chatbot")
    qa_chain = build_qa_chain(vectorstore, LLM_MODEL_NAME, LLM_LOCAL_ENDPOINT)

    print_rich(
        category=Category.SYSTEM,
        text="Type your question, or press Enter to use your microphone (say 'exit' or 'quit' to leave)"
    )
    while True:
        user_input = text_or_audio_input()
        if user_input == "exit" or user_input == "quit":
            print_rich(category=Category.SYSTEM, text="AskMyPdf - Exiting")
            break

        full_prompt = f"{SYSTEM_PROMPT}\n\n{user_input}"
        answer = qa_chain.invoke(full_prompt)
        print_rich(category=Category.CHATBOT, text=answer["result"])


if __name__ == "__main__":
    run()
