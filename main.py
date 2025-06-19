import os

from dotenv import load_dotenv
from src.pdf_loader import load_pdf, split_documents
from src.qa_chain import build_qa_chain
from src.tts import speak_text
from src.sst import get_audio_input
from src.translate import translate_to_english
from src.vectorstore import create_vectorstore
from rich import print
from langdetect import detect


load_dotenv()  # Carrega as variáveis do .env


LLM_LOCAL_ENDPOINT: str = os.getenv("LLM_LOCAL_ENDPOINT")
LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME")
pdf_path: str = os.getenv("PDF_PATH")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def text_or_audio_input():
    while True:
        user_input = input(
            "\033[94mAsk something (or press Enter for audio): \033[0m"
        ).strip()  # Blue prompt

        if not user_input:
            # Try audio input
            user_input = get_audio_input()

        if user_input is not None:
            break

    # Detect if input is not English and translate
    if user_input.strip():
        try:
            lang = detect(user_input)
        except Exception as e:
            print(f"[red]Language detection error: {e}[/red]")
            lang = "en"

        if lang != "en":
            translated = translate_to_english(user_input)
            print(f"[blue]Detected language:[/blue] {lang}")
            print(f"[yellow]Translated to English:[/yellow] [bold]{translated}[/bold]")
            user_input = translated

    return user_input


def text_and_audio_output(text):
    print(f"[green]Answer: {text}[/green]\n")  # Green answer
    speak_text(text)


if __name__ == "__main__":
    print(f"[INFO] Loading PDF: {pdf_path}")
    docs = load_pdf(pdf_path)

    print("[INFO] Splitting into chunks...")
    chunks = split_documents(docs)

    print("[INFO] Creating vector store...")
    vectorstore = create_vectorstore(chunks, LLM_LOCAL_ENDPOINT)

    print("[INFO] Initializing chatbot...")
    qa_chain = build_qa_chain(vectorstore, LLM_MODEL_NAME, LLM_LOCAL_ENDPOINT)

    print(
        "Type your question, or press Enter to use your microphone (say 'exit' or 'quit' to leave):"
    )
    while True:
        user_input = text_or_audio_input()
        if user_input.lower() in ["exit", "quit"]:
            break

        answer = qa_chain.invoke(user_input)  # Use invoke no lugar de run
        text_and_audio_output(answer["result"])
