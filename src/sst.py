import speech_recognition as sr
from rich import print


def get_audio_input(color: str):
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    while(True):
        print("Speak your question (or stay silent to type):")

        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, phrase_time_limit=30)

        try:
            # Always pass a string for language
            text = recognizer.recognize_google(audio, language="en")
            print(f"[{color}][bold]AUDIO INPUT[/bold]: {text}[/{color}]\n")
        
            is_correct: str = input("\033[94mIs the input provided correct? (y-Yes/n-no): \033[0m").strip()
            status: bool = is_correct == '' or is_correct.lower() in ["y", "yes", "true", "1"]

            if not status:
                continue

            if text.lower() in ["exit", "quit"]:
                return None

            return text

        except sr.UnknownValueError as e:
            print("[INFO] Could not understand audio. Please type your question.")
            return None
        except sr.RequestError as e:
            print(f"[ERROR] Speech recognition error: {e}")
            return None
        except Exception as e:
            print(f"[ERROR] Speech recognition error: {e}")
            return None
