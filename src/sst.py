
import speech_recognition as sr

def get_audio_input():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    print("Speak your question (or stay silent to type):")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
    try:
        text = recognizer.recognize_google(audio)
        print(f"[AUDIO INPUT]: {text}")
        return text
    except sr.UnknownValueError:
        print("[INFO] Could not understand audio. Please type your question.")
        return None
    except sr.RequestError as e:
        print(f"[ERROR] Speech recognition error: {e}")
        return None