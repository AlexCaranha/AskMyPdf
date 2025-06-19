# import os
# import tempfile

# from gtts import gTTS
# from playsound import playsound


# def speak_text(text, lang="en"):
#     """Convert text to speech and play it using gTTS and playsound."""
#     tts = gTTS(text=text, lang=lang)
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
#         temp_mp3 = fp.name
#     tts.save(temp_mp3)
#     playsound(temp_mp3)
#     os.remove(temp_mp3)


import pyttsx3


def setup():
    engine = pyttsx3.init()

    # Set voice to English if available
    for voice in engine.getProperty('voices'):
        if (len(voice.languages) > 0 and "en" in voice.languages[0].decode() or "English" in voice.name):
            engine.setProperty('voice', voice.id)
            break

    return engine


def speak_text(text, lang="en"):
    engine.say(text)
    engine.runAndWait()


engine = setup()
