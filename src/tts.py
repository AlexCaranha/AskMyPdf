import os
import tempfile

from gtts import gTTS
from playsound import playsound


def speak_text(text, lang="en"):
    """Convert text to speech and play it using gTTS and playsound."""
    tts = gTTS(text=text, lang=lang)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        temp_mp3 = fp.name
    tts.save(temp_mp3)
    playsound(temp_mp3)
    os.remove(temp_mp3)
