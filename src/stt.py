import os
import whisper
import time

import pyaudio
import wave
import keyboard
import threading

from src.util import Category, print_rich


# models: "base", "base.en", "tiny.en"
model = whisper.load_model("tiny.en")

# Configurações de áudio
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # 16 kHz é bom para voz
CHUNK = 1024
OUTPUT_FILENAME = "audio_input.wav"
MAX_DURATION = 30  # segundos

recording = False


def gravar_audio():
    global recording

    audio = pyaudio.PyAudio()

    stream = audio.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
    )

    print_rich(
        category=Category.SYSTEM,
        text="Gravando... Pressione 'enter' para parar ou aguarde 30 segundos.",
    )
    frames = []
    start_time = time.time()

    while recording and (time.time() - start_time < MAX_DURATION):
        data = stream.read(CHUNK)
        frames.append(data)

    print_rich(category=Category.SYSTEM, text="Parando gravação...")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(OUTPUT_FILENAME, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))


def record_audio_from_microphone():
    global recording
    recording = False

    print_rich(
        category=Category.SYSTEM,
        text="Say your question, then press '[blue]Enter[/blue]' to end the recording:",
    )
    if not recording:
        recording = True
        thread = threading.Thread(target=gravar_audio)
        thread.start()

    keyboard.wait("enter")
    if recording:
        recording = False
        thread.join()

    return OUTPUT_FILENAME


def get_audio_input() -> str:
    record_audio_from_microphone()

    model_result = model.transcribe(OUTPUT_FILENAME)
    text = model_result.get("text", None)

    if os.path.exists(OUTPUT_FILENAME):
        os.remove(OUTPUT_FILENAME)

    return text
