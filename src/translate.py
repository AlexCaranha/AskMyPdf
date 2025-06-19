
from deep_translator import GoogleTranslator


def translate_to_english(text):
    translator = GoogleTranslator(source="auto", target="en")
    result = translator.translate(text)
    return result
