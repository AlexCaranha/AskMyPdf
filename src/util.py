from rich import print
from src.tts import speak_text
from enum import Enum


class Category(Enum):
    NONE = 1
    SYSTEM = 2
    USER = 3
    TRANSLATOR = 3
    CHATBOT = 4


def print_rich(category: Category, text: str):

    color = get_category_color(category)

    if category is None:
        print(f"[{color}]{text}[/{color}]\n")
    else:
        print(f"[{color}][bold]{category}[/bold]: {text}[/{color}]\n")

    speak = category == Category.CHATBOT or category == Category.USER
    if speak:
        speak_text(text)


def get_category_color(category: Category) -> str:
    category_colors = {
        Category.SYSTEM: "yellow",
        Category.TRANSLATOR: "yellow",
        Category.CHATBOT: "green",
        Category.USER: "blue",
        Category.NONE: "white",
    }
    return category_colors.get(category, "white")
