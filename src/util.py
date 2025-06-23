from rich import print
from enum import Enum


class Category(Enum):
    ASKMYPDF = 1
    SYSTEM = 2
    USER = 3
    TRANSLATOR = 3
    CHATBOT = 4


def print_rich(category: Category, text: str):

    color = get_category_color(category)

    if category == Category.ASKMYPDF:
        print(f"[{color}][bold]AskMyPDF[/bold]: {text}[/{color}]\n")
    else:
        print(f"[{color}][bold]{category.name}[/bold]: {text}[/{color}]\n")


def get_category_color(category: Category) -> str:
    category_colors = {
        Category.SYSTEM: "yellow",
        Category.TRANSLATOR: "yellow",
        Category.CHATBOT: "green",
        Category.USER: "blue",
        Category.ASKMYPDF: "green",
    }
    return category_colors.get(category, "white")
