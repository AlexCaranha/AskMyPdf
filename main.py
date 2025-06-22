from fastapi import Depends, FastAPI, Header, HTTPException
from langchain_core.runnables import RunnableLambda
from typing_extensions import Annotated
from langserve import add_routes


# Autenticação simples via header X-Token
async def verify_token(x_token: Annotated[str, Header()] = "") -> None:
    if x_token != "secret-token":
        raise HTTPException(status_code=400, detail="X-Token header invalid")


app = FastAPI()


# Exemplo de função simples para servir via langserve
def make_a_question(question: str) -> str:
    return question.upper()


chain = RunnableLambda(make_a_question)


add_routes(
    app,
    chain,
    path="/AskMyPdf",
)


@app.get("/")
def root():
    return {"message": "LangServe server is running!"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8002)
