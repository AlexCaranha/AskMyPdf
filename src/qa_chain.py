
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI


def build_qa_chain(vectorstore, llm_model_name: str, llm_local_endpoint: str):
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(
        temperature=0,
        model_name=llm_model_name,
        openai_api_key="lmstudio",
        openai_api_base=llm_local_endpoint,
    )
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    return qa