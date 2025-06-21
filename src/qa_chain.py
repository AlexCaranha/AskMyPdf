from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI


def build_qa_chain(vectorstore, llm_model_name: str, llm_local_endpoint: str):
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(
        temperature=0,
        model=llm_model_name,           # Use 'model' em vez de 'model_name'
        base_url=llm_local_endpoint,    # Use 'base_url' em vez de 'openai_api_base'
    )
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    return qa
 