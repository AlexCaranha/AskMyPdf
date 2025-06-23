from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

SYSTEM_PROMPT = "Answer briefly and directly. Only provide essential information. Add a short justification after your answer."

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="messages"),
])

def build_qa_chain(vector_store, llm_model_name, llm_local_endpoint):
    llm_model = ChatOpenAI(
        temperature=0,
        model=llm_model_name,
        base_url=llm_local_endpoint,
    )
    retriever = vector_store.as_retriever()

    def add_context_to_messages(inputs):
        # Pega a última mensagem do usuário
        human_message = inputs["messages"][-1]
        question = human_message.content

        # Busca contexto relevante
        context_docs = retriever.invoke(question)
        context = "\n".join(doc.page_content for doc in context_docs)

        # Adiciona o contexto como mensagem do sistema antes do histórico
        messages = [
            {"role": "system", "content": f"Context:\n{context}"},
        ] + inputs["messages"]

        return {"messages": messages}

    chain = add_context_to_messages | prompt | llm_model | StrOutputParser()
    return chain
