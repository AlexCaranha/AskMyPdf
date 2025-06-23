from pydantic import BaseModel
from typing import List, Dict, Any, Union
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


class MessageListInput(BaseModel):
    """Input for the chat endpoint."""

    messages: List[Union[HumanMessage, AIMessage]] = Field(
        ...,
        description="The chat messages representing the current conversation.",
        extra={"widget": {"type": "chat", "input": "messages"}},
    )
