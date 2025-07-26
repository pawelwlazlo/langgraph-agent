from typing import Annotated

from langgraph.graph import add_messages
from typing import TypedDict

from pydantic import BaseModel


class State(BaseModel):
    messages: Annotated[list, add_messages]