from dotenv import load_dotenv
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


load_dotenv()

class State(BaseModel):
    messages: Annotated[list, add_messages]

llm = init_chat_model(
    "anthropic:claude-3-5-sonnet-latest"
)

graph_builder = StateGraph(State)

def chatbox(state_provided: State):
    return {"messages": [llm.invoke(state_provided.messages)]}

graph_builder.add_node("chatbox", chatbox)
graph_builder.add_edge(START, "chatbox")
graph_builder.add_edge("chatbox", END)

graph = graph_builder.compile()

user_input = input("Enter your message: ")
initial_state = State(messages=[{"role": "user", "content": user_input}])
response = graph.invoke(initial_state)
print(response["messages"][-1].content)