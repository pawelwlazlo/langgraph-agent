from dotenv import load_dotenv
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()


class MessageClassifier(BaseModel):
    message_type: Literal["emotional", "logical"] = Field(
        ..., description="Classify if the message requieres an emotional or logical response.."
    )


class State(BaseModel):
    messages: Annotated[list, add_messages]
    message_type: str | None
    next: str | None


llm = init_chat_model(
    "anthropic:claude-3-5-sonnet-latest"
)

graph_builder = StateGraph(State)


def classify_message(state: State):
    last_message = state.messages[-1]
    classifier_llm = llm.with_structured_output(MessageClassifier)

    result = classifier_llm.invoke([
        {
            "role": "system",
            "content": """Classify the user message as either:
            - 'emotional': if it asks for emotional support, therapy, deals with feelings, or personal problems
            - 'logical': if it asks for facts, information, logical analysis, or practical solutions
            """
        },
        {"role": "user", "content": last_message.content}
    ])
    return {"message_type": result.message_type}


def router(state: State):
    message_type = state.message_type if state.message_type is not None else "logical"
    if message_type == "emotional":
        return {"next": "therapist"}

    return {"next": "logical"}


def therapist_agent(state: State):
    last_message = state.messages[-1]

    messages = [
        {"role": "system",
         "content": """You are a compassionate therapist. Focus on the emotional aspects of the user's message.
                        Show empathy, validate their feelings, and help them process their emotions.
                        Ask thoughtful questions to help them explore their feelings more deeply.
                        Avoid giving logical solutions unless explicitly asked."""
         },
        {
            "role": "user",
            "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}


def logical_agent(state: State):
    last_message = state.messages[-1]

    messages = [
        {"role": "system",
         "content": """You are a purely logical assistant. Focus only on facts and information.
            Provide clear, concise answers based on logic and evidence.
            Do not address emotions or provide emotional support.
            Be direct and straightforward in your responses."""
         },
        {
            "role": "user",
            "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}


graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("router", router)
graph_builder.add_node("therapist", therapist_agent)
graph_builder.add_node("logical", logical_agent)

graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "router")

graph_builder.add_conditional_edges(
    "router",
    lambda state: state.next if state.next is not None else "logical",
    {"therapist": "therapist", "logical": "logical"}
)

graph_builder.add_edge("therapist", END)
graph_builder.add_edge("logical", END)

graph = graph_builder.compile()


def run_chatbot():
    state = State(messages=[], message_type=None, next=None)

    while True:
        user_input = input("Message: ")
        if user_input == "exit":
            print("Bye")
            break

        new_messages = state.messages + [{"role": "user", "content": user_input}]
        state = State(messages=new_messages, message_type=state.message_type, next=state.next)

        # graph.invoke zwraca słownik
        result = graph.invoke(state)

        if "messages" in result and len(result["messages"]) > 0:
            last_message = result["messages"][-1]
            print(f"Assistant: {last_message.content}")

        state = State(
            messages=result.get("messages", []),
            message_type=result.get("message_type"),
            next=result.get("next")
        )

if __name__ == "__main__":
    run_chatbot()
