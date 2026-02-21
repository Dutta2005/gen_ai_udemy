from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langgraph.checkpoint.mongodb import MongoDBSaver

load_dotenv()

llm = init_chat_model(
    model="gpt-4.1-mini",  # Fixed model name
    model_provider="openai",
)

class State(TypedDict):
    messages: Annotated[list, add_messages]
    
def chatbot(state: State):
    response = llm.invoke(state.get("messages"))
    return {"messages": [response]}

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

def compile_graph_with_checkpointer(checkpointer):
    return graph_builder.compile(checkpointer=checkpointer)

# Try different connection strings:
# Option 1:
# DB_URI = "mongodb://admin:admin@localhost:27017"

# Option 2 (if Option 1 fails):
# DB_URI = "mongodb://admin:admin@localhost:27017/admin"

# Option 3 (development only - no auth):
DB_URI = "mongodb://localhost:27017"

try:
    with MongoDBSaver.from_conn_string(DB_URI) as checkpointer:
        graph_with_checkpointer = compile_graph_with_checkpointer(checkpointer)

        config = {
            "configurable": {
                "thread_id": "raj"
            }
        }

        for chunk in graph_with_checkpointer.stream(
            {"messages": ["Can you suggest me what should I have on dinner?"]},
            config,
            stream_mode="values"
        ):
            chunk["messages"][-1].pretty_print()
except Exception as e:
    print(f"Error: {e}")
    print("\nTroubleshooting tips:")
    print("1. Make sure MongoDB container is running: docker ps")
    print("2. Check MongoDB logs: docker logs <container-name>")
    print("3. Test direct connection with mongosh or Python")