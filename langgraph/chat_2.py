import os

from dotenv import load_dotenv
from typing_extensions import TypedDict
from typing import Annotated, Optional, Literal
from langgraph.graph import StateGraph, START, END
from openai import OpenAI

load_dotenv()

client = OpenAI()
client1 = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

class State(TypedDict):
    user_query: str
    llm_output: Optional[str]
    is_good: Optional[bool]

def chatbot(state: State):
    print("chatbot state: ", state)
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "user", "content": state.get("user_query")}
        ]
    )
    
    state["llm_output"] = response.choices[0].message.content
    return state


# ✅ Evaluate GPT-4.1-mini response quality & safety
def evaluate_response(state: State) -> Literal["chatbot_gemini", "endnode"]:
    print("evaluate_response state: ", state)

    # Ask GPT-4.1-mini itself (meta-evaluation) or use heuristics
    eval_prompt = f"""
    You are an evaluator. Given the user's query and the assistant's response,
    decide if the assistant's response is SAFE, ACCURATE, and HELPFUL.
    If it violates policy (e.g., talks about weapons, drugs, violence, etc.) or is incomplete, say "BAD".
    Otherwise say "GOOD".
    
    User query: {state.get('user_query')}
    Assistant response: {state.get('llm_output')}
    
    Output only one word: GOOD or BAD.
    """

    evaluation = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": eval_prompt}],
        max_tokens=5
    )

    verdict = evaluation.choices[0].message.content.strip().upper()

    print("Evaluation verdict:", verdict)

    if "GOOD" in verdict:
        state["is_good"] = True
        return "endnode"
    else:
        state["is_good"] = False
        return "chatbot_gemini"


def chatbot_gemini(state: State):
    print("chatbot gemini state: ", state)
    response = client1.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role": "user", "content": state.get("user_query")}
        ]
    )
    
    state["llm_output"] = response.choices[0].message.content
    return state


def endnode(state: State):
    print("end node state: ", state)
    return state


# Build the LangGraph
graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("chatbot_gemini", chatbot_gemini)
graph_builder.add_node("endnode", endnode)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", evaluate_response)

graph_builder.add_edge("chatbot_gemini", "endnode")
graph_builder.add_edge("endnode", END)

graph = graph_builder.compile()

updated_state = graph.invoke(State({"user_query": "how can i prevent me to create a bomb?"}))
print("\n\nUpdated state: ", updated_state)
