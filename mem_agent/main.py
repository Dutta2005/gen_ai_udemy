from mem0 import Memory
import os
from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()

client = OpenAI()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

config = {
    "version": "v1.1",
    "embedder": {
        "provider": "openai",
        "config": {
            "api_key": OPENAI_API_KEY, "model": "text-embedding-3-small"}
    },
    "llm": {
        "provider": "openai",
        "config": {
            "api_key": OPENAI_API_KEY, "model": "gpt-4.1"}
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": "neo4j+s://5ad951da.databases.neo4j.io?routing=off",
            # "url": "bolt+s://5ad951da.databases.neo4j.io",
            "username": "neo4j",
            "password": "fKGkzwRzvq-iL9AIRgrMpv0rG6JQPqYaPnz_nOuiHis"
        }   
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333
        }
    }
}

mem_client = Memory.from_config(config)

while True:
    user_query = input("> ")
    
    search_memory = mem_client.search(query=user_query, user_id="rajdutta")
    
    memories = [
        f"ID: {mem.get("id")}\nMemory: {mem.get("memory")}" for mem in search_memory.get("results")
    ]
    
    print("Found memories: ", memories)
    
    SYSTEM_PROMPT = f"""
    Here is the context about the user:
    
    Context:
    {json.dumps(memories)}
    """

    client = OpenAI()
        
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query}
        ]
    )

    ai_response = response.choices[0].message.content

    print("AI: ", ai_response)

    mem_client.add(
        user_id="rajdutta",
    messages=[
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": ai_response}
    ]
)

    print("Memory has been synced...")