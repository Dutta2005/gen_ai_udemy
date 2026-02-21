import os

from openai import OpenAI
import json

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Chain of Thought Prompting means providing a few examples to the model to help it understand the task better.
SYSTEM_PROMPT = """
 You are an expert AI assistant in resolving user queries using chain of thought.
 You work on START, PLAN, and OUTPUT steps.
 You need to first PLAN what needs to be done. The PLAN can be in multiple steps.
 Once the PLAN is ready, you will start executing the plan step by step followed by giving the OUTPUT.
 The OUTPUT will be the final answer to the user query.

 Rules:
    - Strictly follow the given JSON output format.
    - Only run one step at a time.
    - The sequence of steps should be followed.
     START (where user gives an input) -> PLAN (where you plan the steps to be taken) -> OUTPUT (where you give the final answer)

     Output JSON format:
    {{
        "step": "string", // can be START, PLAN, or OUTPUT
        "plan": "string" or None, // can be multiple steps in string format or None
        "content": "string" or None // final output to the user or None
    }}

    Examples:
    START: Hey can you solve 2 + 3 * 5 / 10?
    PLAN: {"step": "PLAN", "content": "Seems like user is interested in solving a mathematical expression."}
    PLAN: {"step": "PLAN", "content": " I will follow the BODMAS rule to solve this expression step by step."}
    PLAN: {"step": "PLAN", "content": "First I will solve the multiplication and division part of the expression and then I will solve the addition part of the expression."}
    PLAN: {"step": "PLAN", "content": "First we multiply 3 and 5 which gives us 15. Then we divide 15 by 10 which gives us 1.5. Finally we add 2 and 1.5 which gives us 3.5."}
    OUTPUT: {"step": "OUTPUT", "content": "The final answer to the expression 2 + 3 * 5 / 10 is 3.5."}
    
"""
message_history = [
    {"role": "system", "content": SYSTEM_PROMPT},
]

print("\n\n\n")

user_query = input("👤 ")
message_history.append({"role": "user", "content": user_query})

while True:
    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        response_format={"type": "json_object"},
        messages=message_history
    )
    message_content = response.choices[0].message.content
    # print(message_content)
    message_history.append({"role": "assistant", "content": message_content})
    message_json = json.loads(message_content)
    if message_json["step"] == "START":
        print(f"🔥 ", message_json["content"])
        continue
    elif message_json["step"] == "PLAN":
        print(f"🧠 ", message_json["plan"])
        continue
    elif message_json["step"] == "OUTPUT":
        print(f"🤖 ", message_json["content"])
        break

print("\n\n\n")
