# Zero Shot Prompting
import os

from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Zero Shot Prompting means Directly asking the model to perform a task without providing any examples.
SYSTEM_PROMPT = "You are a helpful assistant who is expert in Maths and only and only answers in math related questions nothing else."

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Explain to me how can I do differentiation of x^2 + 2x + 1"
        }
    ]
)

print(response.choices[0].message.content)