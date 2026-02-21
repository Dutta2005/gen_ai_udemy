import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {"role": "system", "content": "You are a helpful assistant who is expert in Maths and only and only answers in math related questions nothing else."},
        {
            "role": "user",
            "content": "Explain to me how can I do differentiation of x^2 + 2x + 1"
        }
    ]
)

print(response.choices[0].message.content)