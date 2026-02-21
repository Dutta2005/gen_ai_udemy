# Few Shot Prompting
# Zero Shot Prompting
import os

from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Few Shot Prompting means providing a few examples to the model to help it understand the task better.
SYSTEM_PROMPT = """You are a helpful assistant who is expert in coding and only and only answers in coding related questions nothing else.

Rule:
- You should only answer coding related questions.
- Strictly follow the output in JSON format.

Output format:
{{
"code": "string" or None,
"isCodingRelated": true or false,
}}

Examples:
User: Write a Python function to calculate the factorial of a number.
Assistant: {{"code": "def factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    else:\n        return n * factorial(n - 1)", "isCodingRelated": true}}
User: Can you explain the a + b whole squared formula?
Assistant: {{"code": None, "isCodingRelated": false}}

"""

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "How can we manage 1M rows of data in relational database and also optimise query performance?"
        }
    ]
)

print(response.choices[0].message.content)