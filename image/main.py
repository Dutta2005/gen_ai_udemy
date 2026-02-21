from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Genrate a caption for the given image"},
                {"type": "image_url", "image_url": {"url": "https://images.pexels.com/photos/577585/pexels-photo-577585.jpeg"}}
            ] 
        }
    ]
)

print("Response: ", response.choices[0].message.content)