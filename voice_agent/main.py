import asyncio

import speech_recognition as sr
from openai import OpenAI, AsyncOpenAI
from openai.helpers import LocalAudioPlayer
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

async_client = AsyncOpenAI()

async def tts(speech: str):
    async with async_client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="nova",
        instructions="Always speak in cheerful manner with full of delight and happy.",
        input=speech,
        response_format="pcm"
    ) as response:
        await LocalAudioPlayer().play(response)

def main():
    r = sr.Recognizer() # Speech to Text
    with sr.Microphone() as source: # Mic access
        r.adjust_for_ambient_noise(source)
        r.pause_threshold = 2
        
        SYSTEM_PROMPT = f"""
                You are an expert voice agent. You are given the transcript of wat user has said using voice.
                You need to output as if you are an voice agent and whatever you speak will be converted back to audio using AI and played back to user. 
            """
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
            ]
        
        while True:
            print("Speak something...")
            audio = r.listen(source) # Listen to Mic
            
            print("Processing audio...(STT)")
            text = r.recognize_google(audio)
            print(f"You said: {text}")
            
            messages.append({"role": "user", "content": text})
            
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages
            )
            
            print("AI response: ", response.choices[0].message.content)
            asyncio.run(tts(speech=response.choices[0].message.content))
            
main()