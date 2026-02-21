from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
import json
import requests
from pydantic import BaseModel, Field
from typing import Optional
import os

import asyncio
import speech_recognition as sr
from openai.helpers import LocalAudioPlayer

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

def run_command(cmd: str):
    result = os.system(cmd)
    return result

def get_weather(city: str):
    url = f"https://wttr.in/{city.lower()}?format=%C+%t"
    response = requests.get(url)
    if response.status_code == 200:
        return f"The current weather in {city} is: {response.text}"
    return "Sorry, I couldn't fetch the weather information right now."

available_tools = {
    "get_weather": get_weather,
    "run_command": run_command
}


# Chain of Thought Prompting means providing a few examples to the model to help it understand the task better.
SYSTEM_PROMPT = """
 You are an expert AI assistant in resolving user queries using chain of thought.
 You work on START, PLAN, and OUTPUT steps.
 You need to first PLAN what needs to be done. The PLAN can be in multiple steps.
 Once the PLAN is ready, you will start executing the plan step by step followed by giving the OUTPUT.
 The OUTPUT will be the final answer to the user query.
 You can also a tool from the list of tools if needed.
 For every tool call wait for observe step to get the output from the called tool.

 Rules:
    - Strictly follow the given JSON output format.
    - Only run one step at a time.
    - The sequence of steps should be followed.
     START (where user gives an input) -> PLAN (where you plan the steps to be taken) -> OUTPUT (where you give the final answer)

     Output JSON format:
    {"step": "START" | "PLAN" | "OUTPUT", "content": "string"}
    if step == "TOOL":
        {"step": "TOOL", "tool": "string", "input": "string"}
    if step == "OBSERVE":
        {"step": "OBSERVE", "tool": "string", "input": "string", "output": "string"}
    
    Available tools:
    - get_weather: Takes city name as an input string and returns the current weather in that city.
    - run_command: Takes a system linux command string as an input and executes the command on user's system and returns the output from the command.

    Example 1:
    START: Hey can you solve 2 + 3 * 5 / 10?
    PLAN: {"step": "PLAN", "content": "Seems like user is interested in solving a mathematical expression."}
    PLAN: {"step": "PLAN", "content": " I will follow the BODMAS rule to solve this expression step by step."}
    PLAN: {"step": "PLAN", "content": "First I will solve the multiplication and division part of the expression and then I will solve the addition part of the expression."}
    PLAN: {"step": "PLAN", "content": "First we multiply 3 and 5 which gives us 15. Then we divide 15 by 10 which gives us 1.5. Finally we add 2 and 1.5 which gives us 3.5."}
    OUTPUT: {"step": "OUTPUT", "content": "The final answer to the expression 2 + 3 * 5 / 10 is 3.5."}
    
    Example 2:
    START: What is the current weather of Delhi?
    PLAN: {"step": "PLAN", "content": "Seems like user is interested in getting weather information of Delhi in India."}
    PLAN: {"step": "PLAN", "content": "Let's see if we have any available tools to get weather information."}
    PLAN: {"step": "PLAN", "content": "Great, we have get_weather tool available for this query."}
    PLAN: {"step": "PLAN", "content": "I need to call the get_weather tool to get the weather information of Delhi in India."}
    PLAN: {"step": "TOOL", "tool": "get_weather", "input": "delhi"}
    PLAN: {"step": "OBSERVE", "tool": "get_weather", "input": "delhi", "output": "The current weather is cloudy with 27.5°C and 20°F in delhi"}
    OUTPUT: {"step": "OUTPUT", "content": "The current weather of Delhi in India is cloudy with 27.5°C and 20°F."}
    
"""
message_history = [
    {"role": "system", "content": SYSTEM_PROMPT},
]

class MyOutputFormat(BaseModel):
    step: str = Field(..., description="The ID of the step. Example: START, PLAN, OUTPUT, TOOL, OBSERVE")
    content: Optional[str] = Field(None, description="The optional content of the step. Example: The final answer to the user query. The current weather of Delhi in India is cloudy with 27.5°C and 20°F.")
    tool: Optional[str] = Field(None, description="The optional tool name. Example: get_weather")
    input: Optional[str] = Field(None, description="The optional input for the tool. Example: delhi")
    output: Optional[str] = Field(None, description="The optional output from the tool. Example: The current weather is cloudy with 27.5°C and 20°F in delhi")
    
r = sr.Recognizer() # Speech to Text
with sr.Microphone() as source: # Mic access
    r.adjust_for_ambient_noise(source)
    r.pause_threshold = 2

    while True:
        print("Speak something...")
        audio = r.listen(source) # Listen to Mic
        
        print("Processing audio...(STT)")
        user_query = r.recognize_google(audio)
        print("You said: ", user_query)
        message_history.append({"role": "user", "content": user_query})


        while True:
            response = client.chat.completions.parse(
                model="gpt-4.1-mini",
                response_format=MyOutputFormat,
                messages=message_history,
            )
            message_content = response.choices[0].message.content
            # print(message_content)
            message_history.append({"role": "assistant", "content": message_content})
            
            message_json = response.choices[0].message.parsed
            
            if message_json.step == "START":
                print(f"🔥 ", message_json.content)
                continue
            
            elif message_json.step == "TOOL":
                tool_name = message_json.tool
                tool_input = message_json.input
                print(f"🔧 {tool_name} ({tool_input})")
                
                tool_response = available_tools[tool_name](tool_input)
                print(f"🔧 {tool_name} ({tool_input}) = {tool_response}")
                message_history.append({"role": "developer", "content": json.dumps({"step": "OBSERVE", "tool": tool_name, "input": tool_input, "output": tool_response})})
                continue
                
            
            elif message_json.step == "PLAN":
                print(f"🧠 ", message_json.content)
                continue
            
            elif message_json.step == "OUTPUT":
                print(f"🤖 ", message_json.content)
                asyncio.run(tts(speech=message_json.content))
                break
        

