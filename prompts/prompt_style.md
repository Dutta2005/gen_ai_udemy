# Prompt Style


## 🦙 Alpaca Prompting: An Introduction

### 🧠 What is Alpaca Prompting?

**Alpaca Prompting** is a lightweight and instruction-following fine-tuning method introduced by **Stanford University** for **instruction-tuned language models**.  
It builds on top of **Meta’s LLaMA (Large Language Model Meta AI)** and focuses on creating a smaller, more accessible, and cost-effective model that can follow human instructions effectively.

In short:  
> Alpaca Prompting teaches a base language model how to understand and follow natural language instructions — just like ChatGPT — but using minimal data and computational resources.

---

### 🧩 The Core Idea

The Stanford team used **OpenAI’s `text-davinci-003`** model to generate **52,000 instruction-following examples** based on a small set of **seed prompts**.  
These examples were then used to fine-tune **LLaMA-7B**, resulting in **Stanford Alpaca** — a model that behaves like a helpful assistant.

| Component | Description |
|------------|--------------|
| **Base Model** | LLaMA-7B (Meta’s foundational model) |
| **Data Source** | Synthetic instruction data generated from `text-davinci-003` |
| **Data Size** | ~52,000 examples |
| **Training Objective** | Fine-tune LLaMA to follow human-like instructions |
| **Goal** | Achieve ChatGPT-style responses at lower cost |

---

### 🏗️ How Alpaca Prompting Works

1. **Seed Prompts Creation**  
   A few high-quality human-written prompts are prepared, such as:
   - Explain how rainbows form.
   - Write a poem about technology.
   - Convert this sentence to passive voice.


2. **Data Expansion using GPT**  
Using OpenAI’s `text-davinci-003`, thousands of new instruction-response pairs are auto-generated based on those seed prompts.

3. **Fine-tuning the Model**  
These generated pairs are used to fine-tune LLaMA.  
The fine-tuned model learns to:
- Follow new, unseen instructions
- Generate coherent, structured responses
- Adapt across multiple domains (coding, reasoning, writing)

4. **Deployment**  
The resulting model — **Alpaca** — is lighter, faster, and can be deployed locally or on consumer GPUs.

---

### 🧮 Example: Alpaca Prompt Template

Each prompt follows a simple, structured **instruction–input–response** format:

```text
### Instruction:
Explain the difference between supervised and unsupervised learning.

### Input:
N/A

### Response:
Supervised learning uses labeled data, where the model learns from input-output pairs.
Unsupervised learning uses unlabeled data, where the model identifies patterns or clusters
without explicit outputs.
```
This consistent structure helps the model understand what the user wants, what context is provided, and how to respond effectively.

### ⚙️ Example Code: Running Alpaca Prompting
You can test Alpaca-style prompting using Hugging Face’s transformers library:
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "chavinlo/alpaca-7b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = """
### Instruction:
Explain the purpose of Docker in modern software development.

### Input:
N/A

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))




## 💬 ChatML Prompting: The Format Behind Modern Chat Models

### 🧠 What is ChatML?

**ChatML (Chat Markup Language)** is the structured message format used by **OpenAI’s chat models** (like GPT-3.5, GPT-4, and GPT-5).  
It defines how conversations are represented — who is speaking, what their role is, and how context is passed to the model.

> Think of ChatML as the “conversation protocol” between you and the AI.

Unlike a plain text prompt, ChatML provides explicit message roles (`system`, `user`, `assistant`, `tool`, etc.), making model behavior **more controllable**, **safe**, and **context-aware**.

---

### 🧩 The Core Structure

A ChatML conversation is an **ordered list of messages**, each containing:
- a **role** (who is speaking)
- **content** (what they said)

Example JSON-like structure:

```json
[
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "Explain the concept of recursion."},
  {"role": "assistant", "content": "Recursion is a method of solving problems where the solution depends on smaller instances of the same problem."}
]
```

### 🧱 Message Roles Explained

| Role                | Description                                                                                     |
| ------------------- | ----------------------------------------------------------------------------------------------- |
| **system**          | Defines the model’s persona, behavior, or constraints. Sets the context for the conversation.   |
| **user**            | The main input — represents the person interacting with the model.                              |
| **assistant**       | The model’s own responses (used for few-shot examples or continuing the chat).                  |
| **tool / function** | (Optional) Used when integrating APIs or external tools (e.g., calculators, retrieval systems). |


### ⚙️ Example: ChatML Prompt in OpenAI API

```python
from openai import OpenAI

client = OpenAI()

messages = [
    {"role": "system", "content": "You are Raj Dutta, a friendly AI full stack developer assistant."},
    {"role": "user", "content": "Can you explain what Docker does?"},
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages
)

print(response.choices[0].message.content)
```

#### ✅ Output Example:
```
Docker lets you package an application and its dependencies into a container so it runs consistently across different environments.
```

### 🧮 Why ChatML Works Better Than Plain Text Prompts
| Feature                  | Plain Prompt                  | ChatML                                         |
| ------------------------ | ----------------------------- | ---------------------------------------------- |
| **Structured roles**     | ❌ No clear speaker separation | ✅ Distinct `system`, `user`, `assistant` roles |
| **Multi-turn memory**    | ❌ Hard to maintain context    | ✅ Each message keeps conversation state        |
| **Persona control**      | ⚠️ Unreliable                 | ✅ Stable via `system` role                     |
| **Safety & consistency** | ⚠️ Limited                    | ✅ Easier to enforce constraints                |
| **Tool integration**     | ❌ Not possible                | ✅ Built-in via `tool` / `function` roles       |


## 🧭 Instruction Prompting (Inst Prompting)

### 🧠 What is Instruction Prompting?

**Instruction Prompting**, often shortened as **Inst Prompting**, is a prompting technique that teaches language models **how to follow human-written instructions** directly.  
Instead of just giving examples or raw text, the user **explicitly instructs** the model on what to do — making it more controllable, reliable, and aligned with human intent.

> 🗣️ “Don’t just predict the next word — understand and follow what I’m asking.”

Instruction prompting is the foundation of models like **InstructGPT**, **Alpaca**, **Flan-T5**, and **ChatGPT**.

---

### 🧩 The Core Idea

Traditional language models (LMs) are trained to **predict the next token**, not to **follow instructions**.  
Instruction Prompting changes this by fine-tuning models on **task-specific instructions** and their corresponding responses.

For example:

| Input (Plain LM) | Input (Instruction Prompting) |
|------------------|-------------------------------|
| "Paris is the capital of" | "Answer the following question: What is the capital of France?" |

This small shift in phrasing helps the model understand **intent**, not just pattern.

---

### 📜 General Format of Instruction Prompting

The structure typically looks like this:

#### Instruction:

<The task or goal>

#### Input:

<Optional supporting data or text>

#### Response:
<The desired output> ```

This template makes it explicit what the model should do, what data it has, and where to place the answer.

### ⚙️ Example: Inst Prompting in Action
#### 🧩 Example 1: Simple Task
```shell
### Instruction:
Summarize the following paragraph.

### Input:
Artificial Intelligence (AI) is a branch of computer science that aims to create systems capable of performing tasks that require human intelligence.

### Response:
AI focuses on building systems that replicate human intelligence for complex tasks.
```

#### 🧮 Example 2: Code Generation
```python
### Instruction:
Write a Python function to reverse a string.

### Input:
N/A

### Response:
def reverse_string(s):
    return s[::-1]
```

#### 🧠 Example 3: Explanation Task
```shell
### Instruction:
Explain the difference between supervised and unsupervised learning.

### Input:
N/A

### Response:
Supervised learning uses labeled data, while unsupervised learning uses unlabeled data to find hidden patterns.
```

### 🧩 Why Instruction Prompting Works
Instruction Prompting aligns model outputs with human expectations because it:

🧠 **Encourages reasoning** — model learns to interpret tasks and plan solutions.

🗣️ **Improves clarity** — explicit instructions reduce ambiguity.

⚙️ **Enables generalization** — model can handle unseen instructions.

🤖 **Makes fine-tuning scalable** — data can be synthetically generated (like in Alpaca)

### 🏗️ Implementation Overview
Instruction Prompting is implemented by fine-tuning a pretrained LM with a dataset of instruction–response pairs.
#### Example dataset entry (JSONL format):
```json
{
  "instruction": "Translate the sentence into Spanish.",
  "input": "Hello, how are you?",
  "output": "Hola, ¿cómo estás?"
}
```
During fine-tuning, the model learns to produce output when given `instruction` and `input`.

### 🧮 Example: Using Instruction Prompting in Python
```python 
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = """
### Instruction:
Translate the following English text to French.

### Input:
Good morning, have a great day!

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### ⚖️ Comparison: Instruction Prompting vs. Chat Prompting

| Feature              | **Instruction Prompting**                                      | **ChatML Prompting**                                             |
| -------------------- | -------------------------------------------------------------- | ---------------------------------------------------------------- |
| **Format**           | Flat, structured text with instruction/input/response sections | Hierarchical messages with roles (`system`, `user`, `assistant`) |
| **Usage**            | Fine-tuning models for task performance                        | Multi-turn conversational models                                 |
| **Focus**            | Task completion                                                | Dialogue understanding                                           |
| **Example Models**   | Flan-T5, Alpaca, Dolly                                         | GPT-4, GPT-5, Claude, Gemini                                     |
| **Context Handling** | Single prompt, limited memory                                  | Maintains full conversation history                              |
| **Style Control**    | Explicitly stated in instruction                               | Controlled via `system` role                                     |

