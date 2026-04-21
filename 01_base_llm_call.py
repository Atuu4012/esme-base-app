"""
Lesson 1: Base LLM call.

This script shows the simplest possible Groq chat completion call:
load environment variables, send a system prompt and a user message,
then print the model response.
"""

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Create a Groq client from the API key stored in the environment.
groq_client = Groq()

def simple_call() -> dict:
    """Run one direct LLM request and return the generated text."""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": "You are an expert chef with 15 years of experience working in Michelin star restaurants. Your task is to help user create tasty and easy to prepare reciepies."
            },
            {"role": "user", "content": "Give me 3 meals to prepare for diner."}
        ],
        temperature=0.5
    )

    # The assistant text is stored on the first choice.
    return response.choices[0].message.content

print(simple_call())