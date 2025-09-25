# app.py
import streamlit as st
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, function_tool
from dotenv import load_dotenv
import os
from agents.run import RunConfig

# -----------------------------
# 1️⃣ Load API key
# -----------------------------
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not set in environment variables")

# -----------------------------
# 2️⃣ Provider & Model
# -----------------------------
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider
)

config = RunConfig(
    model=model,
    model_provider=provider
)

# -----------------------------
# 3️⃣ Function tool (Executor)
# -----------------------------


# -----------------------------
# 4️⃣ Agent
# -----------------------------
agent = Agent(
    name="Early Disease Detection Assistant",
    instructions="""
You are a health assistant. Predict possible diseases based on symptoms.
Symptoms: {symptoms}
Age: {age}, Gender: {gender}
Provide risk level (low, medium, high) and advice.
"""
)

# -----------------------------
# 5️⃣ Runner
# -----------------------------
user_input = input("Enter a Diseas : ")
runner = Runner.run_sync(agent,user_input,run_config=config)

print(runner.final_output)

