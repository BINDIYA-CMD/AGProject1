"""
DEMO 2: Implementing Chain-of-Thought (CoT) Reasoning Agent
Module 1: LLM Foundations & Prompt Engineering
---------------------------------------------------------
Goal: Demonstrate how to use CoT prompting to improve LLM reasoning.
Learners will see:
1. The difference between a direct response and a CoT response.
2. How to structure a System Prompt to enforce "Reasoning Steps."
3. How to implement the "Thinking" pattern in a script.
"""

from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from a local .env file (if present)
load_dotenv()

# Read OPENAI_API_KEY from environment
env_openai_api_key = os.getenv("OPENAI_API_KEY", "")

# --- INITIALIZATION ---
# Initialize OpenAI client using the OPENAI_API_KEY from environment (.env)
if not env_openai_api_key:
    raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env or environment variables.")

client = OpenAI(api_key=env_openai_api_key)

def run_cot_demo(user_query):
    print(f"\n{'='*60}")
    print(f"USER QUERY: {user_query}")
    print(f"{'='*60}\n")

    # --- PART 1: ZERO-SHOT (DIRECT) PROMPTING ---
    # This often fails with complex logic as the model jumps to a conclusion.
    print("--- 1. DIRECT RESPONSE (No Reasoning) ---")
    direct_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Answer the question directly and briefly."},
            {"role": "user", "content": user_query}
        ]
    )
    print(f"Result: {direct_response.choices[0].message.content}\n")

    # --- PART 2: CHAIN-OF-THOUGHT (CoT) PROMPTING ---
    # We use a System Prompt to define a persona that MUST reason step-by-step.
    print("--- 2. CHAIN-OF-THOUGHT RESPONSE (Step-by-Step) ---")
    
    cot_system_prompt = (
        "You are a reasoning agent. When given a problem, you must "
        "always follow this structure:\n"
        "1. THOUGHT: Break down the logic and steps required.\n"
        "2. CALCULATION: Perform any necessary math or comparisons.\n"
        "3. FINAL ANSWER: Provide the concluding result based on the thoughts."
    )

    cot_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": cot_system_prompt},
            {"role": "user", "content": user_query}
        ],
        temperature=0  # Keeping temperature low for consistent logic
    )
    
    print(cot_response.choices[0].message.content)

# --- EXAMPLE PROBLEM ---
# A classic logic puzzle that requires multi-step deduction
complex_problem = (
    "I have 3 apples. I give 1 to my neighbor. My neighbor gives it back "
    "to me along with 2 oranges. Then I eat 1 apple and give 1 orange to "
    "my son. How many fruits do I have left, and what are they?"
)

if __name__ == "__main__":
    run_cot_demo(complex_problem)