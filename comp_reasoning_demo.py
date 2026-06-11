import os
from pdb import run
import time
import pandas as pd
from dotenv import load_dotenv

from openai import OpenAI
from anthropic import Anthropic
from google import genai

from langsmith import Client
from langsmith.evaluation import evaluate

load_dotenv()

# ====================================================

# CLIENTS

# ====================================================

openai_client = OpenAI(
api_key=os.getenv("OPENAI_API_KEY")
)

claude_client = Anthropic(
api_key=os.getenv("ANTHROPIC_API_KEY")
)

gemini_client = genai.Client(
api_key=os.getenv("GOOGLE_API_KEY")
)

langsmith_client = Client()

# ====================================================

# DATASET

# ====================================================

DATASET_NAME = "CoT Reasoning Benchmark"

QUESTIONS = [
{
"question": "I have 3 apples. I give 1 to my neighbor. My neighbor gives it back to me along with 2 oranges. Then I eat 1 apple and give 1 orange to my son. How many fruits do I have left, and what are they?",
"answer": "1 apple and 1 orange"
},
{
"question": "A train travels 120 km in 2 hours. It then travels another 180 km in 3 hours. What is the average speed for the entire journey?",
"answer": "60 km/h"
},
{
"question": "An employee earns $60000 per year. They receive a 10 percent raise and then a 5 percent bonus on the new salary. What is their final compensation?",
"answer": "69300"
},
{
"question": "A father is three times as old as his son. In 6 years, the father will be twice as old as the son. How old is the father now?",
"answer": "36"
},
{
"question": "A customer buys a jacket for $80. A 20 percent discount is applied, then a 10 percent tax is added. What is the final amount paid?",
"answer": "70.4"
}
]

# create dataset if needed

try:
    dataset = langsmith_client.read_dataset(
    dataset_name=DATASET_NAME
)
except Exception:
    dataset = langsmith_client.create_dataset(
    dataset_name=DATASET_NAME
)

for item in QUESTIONS:
    langsmith_client.create_example(
        inputs={
            "question": item["question"]
        },
        outputs={
            "answer": item["answer"]
        },
        dataset_id=dataset.id
    )


# ====================================================

# CHAIN OF THOUGHT SYSTEM PROMPT

# ====================================================

COT_PROMPT = """
You are a reasoning agent.

Always respond using:

1. THOUGHT
2. CALCULATION
3. FINAL ANSWER

Reason step by step.
"""

# ====================================================

# PROVIDERS

# ====================================================

def openai_target(inputs):

    response = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0,
    messages=[
        {
            "role":"system",
            "content":COT_PROMPT
        },
        {
            "role":"user",
            "content":inputs["question"]
        }
    ]
)

    return {
    "answer":
    response.choices[0].message.content
}

def claude_target(inputs):


    response = claude_client.messages.create(
    model="claude-3-5-sonnet-latest",
    max_tokens=500,
    messages=[
        {
            "role":"user",
            "content":
            COT_PROMPT + "\n\n" + inputs["question"]
        }
    ]
)

    return {
    "answer":
    response.content[0].text
}


def gemini_target(inputs):

    response = gemini_client.models.generate_content(
    model="gemini-2.5-flash",
    contents=
    COT_PROMPT + "\n\n" + inputs["question"]
)

    return {
    "answer":
    response.text
}


# ====================================================

# ACCURACY EVALUATOR

# ====================================================

judge = OpenAI(
api_key=os.getenv("OPENAI_API_KEY")
)

def accuracy_evaluator(run, example):
    prediction = run.outputs.get(
        "answer",
        ""
    )

    reference = example.outputs["answer"]

    judge_prompt = f"""

Reference Answer:
{reference}

Model Answer:
{prediction}

Score accuracy from 1 to 5.

5 = perfect
4 = mostly correct
3 = partially correct
2 = mostly wrong
1 = incorrect

Return only the number.
"""

    result = judge.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": judge_prompt
            }
        ]
    )

    score = int(
        result.choices[0].message.content.strip()[0]
    )

    return {
        "key": "accuracy",
        "score": score
    }


# ====================================================

# LATENCY TEST

# ====================================================

def avg_latency(fn):
    times = []

    for q in QUESTIONS:
        start = time.time()

        fn({
            "question":
            q["question"]
        })

        times.append(
            time.time() - start
        )

    return round(
        sum(times)/len(times),
        2
    )


# ====================================================

# RUN EVALUATIONS

# ====================================================

evaluate(
openai_target,
data=DATASET_NAME,
evaluators=[accuracy_evaluator],
experiment_prefix="OPENAI_COT"
)

evaluate(
claude_target,
data=DATASET_NAME,
evaluators=[accuracy_evaluator],
experiment_prefix="CLAUDE_COT"
)

evaluate(
gemini_target,
data=DATASET_NAME,
evaluators=[accuracy_evaluator],
experiment_prefix="GEMINI_COT"
)

# ====================================================

# REPORT

# ====================================================

report = pd.DataFrame(
[
{
"Provider":"OpenAI",
"Latency(s)":avg_latency(openai_target)
},
{
"Provider":"Claude",
"Latency(s)":avg_latency(claude_target)
},
{
"Provider":"Gemini",
"Latency(s)":avg_latency(gemini_target)
}
]
)

report.to_csv(
"cot_comparison_report.csv",
index=False
)

print(report)
