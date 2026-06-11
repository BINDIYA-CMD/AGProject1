"""
M1_PROVIDER_COMPARISON_LANGSMITH.py

5 Prompts × 3 Providers Benchmark
Providers:
    - OpenAI GPT
    - Anthropic Claude
    - Google Gemini

Metrics:
    - Accuracy (LangSmith Judge)
    - Latency
    - Cost (LangSmith traces)

Requirements:
pip install openai anthropic google-genai langsmith pandas python-dotenv
"""

import os
import time
import pandas as pd

from dotenv import load_dotenv

from openai import OpenAI
from anthropic import Anthropic
from google import genai

from langsmith import Client
from langsmith.evaluation import evaluate

load_dotenv()

# ==========================================================
# API CLIENTS
# ==========================================================

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

claude_client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

gemini_client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

ls_client = Client()

# ==========================================================
# DATASET
# ==========================================================

DATASET_NAME = "Provider Comparison Benchmark"

PROMPTS = [
    {
        "question": "What is 15 percent of 240?",
        "reference": "36"
    },
    {
        "question": "Explain quantum computing to a 10 year old.",
        "reference": "Simple explanation"
    },
    {
        "question": "Write a Python binary search function.",
        "reference": "Python binary search code"
    },
    {
        "question": "Return customer information as JSON.",
        "reference": "Valid JSON"
    },
    {
        "question": "A train travels 120 km in 2 hours. What is its speed?",
        "reference": "60 km/h"
    }
]

# ==========================================================
# CREATE DATASET IF NOT EXISTS
# ==========================================================

try:
    dataset = ls_client.read_dataset(
        dataset_name=DATASET_NAME
    )
except:
    dataset = ls_client.create_dataset(
        dataset_name=DATASET_NAME,
        description="5 prompt provider benchmark"
    )

    for item in PROMPTS:

        ls_client.create_example(
            inputs={
                "question": item["question"]
            },
            outputs={
                "answer": item["reference"]
            },
            dataset_id=dataset.id
        )

# ==========================================================
# PROVIDER FUNCTIONS
# ==========================================================


def openai_target(inputs):

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": inputs["question"]
            }
        ],
        temperature=0
    )

    return {
        "answer":
            response.choices[0].message.content
    }


def claude_target(inputs):

    response = claude_client.messages.create(
        model="claude-opus-4-8",
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": inputs["question"]
            }
        ]
    )

    return {
        "answer":
            response.content[0].text
    }


def gemini_target(inputs):

    response = gemini_client.models.generate_content(
        model="gemini-2.5-pro",
        contents=inputs["question"]
    )

    return {
        "answer":
            response.text
    }

# ==========================================================
# CUSTOM ACCURACY EVALUATOR
# ==========================================================

judge_model = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def accuracy_evaluator(run, example):

    prediction = run.outputs["answer"]

    reference = example.outputs["answer"]

    judge_prompt = f"""
You are an evaluator.

Reference Answer:
{reference}

Model Answer:
{prediction}

Score accuracy from 1 to 5.

Return ONLY a number.
"""

    result = judge_model.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role":"user",
                "content":judge_prompt
            }
        ],
        temperature=0
    )

    score = int(
        result.choices[0].message.content.strip()[0]
    )

    return {
        "key": "accuracy",
        "score": score
    }

# ==========================================================
# LATENCY BENCHMARK
# ==========================================================

def latency_test(fn):

    total = 0

    for item in PROMPTS:

        start = time.time()

        fn({
            "question":
                item["question"]
        })

        total += (
            time.time() - start
        )

    return round(
        total / len(PROMPTS),
        2
    )

# ==========================================================
# RUN LANGSMITH EVALUATIONS
# ==========================================================

print("\nRunning OpenAI Eval...")

evaluate(
    openai_target,
    data=DATASET_NAME,
    evaluators=[
        accuracy_evaluator
    ],
    experiment_prefix="OPENAI"
)

print("\nRunning Claude Eval...")

evaluate(
    claude_target,
    data=DATASET_NAME,
    evaluators=[
        accuracy_evaluator
    ],
    experiment_prefix="CLAUDE"
)

#print("\nRunning Gemini Eval...")

#evaluate(
#    gemini_target,
#    data=DATASET_NAME,
#    evaluators=[
#        accuracy_evaluator
#    ],
#    experiment_prefix="GEMINI"
#)

# ==========================================================
# LATENCY REPORT
# ==========================================================

openai_latency = latency_test(
    openai_target
)

claude_latency = latency_test(
    claude_target
)

#gemini_latency = latency_test(
#    gemini_target
#)

# ==========================================================
# SUMMARY REPORT
# ==========================================================

report = pd.DataFrame(
    [
        {
            "Provider": "OpenAI GPT-4o",
            "Latency(s)": openai_latency
        },
        {
            "Provider": "Claude Sonnet",
            "Latency(s)": claude_latency
        },
        #{
        #    "Provider": "Gemini Pro",
        #    "Latency(s)": gemini_latency
        #}
    ]
)

report.to_csv(
    "provider_comparison_report.csv",
    index=False
)

print("\nBenchmark Complete")
print(report)

print(
    "\nOpen LangSmith UI "
    "to compare Accuracy, Cost, "
    "Token Usage and Latency."
)
