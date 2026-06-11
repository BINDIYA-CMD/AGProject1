import os
import json
import time
from dotenv import load_dotenv
from pydantic import BaseModel

from openai import OpenAI
from anthropic import Anthropic
#from google import genai

from langsmith import Client
from langsmith.evaluation import evaluate

# =====================================================
# ENVIRONMENT
# =====================================================

load_dotenv()

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

claude_client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

#gemini_client = genai.Client(
#    api_key=os.getenv("GOOGLE_API_KEY")
#)

langsmith_client = Client()

# =====================================================
# MOCK DATABASE
# =====================================================

def get_order_details(order_id):

    db = {
        "ORD-101": {
            "item": "Laptop",
            "price": 1200,
            "status": "Processing"
        },
        "ORD-202": {
            "item": "Smartphone",
            "price": 800,
            "status": "Shipped"
        }
    }

    return db.get(
        order_id,
        {
            "item": "Unknown",
            "price": 0,
            "status": "Order not found"
        }
    )

# =====================================================
# OUTPUT SCHEMA
# =====================================================

class OrderOutput(BaseModel):
    id: str
    product: str
    total_cost: float
    current_status: str

# =====================================================
# DATASET
# =====================================================

DATASET_NAME = "Structured_Output_Benchmark"

TEST_CASES = [
    {
        "question": "Can you give me the details for order ORD-101?",
        "expected_order": "ORD-101"
    },
    {
        "question": "What's the current status of my order ORD-202?",
        "expected_order": "ORD-202"
    },
    {
        "question": "Please check order ORD-101 and provide JSON output.",
        "expected_order": "ORD-101"
    },
    {
        "question": "I need shipping information for ORD-202.",
        "expected_order": "ORD-202"
    },
    {
        "question": "Can you check order ORD-999?",
        "expected_order": "ORD-999"
    }
]

try:
    dataset = langsmith_client.read_dataset(
        dataset_name=DATASET_NAME
    )

except Exception:

    dataset = langsmith_client.create_dataset(
        dataset_name=DATASET_NAME
    )

    for item in TEST_CASES:

        langsmith_client.create_example(
            inputs={
                "question": item["question"]
            },
            outputs={
                "expected_order":
                item["expected_order"]
            },
            dataset_id=dataset.id
        )

# =====================================================
# PROMPT
# =====================================================

SYSTEM_PROMPT = """
Extract order information and return ONLY valid JSON.

Schema:

{
  "id": "",
  "product": "",
  "total_cost": 0,
  "current_status": ""
}
"""

# =====================================================
# OPENAI
# =====================================================

def openai_target(inputs):

    try:

        question = inputs["question"]

        order_id = (
            "ORD-101"
            if "ORD-101" in question
            else "ORD-202"
            if "ORD-202" in question
            else "ORD-999"
        )

        data = get_order_details(order_id)

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={
                "type": "json_object"
            },
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content":
                    f"Convert this into JSON. "
                    f"Order ID={order_id}, "
                    f"Data={data}"
                }
            ]
        )

        return {
            "answer":
            json.loads(
                response.choices[0]
                .message.content
            )
        }

    except Exception as e:

        return {
            "answer": {
                "error": str(e)
            }
        }

# =====================================================
# CLAUDE
# =====================================================

def claude_target(inputs):

    try:

        question = inputs["question"]

        order_id = (
            "ORD-101"
            if "ORD-101" in question
            else "ORD-202"
            if "ORD-202" in question
            else "ORD-999"
        )

        data = get_order_details(order_id)

        response = claude_client.messages.create(
            model="claude-opus-4-8",
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content":
                    SYSTEM_PROMPT +
                    f"\nOrder ID={order_id}\n"
                    f"Data={data}"
                }
            ]
        )

        return {
            "answer":
            json.loads(
                response.content[0].text
            )
        }

    except Exception as e:

        return {
            "answer": {
                "error": str(e)
            }
        }

# =====================================================
# GEMINI (disabled)
# =====================================================
# Gemini integration is disabled because the Google API key
# may be missing or invalid. To enable, provide a valid
# GOOGLE_API_KEY in your .env and uncomment the implementation.

#def gemini_target(inputs):
#    # Implement Gemini call similar to OpenAI/Claude targets
#    # response = gemini_client.models.generate_content(...)
#    # return {"answer": json.loads(response.text)}
#    pass

# =====================================================
# EVALUATORS
# =====================================================

def accuracy_evaluator(run, example):

    try:

        output = run.outputs["answer"]

        expected_order = (
            example.outputs["expected_order"]
        )

        score = 0

        if output.get("id") == expected_order:
            score += 3

        if "product" in output:
            score += 1

        if "current_status" in output:
            score += 1

        return {
            "key": "accuracy",
            "score": score
        }

    except Exception:

        return {
            "key": "accuracy",
            "score": 0
        }


def json_schema_evaluator(run, example):

    try:

        OrderOutput(
            **run.outputs["answer"]
        )

        return {
            "key": "json_valid",
            "score": 1
        }

    except Exception:

        return {
            "key": "json_valid",
            "score": 0
        }

# =====================================================
# LATENCY
# =====================================================

def measure_latency(fn):

    times = []

    for item in TEST_CASES:

        start = time.time()

        fn({
            "question":
            item["question"]
        })

        times.append(
            time.time() - start
        )

    return round(
        sum(times) / len(times),
        2
    )

# =====================================================
# EVALUATIONS
# =====================================================

print("\nRunning OpenAI Evaluation...")

evaluate(
    openai_target,
    data=DATASET_NAME,
    evaluators=[
        accuracy_evaluator,
        json_schema_evaluator
    ],
    experiment_prefix="OPENAI_STRUCTURED"
)

print("\nRunning Claude Evaluation...")

evaluate(
    claude_target,
    data=DATASET_NAME,
    evaluators=[
        accuracy_evaluator,
        json_schema_evaluator
    ],
    experiment_prefix="CLAUDE_STRUCTURED"
)

#print("\nRunning Gemini Evaluation...")

#evaluate(
#    gemini_target,
#    data=DATASET_NAME,
#    evaluators=[
#        accuracy_evaluator,
#        json_schema_evaluator
#    ],
 #   experiment_prefix="GEMINI_STRUCTURED"
#)

# =====================================================
# REPORT
# =====================================================

print("\n==============================")
print("LATENCY REPORT")
print("==============================")

print(
    f"OpenAI : "
    f"{measure_latency(openai_target)} sec"
)

print(
    f"Claude : "
    f"{measure_latency(claude_target)} sec"
)

#print(
#    f"Gemini : "
#    f"{measure_latency(gemini_target)} sec"
#)

print(
    "\nOpen LangSmith and compare:"
)

print("OPENAI_STRUCTURED")
print("CLAUDE_STRUCTURED")
print("GEMINI_STRUCTURED")