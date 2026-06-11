# Agentic AI Architecture Foundations – Module 1

This repository contains hands-on demos and evaluation projects for learning the fundamentals of Agentic AI systems, Prompt Engineering, Function Calling, Structured Outputs, and Multi-Provider LLM benchmarking.

---

# Project Overview

This repository includes:

| Demo | Description |
|--------|-------------|
| Demo 1 | Multi-SDK Chatbot Interface |
| Demo 2 | Chain-of-Thought (CoT) Reasoning Agent |
| Demo 3 | Function Calling Agent |
| Demo 4 | Structured JSON Output Agent |
| Evaluation 1 | Multi-Provider Chatbot Benchmark |
| Evaluation 2 | CoT Reasoning Benchmark |
| Evaluation 3 | Structured Output Benchmark |

Providers Supported:

- OpenAI GPT Models
- Google Gemini Models
- Anthropic Claude Models

Evaluation Platform:

- LangSmith

---

# Repository Structure

```text
AGProject1/
│
├── m1_openai_chatbot_demo.py
├── m1_cot_reasoning_demo.py
├── m1_structured_function_demo.py
│
├── chatbot_provider_comparison.py
├── cot_provider_comparison.py
├── structured_output_comparison.py
│
├── reports/
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

# Demo 1 – Multi-SDK Chatbot Interface

## Objective

Build a conversational AI chatbot using the OpenAI SDK.

### Concepts Covered

- OpenAI SDK Initialization
- Chat Completion API
- Streaming Responses
- Session-Based Memory
- Prompt Engineering

### Features

- Interactive Chat Interface
- Streaming Token Output
- Temperature Control
- Conversation History

### Run

```bash
streamlit run m1_openai_chatbot_demo.py
```

---

# Demo 2 – Chain-of-Thought Reasoning Agent

## Objective

Improve reasoning quality using structured Chain-of-Thought prompting.

### Concepts Covered

- System Prompts
- Step-by-Step Reasoning
- Logical Deduction
- Mathematical Reasoning

### CoT Structure

```text
THOUGHT
CALCULATION
FINAL ANSWER
```

### Example

```text
A father is three times as old as his son.
In six years he will be twice as old.
How old is the father now?
```

### Run

```bash
python m1_cot_reasoning_demo.py
```

---

# Demo 3 – Function Calling Agent

## Objective

Use LLMs to identify user intent and invoke backend functions.

### Concepts Covered

- Function Calling
- Tool Schemas
- Argument Extraction
- Agent-to-Tool Communication

### Example Tool

```python
get_order_details(order_id)
```

### Example Query

```text
What is the status of order ORD-202?
```

### Workflow

```text
User Query
    ↓
LLM
    ↓
Function Call
    ↓
Database Lookup
    ↓
Response
```

---

# Demo 4 – Structured JSON Output

## Objective

Generate machine-readable outputs using enforced JSON schemas.

### Concepts Covered

- JSON Mode
- Structured Output
- Schema Validation
- Pydantic Models

### Output Format

```json
{
  "id": "ORD-202",
  "product": "Smartphone",
  "total_cost": 800,
  "current_status": "Shipped"
}
```

### Validation

```python
class OrderOutput(BaseModel):
    id: str
    product: str
    total_cost: float
    current_status: str
```

---

# LangSmith Evaluations

The repository includes benchmark suites that compare:

- OpenAI
- Claude
- Gemini

using LangSmith Evaluation Framework.

---

# Evaluation 1 – Chatbot Benchmark

## Metrics

- Accuracy
- Cost
- Latency

### Test Matrix

```text
5 Prompts × 3 Providers
```

### Providers

```text
OpenAI
Claude
Gemini
```

---

# Evaluation 2 – Chain-of-Thought Benchmark

## Goal

Compare reasoning quality across providers.

### Metrics

- Accuracy
- Cost
- Latency

### Test Cases

1. Fruit Logic
2. Train Speed
3. Salary Calculation
4. Family Age Puzzle
5. Shopping Discount Calculation

---

# Evaluation 3 – Structured Output Benchmark

## Goal

Compare function-calling and JSON generation capabilities.

### Metrics

- Accuracy
- JSON Schema Compliance
- Cost
- Latency

### Test Cases

```text
Order ORD-101
Order ORD-202
Invalid Order
JSON Formatting Requests
Shipping Queries
```

---

# LangSmith Setup

Create a LangSmith account:

:contentReference[oaicite:0]{index=0}

---

## Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_key

ANTHROPIC_API_KEY=your_claude_key

GOOGLE_API_KEY=your_gemini_key

LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=AgenticAIBenchmarks
```

---

# Installation

Create Virtual Environment

```bash
python -m venv .venv
```

Activate

Windows

```bash
.venv\Scripts\activate
```

Mac/Linux

```bash
source .venv/bin/activate
```

Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Requirements

```bash
openai
anthropic
google-genai
langsmith
langchain
python-dotenv
pydantic
streamlit
pandas
```

Generate requirements:

```bash
pip freeze > requirements.txt
```

---

# Running Evaluations

## Chatbot Benchmark

```bash
python chatbot_provider_comparison.py
```

## CoT Benchmark

```bash
python cot_provider_comparison.py
```

## Structured Output Benchmark

```bash
python structured_output_comparison.py
```

---

# Viewing Results in LangSmith of three models Comparison report: 5 prompts × 3 providers, scored on accuracy, cost, latency 


Navigate to:

```text
Projects
    ↓
Experiments
```

Compare:

```text
OPENAI
CLAUDE
GEMINI
```

View:

- Accuracy Scores
- Cost Analysis
- Token Usage
- Latency
- Trace Logs

---


---

# Learning Outcomes

After completing these demos, learners will understand:

- Prompt Engineering Fundamentals
- OpenAI SDK Integration
- Anthropic SDK Integration
- Google Gemini SDK Integration
- Chain-of-Thought Prompting
- Function Calling
- Structured Outputs
- Pydantic Validation
- LangSmith Evaluations
- Multi-Provider Benchmarking

---


<img width="1500" height="775" alt="Screenshot 2026-06-11 053410" src="https://github.com/user-attachments/assets/72dc5c00-662a-4e4f-bde5-463bf27f4e3b" />
<img width="1132" height="647" alt="Screenshot 2026-06-11 070600" src="https://github.com/user-attachments/assets/d6490e85-5986-43b5-bab3-5383f0751005" />
<img width="691" height="235" alt="Screenshot 2026-06-11 080729" src="https://github.com/user-attachments/assets/b8a86a10-0217-45a9-a9e2-65b2be7cd673" />
<img width="612" height="247" alt="Screenshot 2026-06-11 080805" src="https://github.com/user-attachments/assets/518070c1-a605-4397-a917-70ca1ab2d9a8" />
<img width="1143" height="277" alt="Screenshot 2026-06-11 081006" src="https://github.com/user-attachments/assets/36c4414c-7bcb-4e22-bbdd-403cc3572fcf" />
<img width="1102" height="291" alt="Screenshot 2026-06-11 081025" src="https://github.com/user-attachments/assets/dd3eab87-ffb1-462c-a077-5f99365ee15f" />
<img width="1265" height="640" alt="Screenshot 2026-06-11 081121" src="https://github.com/user-attachments/assets/594017c1-23ef-4def-943e-941ba2824e46" />


