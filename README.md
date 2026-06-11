A provider-agnostic LLM client wrapping OpenAI, Gemini, and Claude SDKs behind one interface 

Side-by-side response comparison for the same prompt across all three models 

Prompt-strategy switcher: zero-shot, few-shot, and Chain-of-Thought modes 

Structured-output mode using JSON Schema enforcement and Pydantic validation 

A basic function-calling demo showing one tool definition and dispatch loop 

Token and cost telemetry per provider per call 

Tech stack 

OpenAI SDK  ·  Google Gemini SDK  ·  Anthropic Claude SDK  ·  Pydantic  ·  Python 3.11+ 

Deliverables 

m1_openai_chatbot_demo.py — multi-SDK chatbot interface 

m1_cot_reasoning_demo.py — Chain-of-Thought reasoning agent 

m1_structured_function_demo.py — JSON schema enforcement and function calling 

Comparison report: 5 prompts × 3 providers, scored on accuracy, cost, latency 
<img width="1920" height="1080" alt="Screenshot (5)" src="https://github.com/user-attachments/assets/901d7345-7328-43a8-af25-ccb823635541" />
<img width="1920" height="1080" alt="Screenshot (4)" src="https://github.com/user-attachments/assets/204698ba-31ec-4873-ab00-ca69e1a1e2b4" />

