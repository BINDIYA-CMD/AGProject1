"""
DEMO 3 & 4: Combined Function Calling & JSON Structured Output
Module 1: LLM Foundations & Prompt Engineering
---------------------------------------------------------
Scenario: A Customer Support Agent checks an order status (Function Calling)
and then formats the final update as a machine-readable JSON (Structured Output).

Learners will see:
1. Secure key loading via dotenv.
2. Function schema definition and "tool_calls" handling.
3. Using response_format={"type": "json_object"} for the final step.
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

# --- INITIALIZATION ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- 1. MOCK TOOL: DATABASE LOOKUP ---
def get_order_details(order_id):
    """Simulates fetching raw data from a database[cite: 14]."""
    db = {
        "ORD-101": {"item": "Laptop", "price": 1200, "status": "Processing"},
        "ORD-202": {"item": "Smartphone", "price": 800, "status": "Shipped"}
    }
    return db.get(order_id, {"error": "Order not found"})

# --- 2. TOOL DEFINITION (SCHEMA) ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_order_details",
            "description": "Retrieve item details and status for a specific order ID[cite: 14, 58].",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "The order ID, e.g., ORD-101"}
                },
                "required": ["order_id"]
            }
        }
    }
]


# --- Pydantic model for the structured output ---
class OrderOutput(BaseModel):
    id: str
    product: str
    total_cost: float
    current_status: str

def run_combined_demo(user_query):
    print(f"Step 1: Processing User Query -> '{user_query}'")

    # --- PHASE A: FUNCTION CALLING ---
    # The model identifies the intent and extracts the Order ID[cite: 14, 58].
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": user_query}],
        tools=tools
    )

    message = response.choices[0].message
    
    if message.tool_calls:
        tool_call = message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)
        print(f"Agent decided to call: {tool_call.function.name} with {args}")

        # Execute the local mock function [cite: 19]
        raw_data = get_order_details(args['order_id'])
        print(f"Raw Database Result: {raw_data}")

        # --- PHASE B: STRUCTURED JSON OUTPUT ---
        # Now we take the raw data and force the AI to return a clean JSON Schema[cite: 13, 18, 25].
        print("\nStep 2: Formatting output into Structured JSON...")
        
        system_format_instruction = (
            "You are a data processing agent. Take the provided order data and "
            "convert it into a valid JSON object with keys: 'id', 'product', 'total_cost', and 'current_status'."
        )

        final_response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"}, # Enforcement 
            messages=[
                {"role": "system", "content": system_format_instruction},
                {"role": "user", "content": f"Format this data: {raw_data} for ID {args['order_id']}"}
            ]
        )

        # Parse and validate the final structured output using Pydantic
        structured_json = json.loads(final_response.choices[0].message.content)
        try:
            order = OrderOutput(**structured_json)
            print("\n--- FINAL STRUCTURED OUTPUT (Validated) ---")
            # Pydantic v2: use model_dump_json instead of the deprecated .json()
            print(order.model_dump_json(indent=4))
        except ValidationError as e:
            print("Validation error parsing structured output:", e)
            print("Raw structured JSON:")
            print(json.dumps(structured_json, indent=4))

# --- EXECUTION ---
if __name__ == "__main__":
    run_combined_demo("Can you give me the details for order ORD-202?")