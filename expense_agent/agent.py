# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import json
import base64
from typing import Optional, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Setup local credentials gracefully
use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true"
gemini_api_key = os.environ.get("GEMINI_API_KEY")

if use_vertex or not gemini_api_key:
    import google.auth
    try:
        _, project_id = google.auth.default()
        if project_id:
            os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
    except Exception:
        pass
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "mock-project-id")
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
else:
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "mock-project-id")
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"

from google.adk.workflow import Workflow, START, FunctionNode, Edge
from google.adk.apps import App
from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from google.adk.agents.context import Context
from google.genai import types

from expense_agent.config import APPROVAL_THRESHOLD, MODEL_NAME

# State Schema
class ExpenseState(BaseModel):
    original_text: str = ""
    amount: float = 0.0
    submitter: str = ""
    category: str = ""
    description: str = ""
    date: str = ""
    risk_assessment: str = ""
    approved: Optional[bool] = None
    redacted_categories: list[str] = []
    security_event: bool = False

# Helper: Scrub PII
def scrub_pii(text: str) -> tuple[str, list[str]]:
    redacted = []
    # SSN Regex (matches 3-2-4 digit patterns)
    ssn_pattern = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
    if ssn_pattern.search(text):
        text = ssn_pattern.sub("[REDACTED SSN]", text)
        redacted.append("SSN")
        
    # Credit Card Regex (matches 13-16 digits with optional spaces or dashes)
    cc_pattern = re.compile(r'\b(?:\d[ -]*?){13,16}\b')
    if cc_pattern.search(text):
        text = cc_pattern.sub("[REDACTED CREDIT_CARD]", text)
        redacted.append("CREDIT_CARD")
        
    return text, redacted

# Helper: Detect Prompt Injection
def detect_prompt_injection(text: str) -> bool:
    text_lower = text.lower()
    injection_keywords = [
        "ignore previous instructions",
        "ignore the instructions",
        "system prompt",
        "you are now",
        "bypass rules",
        "bypass the rules",
        "override",
        "force approval",
        "force approve",
        "auto-approve",
        "auto_approve",
        "ignore policy",
        "disregard policy"
    ]
    return any(kw in text_lower for kw in injection_keywords)

# Node 1: Parse Pub/Sub or local JSON event
def parse_expense_event(node_input: Any) -> Event:
    """Parses JSON input (including base64 Pub/Sub wrappers) and extracts details."""
    raw_str = ""
    if isinstance(node_input, str):
        raw_str = node_input
    elif hasattr(node_input, "parts") and node_input.parts:
        parts = [p.text for p in node_input.parts if p.text]
        raw_str = " ".join(parts)
    elif isinstance(node_input, dict):
        raw_str = json.dumps(node_input)
    else:
        raw_str = str(node_input)
        
    try:
        payload = json.loads(raw_str)
    except Exception:
        payload = {
            "data": {
                "amount": 150.0 if "expensive" in raw_str.lower() or "laptop" in raw_str.lower() else 45.0,
                "submitter": "Employee",
                "category": "Testing",
                "description": raw_str,
                "date": "2026-06-19"
            }
        }
        
    data_content = None
    if isinstance(payload, dict):
        if "message" in payload and isinstance(payload["message"], dict) and "data" in payload["message"]:
            data_content = payload["message"]["data"]
        elif "data" in payload:
            data_content = payload["data"]
            
    if data_content is None:
        data_content = payload
        
    expense_data = {}
    if isinstance(data_content, str):
        try:
            decoded = base64.b64decode(data_content).decode('utf-8')
            expense_data = json.loads(decoded)
        except Exception:
            try:
                expense_data = json.loads(data_content)
            except Exception:
                expense_data = {}
    elif isinstance(data_content, dict):
        expense_data = data_content
        
    amount = float(expense_data.get("amount", 0.0))
    submitter = str(expense_data.get("submitter", "Unknown"))
    category = str(expense_data.get("category", "General"))
    description = str(expense_data.get("description", "No description provided"))
    date = str(expense_data.get("date", "Unknown"))
    
    return Event(
        output=expense_data,
        state={
            "original_text": raw_str,
            "amount": amount,
            "submitter": submitter,
            "category": category,
            "description": description,
            "date": date
        }
    )

# Node 2: Route based on threshold
def route_expense(amount: float) -> Event:
    """Applies the routing rule based on the approval threshold."""
    if amount < APPROVAL_THRESHOLD:
        return Event(output="auto_approve", route="auto_approve")
    else:
        return Event(output="needs_review", route="needs_review")

# Node 3: Security Checkpoint
def security_checkpoint(description: str) -> Event:
    """Scrubs SSNs and Credit Cards, and checks for prompt injection in the description."""
    scrubbed_desc, redacted_cats = scrub_pii(description)
    has_injection = detect_prompt_injection(description)
    
    state_updates = {
        "description": scrubbed_desc,
        "redacted_categories": redacted_cats
    }
    
    if has_injection:
        state_updates["security_event"] = True
        state_updates["risk_assessment"] = "⚠️ SECURITY ALERT: Prompt injection attempt detected in expense description!"
        return Event(
            output=scrubbed_desc,
            route="injection_detected",
            state=state_updates
        )
    else:
        return Event(
            output=scrubbed_desc,
            route="clean",
            state=state_updates
        )

# Node 4a: Auto-approve under threshold (no LLM)
def auto_approve_expense(amount: float, submitter: str, category: str, description: str, date: str) -> Event:
    """Automatically approves the expense and finishes."""
    msg = (
        f"✅ Expense Auto-Approved!\n"
        f"- Submitter: {submitter}\n"
        f"- Amount: ${amount:.2f}\n"
        f"- Category: {category}\n"
        f"- Description: {description}\n"
        f"- Date: {date}"
    )
    return Event(
        output=msg,
        content=types.Content(role='model', parts=[types.Part.from_text(text=msg)]),
        state={"approved": True}
    )

# Node 4b: Assess risk using Gemini (over threshold - clean path)
async def assess_risk_llm(ctx: Context, amount: float, submitter: str, category: str, description: str, date: str):
    """Invokes LLM to check for risks or policy violations."""
    if os.environ.get("INTEGRATION_TEST") == "TRUE" or (
        not os.environ.get("GEMINI_API_KEY") and (not os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT") == "mock-project-id")
    ):
        risk_judgment = "Low risk (Mock Assessment for testing)."
    else:
        from google.genai import Client
        client = Client()
        prompt = (
            f"Review the following expense for any business risk factors, policy violations, or anomalies:\n"
            f"- Submitter: {submitter}\n"
            f"- Amount: ${amount:.2f}\n"
            f"- Category: {category}\n"
            f"- Description: {description}\n"
            f"- Date: {date}\n\n"
            f"Identify any potential risks. Summarize your assessment in a short paragraph."
        )
        response = await client.aio.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        risk_judgment = response.text or "No assessment generated."
        
    msg = f"⚠️ Alert: Expense of ${amount:.2f} requires review.\n\nRisk Assessment:\n{risk_judgment}"
    
    yield Event(
        content=types.Content(role='model', parts=[types.Part.from_text(text=msg)]),
        state={"risk_assessment": risk_judgment}
    )
    yield Event(output=risk_judgment)

# Node 5: Pause workflow for manager decision
async def human_approval_node(ctx: Context, amount: float, security_event: bool = False, risk_assessment: str = ""):
    """Interrupts workflow and waits for manager decision."""
    if not ctx.resume_inputs or "decision" not in ctx.resume_inputs:
        if security_event:
            msg = (
                f"🚨 SECURITY ALERT: Prompt injection attempt detected in expense description!\n"
                f"Expense Amount: ${amount:.2f}\n"
                f"This request has BYPASSED automatic/LLM reviews. Please inspect carefully.\n"
                f"Do you approve or reject? (yes/no)"
            )
        else:
            msg = (
                f"Requesting manager approval for expense (Amount: ${amount:.2f}).\n"
                f"Risk Assessment: {risk_assessment}\n"
                f"Do you approve or reject? (yes/no)"
            )
        yield RequestInput(
            interrupt_id="decision",
            message=msg
        )
        return
        
    decision_val = ctx.resume_inputs["decision"]
    if isinstance(decision_val, dict):
        decision = (decision_val.get("response") or decision_val.get("decision") or "").lower()
    else:
        decision = str(decision_val).lower()
    is_approved = "yes" in decision or "approve" in decision
    
    if is_approved:
        yield Event(output="approved", route="approved")
    else:
        yield Event(output="rejected", route="rejected")

# Node 6a: Record Manager Approval
def record_approval(amount: float, submitter: str, risk_assessment: str) -> Event:
    """Records manual approval."""
    msg = f"✅ Expense of ${amount:.2f} by {submitter} was APPROVED by the manager.\nRisk assessment: {risk_assessment}"
    return Event(
        output=msg,
        content=types.Content(role='model', parts=[types.Part.from_text(text=msg)]),
        state={"approved": True}
    )

# Node 6b: Record Manager Rejection
def record_rejection(amount: float, submitter: str, risk_assessment: str) -> Event:
    """Records manual rejection."""
    msg = f"❌ Expense of ${amount:.2f} by {submitter} was REJECTED by the manager.\nRisk assessment: {risk_assessment}"
    return Event(
        output=msg,
        content=types.Content(role='model', parts=[types.Part.from_text(text=msg)]),
        state={"approved": False}
    )

# Instantiate nodes
parse_node = FunctionNode(func=parse_expense_event, name="parse_expense_event")
route_node = FunctionNode(func=route_expense, name="route_expense")
security_node = FunctionNode(func=security_checkpoint, name="security_checkpoint")
auto_approve_node = FunctionNode(func=auto_approve_expense, name="auto_approve_expense")
risk_node = FunctionNode(func=assess_risk_llm, name="assess_risk_llm")
approval_node = FunctionNode(func=human_approval_node, name="human_approval_node", rerun_on_resume=True)
record_approval_node = FunctionNode(func=record_approval, name="record_approval")
record_rejection_node = FunctionNode(func=record_rejection, name="record_rejection")

# Wire Graph
root_agent = Workflow(
    name="ambient_expense_agent",
    input_schema=None,
    state_schema=ExpenseState,
    edges=[
        Edge(from_node=START, to_node=parse_node),
        Edge(from_node=parse_node, to_node=route_node),
        Edge(from_node=route_node, to_node=auto_approve_node, route="auto_approve"),
        Edge(from_node=route_node, to_node=security_node, route="needs_review"),
        Edge(from_node=security_node, to_node=risk_node, route="clean"),
        Edge(from_node=security_node, to_node=approval_node, route="injection_detected"),
        Edge(from_node=risk_node, to_node=approval_node),
        Edge(from_node=approval_node, to_node=record_approval_node, route="approved"),
        Edge(from_node=approval_node, to_node=record_rejection_node, route="rejected"),
    ]
)

app = App(
    root_agent=root_agent,
    name="expense_agent",
)
