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
import sys
import json
import asyncio
from pathlib import Path
from typing import Any, Optional

# Setup environment variables for mock run
os.environ["GOOGLE_CLOUD_PROJECT"] = "mock-project-id"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
os.environ["INTEGRATION_TEST"] = "TRUE"

# Apply mock conftest auth
try:
    import google.auth
    from google.auth.credentials import Credentials
    class MockCredentials(Credentials):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.token = "mock-token"
        def refresh(self, request):
            self.token = "mock-token"
    google.auth.default = lambda *args, **kwargs: (MockCredentials(), "mock-project-id")
except ImportError:
    pass

from google.adk.apps import App
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from expense_agent.agent import root_agent

async def main():
    dataset_path = Path("tests/eval/datasets/basic-dataset.json")
    output_path = Path("artifacts/traces/generated_traces.json")
    
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    eval_cases = data.get("eval_cases", [])
    print(f"Loaded {len(eval_cases)} eval case(s).")
    
    session_service = InMemorySessionService()
    runner = Runner(agent=root_agent, session_service=session_service, app_name="expense_agent")
    
    output_cases = []
    
    for i, case in enumerate(eval_cases):
        case_id = case["eval_case_id"]
        prompt_text = case["prompt"]["parts"][0]["text"]
        print(f"Running inference for case {i+1}/{len(eval_cases)}: {case_id}")
        
        session = await runner.session_service.create_session(app_name="expense_agent", user_id="eval-user")
        
        # Turn 0: Send initial event payload
        msg0 = types.Content(role="user", parts=[types.Part.from_text(text=prompt_text)])
        events = []
        
        async for event in runner.run_async(user_id="eval-user", session_id=session.id, new_message=msg0):
            events.append(event)
            
        # Check if workflow is interrupted for manual approval
        interrupted = False
        for event in events:
            if event.interrupted or (event.long_running_tool_ids and "decision" in event.long_running_tool_ids):
                interrupted = True
                break
                
        if interrupted:
            print(f"Case {case_id} paused for human-in-the-loop decision.")
            # Automate decision: Reject if prompt injection, otherwise approve
            should_approve = case_id != "prompt_injection_containment"
            decision_val = "yes" if should_approve else "no"
            print(f"Automating decision: {decision_val}")
            
            # Turn 1: Send decision
            msg1 = types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            name="adk_request_input",
                            id="decision",
                            response={"response": decision_val}
                        )
                    )
                ]
            )
            
            async for event in runner.run_async(user_id="eval-user", session_id=session.id, new_message=msg1):
                events.append(event)
                
        # Group events by invocation ID to form turns
        invocation_to_events = {}
        for event in events:
            event_dict = event.model_dump(exclude_none=True)
            inv_id = event_dict.get("invocationId") or "default"
            if inv_id not in invocation_to_events:
                invocation_to_events[inv_id] = []
            invocation_to_events[inv_id].append(event_dict)
            
        turns = []
        for idx, (inv_id, turn_events) in enumerate(invocation_to_events.items()):
            cleaned_events = []
            for ev in turn_events:
                clean_ev = {}
                if ev.get("author"):
                    clean_ev["author"] = str(ev["author"])
                if ev.get("content"):
                    clean_ev["content"] = ev["content"]
                if clean_ev.get("content"):
                    cleaned_events.append(clean_ev)
            if cleaned_events:
                turns.append({
                    "turn_index": idx,
                    "turn_id": f"turn_{idx}",
                    "events": cleaned_events
                })
            
        # Extract final candidate response
        final_response = None
        for event in reversed(events):
            event_dict = event.model_dump(exclude_none=True)
            content = event_dict.get("content")
            if content and content.get("parts"):
                texts = [p.get("text") for p in content["parts"] if p.get("text")]
                if texts:
                    final_response = {
                        "role": "model",
                        "parts": [{"text": "".join(texts)}]
                    }
                    break
                    
        # Package case
        case_result = {
            "eval_case_id": case_id,
            "agent_data": {
                "agents": {
                    "expense_agent": {
                        "agent_id": "expense_agent",
                        "instruction": getattr(root_agent, "instruction", "Expense approval workflow")
                    }
                },
                "turns": turns
            }
        }
        if final_response:
            case_result["responses"] = [{"response": final_response}]
            
        output_cases.append(case_result)
        print(f"Case {case_id} done.")
        
    def make_json_serializable(data: Any) -> Any:
        if isinstance(data, dict):
            return {k: make_json_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [make_json_serializable(x) for x in data]
        elif isinstance(data, set):
            return [make_json_serializable(x) for x in sorted(list(data))]
        return data

    result_dataset = {"eval_cases": make_json_serializable(output_cases)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_dataset, f, indent=2)
        
    print(f"Wrote traces to {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
