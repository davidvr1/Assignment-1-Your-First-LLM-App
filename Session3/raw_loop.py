"""
Task 3.1 - The ReAct loop, written by hand, no framework.

Run one multi-hop task through it and print every message, so the LangGraph
agent built next (agent.py) is never a black box.

Uses the OpenAI-compatible endpoint against Anthropic, matching the rest of
this repo's Assignment 3 code (rag_pipeline.py) rather than introducing a
second client style just for this file.
"""

import json

from openai import OpenAI

from rag_pipeline import load_api_key
from tools import ALL_TOOLS

AGENT_MODEL = "claude-haiku-4-5"
MAX_STEPS = 10

SYSTEM_PROMPT = """את/ה עוזר סוכן (agent) לתכניות ביטוח בריאות משלים (שב"ן).
ענה/י אך ורק על סמך מה שהכלים מחזירים בפועל - אל תמציא/י מספרים או עובדות.
אם כלי נכשל פעמיים ברצף, הפסק/י ואמור/י שלא ניתן להשלים את המשימה.
אל תקרא/י לכלי אם כבר יש לך את התשובה. סרב/י בנימוס אם התשובה לא זמינה
במסמכים או בכלים, אחרי חיפוש מוגבל (לא מעל 3-4 קריאות כלי)."""

# Map tool name -> callable, and build the OpenAI-style tool schema by hand
# (langchain's .args_schema gives us the JSON schema without re-declaring it).
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}


def _tool_schema(t) -> dict:
    return {
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": t.args_schema.model_json_schema() if t.args_schema else {"type": "object", "properties": {}},
        },
    }


TOOL_SCHEMAS = [_tool_schema(t) for t in ALL_TOOLS]


def run_raw_loop(task: str, verbose: bool = True) -> str:
    client = OpenAI(base_url="https://api.anthropic.com/v1/", api_key=load_api_key())

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]

    for step in range(MAX_STEPS):
        resp = client.chat.completions.create(
            model=AGENT_MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            temperature=0,
        )
        msg = resp.choices[0].message
        if verbose:
            print(f"\n=== step {step} ===")
            print("assistant:", msg.content or "(no text, tool call only)")

        if not msg.tool_calls:
            return msg.content or ""

        # Append the assistant turn (with tool_calls) then one tool result per call.
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ],
        })
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {}
            tool = TOOLS_BY_NAME.get(name)
            if tool is None:
                result = f"ERROR: unknown tool '{name}'."
            else:
                result = tool.invoke(args)
            if verbose:
                print(f"  tool call: {name}({args}) -> {str(result)[:200]}")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result),
            })

    return "STEP_LIMIT_REACHED: could not complete the task within 10 steps."


if __name__ == "__main__":
    task = (
        "כמה יעלה בסך הכל למבוטח מכבי זהב לבצע 3 ביקורים אצל רופא מומחה "
        "השנה, לפי דמי ההשתתפות העצמית שלו?"
    )
    answer = run_raw_loop(task)
    print("\n=== FINAL ANSWER ===")
    print(answer)
