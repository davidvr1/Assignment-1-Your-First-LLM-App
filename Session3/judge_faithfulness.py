"""
Faithfulness judge for Task 5 - Sonnet, given the answer AND the tool
outputs/retrieved chunks from the trace. Follows the Assignment 2 pattern:
rubric in the prompt, Pydantic schema, `explanation` before `verdict`.
"""

import json

from openai import OpenAI
from pydantic import BaseModel

from rag_pipeline import load_api_key

JUDGE_MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = """את/ה שופט/ת אמינות (faithfulness) לתשובות של מערכת RAG/agent
לתכניות ביטוח בריאות משלים. תקבל/י שאלה, תשובה, ואת ההקשר שהמערכת ראתה בפועל
(קטעים שאוחזרו ו/או תוצאות כלים). קבע/י אם כל טענה עובדתית בתשובה נתמכת
במפורש בהקשר שסופק. תשובה שמסרבת לענות (refusal) כשההקשר אכן לא מכיל את
המידע היא NOT_APPLICABLE (לא מדובר בבעיית אמינות). ספק/י תחילה הסבר קצר,
ורק אח"כ פסק דין."""

USER_TEMPLATE = """שאלה: {question}

תשובת המערכת: {answer}

הקשר שהמערכת ראתה בפועל (קטעים שאוחזרו / תוצאות כלים):
{context}
"""

JUDGE_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_verdict",
        "description": "Submit the faithfulness verdict.",
        "parameters": {
            "type": "object",
            "properties": {
                "explanation": {"type": "string"},
                "verdict": {"type": "string", "enum": ["faithful", "unfaithful", "not_applicable"]},
            },
            "required": ["explanation", "verdict"],
        },
    },
}


class FaithfulnessVerdict(BaseModel):
    explanation: str
    verdict: str


def judge_faithfulness(question: str, answer: str, context: str) -> FaithfulnessVerdict:
    client = OpenAI(base_url="https://api.anthropic.com/v1/", api_key=load_api_key())
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(question=question, answer=answer, context=context[:8000])},
        ],
        tools=[JUDGE_TOOL],
        tool_choice={"type": "function", "function": {"name": "submit_verdict"}},
        temperature=0,
    )
    args = json.loads(resp.choices[0].message.tool_calls[0].function.arguments)
    return FaithfulnessVerdict.model_validate(args)
