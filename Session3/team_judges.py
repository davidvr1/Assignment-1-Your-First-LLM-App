"""
Assignment 5, Task 6 - the LLM-judge calls for the team config: faithfulness,
per-agent success, and (as a fallback) task success where no code checker
exists. Same pattern as Assignment 2 / judge_faithfulness.py: rubric in the
prompt, Pydantic schema, `explanation` before `verdict`, Sonnet as judge
(never the model being judged).

One call per team run (not one per agent) to keep judge cost bounded -
the single call is given every agent's payload+output from the trace and
returns a verdict per agent plus the run-level faithfulness/task-success
calls, so attribution stays possible without 3x-ing the judge bill.
"""

import json

from openai import OpenAI
from pydantic import BaseModel, Field

from rag_pipeline import load_api_key

JUDGE_MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = """את/ה שופט/ת למערכת רב-סוכנית (multi-agent) לשב"ן (ביטוח בריאות
משלים בישראל). תקבל/י את המשימה המקורית, את מסלול השיגור (אילו סוכנים
פעלו ולפי איזה סדר), את הפלט של כל סוכן, ואת התשובה הסופית. שלושה דברים
לשפוט, כל אחד עם הסבר לפני פסק דין:

1. task_success: האם התשובה הסופית/המצב הסופי נכונים ביחס למשימה
   המקורית, בהינתן קריטריון ההצלחה שסופק (אם אין קריטריון קוד, שפוט/י
   לפי נכונות עניינית).
2. faithfulness: האם כל טענה בתשובה הסופית נתמכת במפורש בפלטים של
   הסוכנים (ולא הומצאה)? תשובת סירוב כשאכן לא נמצא מידע היא
   not_applicable, לא unfaithful.
3. per_agent: לכל סוכן שפעל במסלול (researcher/analyst/writer בלבד, אלה
   שמופיעים ב-route) - האם הוא ביצע את *תפקידו שלו* נכון, בהינתן מה
   שהוא קיבל? סוכן שהחזיר תוצאה נכונה למידע שגוי שקיבל נחשב success -
   הכשל אז שייך לשכבה שמעליו, לא לו.
"""

USER_TEMPLATE = """משימה מקורית: {task}
קריטריון הצלחה (אם קיים): {success_criteria}

מסלול שיגור: {route}

פלטי הסוכנים (agent -> output):
{agent_outputs}

תשובה סופית: {answer}
מצב סופי: {terminal_state}
"""


class AgentVerdict(BaseModel):
    explanation: str
    success: bool


class TeamRunVerdict(BaseModel):
    task_success_explanation: str
    task_success: bool
    faithfulness_explanation: str
    faithfulness: str  # faithful / unfaithful / not_applicable
    per_agent: dict[str, AgentVerdict] = Field(default_factory=dict)


JUDGE_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_team_verdict",
        "description": "Submit the team-run verdict.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_success_explanation": {"type": "string"},
                "task_success": {"type": "boolean"},
                "faithfulness_explanation": {"type": "string"},
                "faithfulness": {"type": "string", "enum": ["faithful", "unfaithful", "not_applicable"]},
                "per_agent": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "explanation": {"type": "string"},
                            "success": {"type": "boolean"},
                        },
                        "required": ["explanation", "success"],
                    },
                },
            },
            "required": ["task_success_explanation", "task_success", "faithfulness_explanation",
                         "faithfulness", "per_agent"],
        },
    },
}


def judge_team_run(task: str, success_criteria: str, route: list[str], facts: dict,
                    answer: str, terminal_state: str) -> TeamRunVerdict:
    agent_outputs = "\n".join(
        f"- {agent}: {facts.get(f'{agent}_output', '(no output recorded)')[:600]}"
        for agent in dict.fromkeys(route)  # unique, order-preserving
    ) or "(no agent ran)"

    client = OpenAI(base_url="https://api.anthropic.com/v1/", api_key=load_api_key())

    last_error = None
    for _attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_TEMPLATE.format(
                        task=task, success_criteria=success_criteria or "(none - judge on substance)",
                        route=route, agent_outputs=agent_outputs, answer=answer, terminal_state=terminal_state,
                    )},
                ],
                tools=[JUDGE_TOOL],
                tool_choice={"type": "function", "function": {"name": "submit_team_verdict"}},
            )
            args = json.loads(resp.choices[0].message.tool_calls[0].function.arguments)
            if isinstance(args.get("per_agent"), str):
                args["per_agent"] = json.loads(args["per_agent"]) if args["per_agent"] else {}
            args.setdefault("task_success", False)
            args.setdefault("task_success_explanation", "(judge omitted this field)")
            args.setdefault("faithfulness", "not_applicable")
            args.setdefault("faithfulness_explanation", "(judge omitted this field)")
            return TeamRunVerdict.model_validate(args)
        except Exception as e:  # noqa: BLE001 - one flaky judge call must not crash the eval matrix
            last_error = e
            continue

    # every retry failed - never crash the eval matrix on a judge hiccup; record it as such
    return TeamRunVerdict(
        task_success_explanation=f"judge call failed after retries: {last_error}",
        task_success=False,
        faithfulness_explanation="judge call failed after retries",
        faithfulness="not_applicable",
        per_agent={},
    )
