"""
Task 5, part 1 - The four LLM judges (Sonnet), plus hit-rate@K and refusal
correctness in code.

Judge/generator split follows Assignment 2's rule: claude-sonnet-5 judges,
claude-haiku-4-5 generates -- never the same model marking its own work.
Each judge's Pydantic schema puts `explanation` BEFORE `verdict` (Assignment
2 finding: this forces the model to reason before committing, instead of
picking a verdict and rationalizing it after).

Critically, each judge sees ONLY what it needs:
- context_relevance: question + chunks (NOT the answer -- else it reasons
  backwards from the answer instead of judging retrieval on its own merit).
- faithfulness: answer + chunks (NOT the reference answer -- faithfulness is
  about whether the answer is supported by what was retrieved, not whether
  it's correct).
- answer_relevance: question + answer (NOT the chunks or reference).
- correctness: answer + reference_answer (NOT the chunks).
"""

import os
import re
from pathlib import Path

from anthropic import Anthropic
from pydantic import BaseModel

JUDGE_MODEL = "claude-sonnet-5"
KEY_FILE = Path(__file__).parent / "key.txt"


class JudgeVerdict(BaseModel):
    explanation: str  # MUST come first -- reasoning before verdict, not after
    verdict: bool      # True = good/pass, False = bad/fail


def load_api_key() -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        return api_key
    if KEY_FILE.exists():
        match = re.search(r'ANTHROPIC_API_KEY="([^"]+)"', KEY_FILE.read_text())
        if match:
            return match.group(1)
    raise RuntimeError(f"ANTHROPIC_API_KEY not set and not found in {KEY_FILE}")


_client = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic(api_key=load_api_key())
    return _client


def _call_judge(system_prompt: str, user_prompt: str, retries: int = 3) -> JudgeVerdict:
    client = _get_client()
    last_error = None
    for attempt in range(retries):
        resp = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=600,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            tools=[{
                "name": "submit_verdict",
                "description": "Submit your judgment.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "explanation": {"type": "string", "description": "Reasoning, written BEFORE deciding the verdict."},
                        "verdict": {"type": "boolean", "description": "true = good/pass, false = bad/fail"},
                    },
                    "required": ["explanation", "verdict"],
                },
            }],
            tool_choice={"type": "tool", "name": "submit_verdict"},
        )
        tool_use = next((b for b in resp.content if b.type == "tool_use"), None)
        # Occasionally the tool_use block comes back with an empty/partial
        # input (transient API hiccup, or the model hit max_tokens mid-call).
        # Retry rather than crash a 150-call batch job over one bad response.
        if tool_use is not None:
            try:
                return JudgeVerdict.model_validate(tool_use.input)
            except Exception as e:
                last_error = e
        else:
            last_error = RuntimeError(f"no tool_use block, stop_reason={resp.stop_reason}")
    raise RuntimeError(f"Judge call failed after {retries} attempts: {last_error}")


def judge_context_relevance(question: str, chunks_text: str) -> JudgeVerdict:
    system = (
        "אתה שופט איכות שמעריך רלוונטיות של אחזור מידע (retrieval) עבור מערכת RAG. "
        "קיבלת שאלה וקטעי טקסט שאוחזרו עבורה. אתה לא רואה את התשובה הסופית -- "
        "תפקידך אך ורק להעריך האם הקטעים עצמם רלוונטיים לנושא השאלה, גם אם הם "
        "אינם עונים עליה במדויק. verdict=true אם לפחות חלק ניכר מהקטעים "
        "רלוונטיים לנושא השאלה; verdict=false אם הקטעים עוסקים בנושא אחר לגמרי."
    )
    user = f"שאלה:\n{question}\n\nקטעים שאוחזרו:\n{chunks_text}"
    return _call_judge(system, user)


def judge_faithfulness(answer: str, chunks_text: str) -> JudgeVerdict:
    system = (
        "אתה שופט איכות שמעריך נאמנות (faithfulness) של תשובה למקורות שסופקו לה. "
        "אתה לא רואה את השאלה המקורית ולא תשובת ייחוס -- תפקידך אך ורק לבדוק האם "
        "כל טענה עובדתית בתשובה נתמכת על ידי הקטעים שסופקו. verdict=true אם כל "
        "טענה בתשובה נתמכת על ידי הקטעים (או שהתשובה היא סירוב שאינו טוען דבר); "
        "verdict=false אם התשובה מכילה טענה שאינה מופיעה בקטעים (הזיה)."
    )
    user = f"קטעי מקור:\n{chunks_text}\n\nתשובת המערכת:\n{answer}"
    return _call_judge(system, user)


def judge_answer_relevance(question: str, answer: str) -> JudgeVerdict:
    system = (
        "אתה שופט איכות שמעריך רלוונטיות של תשובה לשאלה שנשאלה. אתה לא רואה את "
        "מקורות המידע ולא תשובת ייחוס. verdict=true אם התשובה מתייחסת ישירות "
        "לשאלה שנשאלה (כולל סירוב עדין וממוקד כשמתאים); verdict=false אם התשובה "
        "סוטה מהנושא, עונה על שאלה אחרת, או כללית מדי מכדי להיחשב מענה לשאלה."
    )
    user = f"שאלה:\n{question}\n\nתשובה:\n{answer}"
    return _call_judge(system, user)


def judge_correctness(answer: str, reference_answer: str) -> JudgeVerdict:
    system = (
        "אתה שופט איכות שמעריך נכונות (correctness) של תשובה מול תשובת ייחוס "
        "(reference answer) שנחשבת נכונה. verdict=true אם התשובה תואמת מהותית "
        "את תשובת הייחוס (מספרים, תנאים ועובדות מרכזיות זהים או שקולים); "
        "verdict=false אם התשובה סותרת את תשובת הייחוס, חסרה פרט מהותי, או "
        "שהיא סירוב כאשר תשובת הייחוס מספקת תשובה קונקרטית."
    )
    user = f"תשובת ייחוס (נכונה):\n{reference_answer}\n\nתשובת המערכת להערכה:\n{answer}"
    return _call_judge(system, user)
