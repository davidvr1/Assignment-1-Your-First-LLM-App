"""
Task 1 - No-RAG baseline.

For every question in eval_set.json, ask the generator model (claude-haiku-4-5)
directly, with NO context and NO documents. Same system-prompt discipline as
Assignment 1 (Session1/file_qa.py): answer only if you know, otherwise refuse.

Records answer, latency, token counts, and a first-pass automatic classification
(refused / answered / needs_review) based on refusal-phrase detection. The
"answered correctly" vs. "hallucinated" split needs a human (or judge) to compare
against reference_answer -- that final call is left as a blank column for you to
fill in by reading the answers, same as the human rubric columns in Session2.

Reads:  eval_set.json
Writes: baseline_results.xlsx
"""

import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

MODEL = "claude-haiku-4-5"
EVAL_SET_FILE = "eval_set.json"
OUTPUT_FILE = "baseline_results.xlsx"
KEY_FILE = Path(__file__).parent / "key.txt"

SYSTEM_PROMPT = (
    "You answer questions about Israeli supplementary health insurance plans "
    "(שב\"ן) using ONLY your own knowledge. You have not been given any "
    "documents. Answer in Hebrew. If you know the answer with confidence, "
    "answer it directly. If you do not know, or are not sure, reply with "
    "exactly this sentence and nothing else: \"אין לי מידע מספיק כדי לענות "
    "על כך בביטחון.\" Do not guess or invent specific numbers, caps, or plan "
    "codes."
)

REFUSAL_MARKERS = [
    "אין לי מידע מספיק",
    "אינני יודע",
    "אני לא יודע",
    "לא ידוע לי",
    "אין לי מספיק מידע",
]


def load_api_key() -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        return api_key
    if KEY_FILE.exists():
        match = re.search(r'ANTHROPIC_API_KEY="([^"]+)"', KEY_FILE.read_text())
        if match:
            return match.group(1)
    raise RuntimeError(
        f"ANTHROPIC_API_KEY not set and not found in {KEY_FILE}. "
        "Set the env var, or create key.txt with: ANTHROPIC_API_KEY=\"...\""
    )


def classify_refusal(answer: str) -> str:
    return "refused" if any(m in answer for m in REFUSAL_MARKERS) else "answered"


def main() -> None:
    with open(EVAL_SET_FILE, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    client = OpenAI(
        base_url="https://api.anthropic.com/v1/",
        api_key=load_api_key(),
    )

    rows = []
    for item in eval_set:
        start = time.perf_counter()
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["question"]},
            ],
            temperature=0,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        answer = resp.choices[0].message.content.strip()
        auto_class = classify_refusal(answer)

        rows.append({
            "id": item["id"],
            "question": item["question"],
            "reference_answer": item["reference_answer"],
            "evidence_doc": item["evidence_doc"],
            "evidence_page": item["evidence_page"],
            "answerable": item["answerable"],
            "difficulty": item["difficulty"],
            "baseline_answer": answer,
            "baseline_auto_class": auto_class,  # refused / answered
            "baseline_final_class": "",  # fill in by hand: refused / correct / hallucinated
            "latency_ms": round(latency_ms, 1),
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
        })
        print(f"[{item['id']}] {auto_class} ({latency_ms:.0f} ms) - {item['question'][:50]}")

    df = pd.DataFrame(rows)
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(rows)} rows to {OUTPUT_FILE}")
    print(
        "\nNext: open the file, read each 'baseline_answer' against "
        "'reference_answer', and fill 'baseline_final_class' with "
        "refused / correct / hallucinated for every row."
    )


if __name__ == "__main__":
    main()
