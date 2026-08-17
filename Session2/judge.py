"""
Task 5/6 - LLM-as-a-judge.

Uses a small local model (TinyLlama-1.1B-Chat, free, no API key) to apply the
rubric in evaluation_rubric.md to every row of electrical_items_dataset2.xlsx.

Judged criteria: Fluency, Grammar, Tone, Length, Grounding (each returns an
explanation before a good/ok/bad verdict). One short generation call per
criterion is used instead of a single structured call, since the small local
model follows a simple "Explanation: ... / Verdict: ..." format far more
reliably than JSON or nested tool schemas.
Latency is excluded from the judge (per the assignment) and is instead
computed programmatically from the measured latency_ms column.

final_score (pass/fail) is derived from the same two-step rule as the human
rubric: any 'bad' -> REJECT; else >=4 'good' among the six -> PASS, else FAIL.

Reads:  electrical_items_dataset2.xlsx
Writes: electrical_items_dataset2.xlsx (adds judge_* columns, keeps human
        Fluency/Grammar/Tone/Length/Grounding/Latency/final_score columns
        untouched for manual scoring)
"""

import re

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_FILE = "electrical_items_dataset2.xlsx"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CRITERIA = ["fluency", "grammar", "tone", "length", "grounding"]


class CriterionVerdict:
    def __init__(self, explanation: str, verdict: str):
        self.explanation = explanation
        self.verdict = verdict


def get_criterion(result: dict, criterion: str) -> CriterionVerdict:
    return result[criterion]


# Latency thresholds (ms), full-response time, per evaluation_rubric.md's
# stated assumption (good <= 3.0s, ok <= 7.0s, bad above).
LATENCY_GOOD_MS = 3000
LATENCY_OK_MS = 7000

# Condensed, per-criterion rubric: a 1B local model has a small context window
# and weak instruction-following. A single call asked to return JSON for all
# five criteria at once was unreliable (it ignored the format and rambled).
# Instead, judge_one() makes one short call per criterion, asking for a plain
# "Explanation: ... / Verdict: good|ok|bad" reply, which a tiny model follows
# far more consistently than JSON.
CRITERION_RUBRIC = {
    "fluency": (
        "Fluency: does it read naturally, like a human wrote it? "
        "good = reads smoothly, no awkward parts. "
        "ok = understandable but a phrase or two feels stiff or off. "
        "bad = hard to parse, broken sentences, or repetition."
    ),
    "grammar": (
        "Grammar: spelling, punctuation, agreement. "
        "good = no errors. "
        "ok = one minor typo or punctuation slip. "
        "bad = agreement/tense errors, or 2+ minor errors."
    ),
    "tone": (
        "Tone: warm, confident, plain sales voice, no hype or pressure. "
        "good = warm and credible, no pressure tactics. "
        "ok = slightly too stiff/corporate or slightly too salesy. "
        "bad = pushy, over-hyped, or robotic."
    ),
    "length": (
        "Length: target is 50-90 words. "
        "good = 50-90 words. "
        "ok = 40-49 or 91-108 words. "
        "bad = under 40 or over 108 words."
    ),
    "grounding": (
        "Grounding: does it only use facts present in the source spec, inventing nothing? "
        "good = every claim traces to the source spec. "
        "ok = no invented facts, only mild unsupported framing or generic filler. "
        "bad = any invented or contradicted fact (a number, feature, or price not in the spec)."
    ),
}

SYSTEM_PROMPT_TEMPLATE = """You are a strict evaluator of one e-commerce product description
(written in Hebrew, generated from English source attributes).

Rubric for this criterion:
{criterion_rubric}

Judge grounding only against the given source spec below, never your own knowledge.
Look only at THIS criterion, and base your answer on THIS specific description below.
You will complete a reply of the form:
Explanation: <one short sentence about this description, for this criterion only>
Verdict: <good, ok, or bad>"""

USER_TEMPLATE = """Product name: {name}

Source spec (ground truth attributes):
{spec}

Generated description to evaluate:
{description}"""


def latency_verdict(latency_ms: float) -> str:
    if latency_ms <= LATENCY_GOOD_MS:
        return "good"
    if latency_ms <= LATENCY_OK_MS:
        return "ok"
    return "bad"


def final_score(verdicts: dict) -> str:
    if any(v == "bad" for v in verdicts.values()):
        return "fail"
    good_count = sum(1 for v in verdicts.values() if v == "good")
    return "pass" if good_count >= 4 else "fail"


def build_messages(criterion: str, name: str, spec: str, description: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(
            criterion_rubric=CRITERION_RUBRIC[criterion])},
        {"role": "user", "content": USER_TEMPLATE.format(name=name, spec=spec, description=description)},
    ]


_VERDICT_RE = re.compile(r"verdict\s*:\s*(good|ok|bad)\b", re.IGNORECASE)

def _generate(tokenizer, model, prompt_text: str, max_new_tokens: int, sample: bool) -> str:
    inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
    input_tokens = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=sample,
            temperature=0.7 if sample else None,
            top_p=0.9 if sample else None,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0][input_tokens:], skip_special_tokens=True)


def judge_criterion(tokenizer, model, criterion: str, name: str, spec: str, description: str,
                     retries: int = 3) -> CriterionVerdict:
    messages = build_messages(criterion, name, spec, description)
    base_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Two-stage generation: a 1B model reliably writes a short explanation but
    # often never reaches a "Verdict:" line within a shared token budget. So
    # generate the explanation first (prefilled with "Explanation:", cut at
    # the first newline), then make a second, separate short call - prefilled
    # with the explanation plus "Verdict:" - forced to emit just the verdict
    # word. This makes both parts independently reliable for a small model.
    last_error = None
    for attempt in range(retries + 1):
        sample = attempt > 0
        explanation_text = _generate(
            tokenizer, model, base_prompt + "Explanation:", max_new_tokens=60, sample=sample
        )
        explanation = explanation_text.strip().splitlines()[0].strip() if explanation_text.strip() else ""
        explanation = re.sub(r"^\s*explanation\s*:\s*", "", explanation, flags=re.IGNORECASE)
        if not explanation:
            last_error = ValueError("empty explanation")
            continue

        verdict_prompt = base_prompt + f"Explanation: {explanation}\nVerdict:"
        verdict_text = _generate(
            tokenizer, model, verdict_prompt, max_new_tokens=6, sample=sample
        )
        verdict_match = _VERDICT_RE.search("verdict:" + verdict_text)
        if not verdict_match:
            last_error = ValueError(f"no verdict found in output: {verdict_text[:100]!r}")
            continue
        return CriterionVerdict(explanation=explanation, verdict=verdict_match.group(1).lower())
    raise last_error


def judge_one(tokenizer, model, name: str, spec: str, description: str) -> dict:
    return {
        criterion: judge_criterion(tokenizer, model, criterion, name, spec, description)
        for criterion in CRITERIA
    }


def main() -> None:
    print(f"Loading {MODEL_NAME} on {DEVICE} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    df = pd.read_excel(DATA_FILE)

    for col in [f"judge_{c}_explanation" for c in CRITERIA] + \
               [f"judge_{c}_verdict" for c in CRITERIA] + \
               ["judge_latency_verdict", "judge_final_score"]:
        if col not in df.columns:
            df[col] = ""

    for i, row in df.iterrows():
        try:
            result = judge_one(
                tokenizer, model, row["name"], row["source_spec"], row["generated_description"]
            )
        except Exception as e:
            print(f"[{i + 1}/{len(df)}] {row['name']} FAILED: {e}")
            continue

        verdicts = {c: get_criterion(result, c).verdict for c in CRITERIA}
        verdicts["latency"] = latency_verdict(row["latency_ms"])

        for c in CRITERIA:
            crit = get_criterion(result, c)
            df.at[i, f"judge_{c}_explanation"] = crit.explanation
            df.at[i, f"judge_{c}_verdict"] = crit.verdict
        df.at[i, "judge_latency_verdict"] = verdicts["latency"]
        df.at[i, "judge_final_score"] = final_score(verdicts)

        print(f"[{i + 1}/{len(df)}] {row['name']} -> {df.at[i, 'judge_final_score']} "
              f"({verdicts})")

    df.to_excel(DATA_FILE, index=False)
    print(f"\nSaved judge verdicts for {len(df)} rows to {DATA_FILE}")


if __name__ == "__main__":
    main()
