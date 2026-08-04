import os
import json
import re
from openai import OpenAI
from pydantic import BaseModel, Field

client = OpenAI(
    base_url="https://api.anthropic.com/v1/",
    api_key=os.environ["ANTHROPIC_API_KEY"],
)

MODEL = "claude-haiku-4-5"

RESULTS_FILE = "results.txt"
open(RESULTS_FILE, "w", encoding="utf-8").close()  # clear file at start of run


def save(heading, text):
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(f"=== {heading} ===\n{text}\n\n")
    print(f"[saved to {RESULTS_FILE}]")


# --- Exercise 1: basic call ---------------------------------------------
resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
)
text = resp.choices[0].message.content
print("=== Exercise 1: basic call ===")
print(text)
save("Exercise 1: basic call", text)


# --- Exercise 1b, step 1: system prompt (persona) -----------------------
resp = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a pirate. Answer in one sentence."},
        {"role": "user", "content": "Say hello in one sentence."},
    ],
)
text = resp.choices[0].message.content
print("\n=== 1b.1: system prompt (pirate persona) ===")
print(text)
save("1b.1: system prompt (pirate persona)", text)


# --- Exercise 1b, step 2: temperature ------------------------------------
print("\n=== 1b.2: temperature=0 (twice, should match) ===")
lines = []
for _ in range(2):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Say hello in one sentence."}],
        temperature=0,
    )
    text = resp.choices[0].message.content
    print(text)
    lines.append(text)
save("1b.2: temperature=0 (twice)", "\n".join(lines))

print("\n=== 1b.2: temperature=1 (twice, should vary) ===")
lines = []
for _ in range(2):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Say hello in one sentence."}],
        temperature=1,
    )
    text = resp.choices[0].message.content
    print(text)
    lines.append(text)
save("1b.2: temperature=1 (twice)", "\n".join(lines))


# --- Exercise 1b, step 3: structured output ------------------------------
JSON_SYSTEM_PROMPT = (
    "Reply with JSON only, no prose, no markdown fences. "
    'The JSON object must have exactly two fields: "answer" (string) '
    'and "confidence" (number between 0 and 1).'
)

resp = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": JSON_SYSTEM_PROMPT},
        {"role": "user", "content": "Say hello in one sentence."},
    ],
    temperature=0,
)
raw = resp.choices[0].message.content.strip()
# Claude sometimes wraps JSON in ```json ... ``` fences despite instructions; strip them.
raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw).strip()
save("1b.3: raw JSON reply", raw)

# (a) raw JSON via the json module
data = json.loads(raw)
print("\n=== 1b.3a: parsed with json module ===")
print("answer:", data["answer"])
print("confidence:", data["confidence"])
save(
    "1b.3a: parsed with json module",
    f"answer: {data['answer']}\nconfidence: {data['confidence']}",
)


# (b) validated with Pydantic
class Answer(BaseModel):
    answer: str
    confidence: float = Field(ge=0, le=1)


validated = Answer.model_validate_json(raw)
print("\n=== 1b.3b: parsed with Pydantic ===")
print(validated)
print("answer:", validated.answer)
print("confidence:", validated.confidence)
save(
    "1b.3b: parsed with Pydantic",
    f"{validated}\nanswer: {validated.answer}\nconfidence: {validated.confidence}",
)
