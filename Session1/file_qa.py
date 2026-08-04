import os
import sys
from openai import OpenAI

client = OpenAI(
    base_url="https://api.anthropic.com/v1/",
    api_key=os.environ["ANTHROPIC_API_KEY"],
)

MODEL = "claude-haiku-4-5"

SYSTEM_PROMPT = (
    "You answer questions using ONLY from the document provided by the user. "
    "Do not use outside knowledge and do not guess. "
    "When you answer, quote the exact passage from the document that supports "
    'your answer as a "Reference:" line. '
    "If the answer is not contained in the document, reply with exactly this "
    "sentence and nothing else: \"I can't find that in the document.\""
)

if len(sys.argv) >= 2:
    file_path = sys.argv[1]
else:
    file_path = input("Path to file: ").strip()

if len(sys.argv) >= 3:
    question = sys.argv[2]
else:
    question = input("Question: ").strip()

with open(file_path, "r", encoding="utf-8") as f:
    document = f.read()

user_message = f"Document:\n\"\"\"\n{document}\n\"\"\"\n\nQuestion: {question}"

resp = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ],
    temperature=0,
)

answer = resp.choices[0].message.content
print(answer)

RESULTS_FILE = "qa_results.txt"
with open(RESULTS_FILE, "a", encoding="utf-8") as f:
    f.write(f"File: {file_path}\nQuestion: {question}\nAnswer: {answer}\n\n")
print(f"[saved to {RESULTS_FILE}]")
