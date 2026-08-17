"""
Task 2 - Generate a product description for every product using a small
local model (Qwen2.5-0.5B-Instruct), and record latency + token counts
alongside each output.

Reads:  electrical_items_dataset.xlsx  (id, category, name, description)
Writes: assignment_02.xlsx             (generation results + blank rubric columns)
"""

import time

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SOURCE_FILE = "electrical_items_dataset.xlsx"
OUTPUT_FILE = "assignment_02.xlsx"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SYSTEM_PROMPT = (
    "You are a copywriter for an electronics retailer. Given a product name "
    "and a technical spec paragraph, write persuasive marketing copy for the "
    "product page.\n\n"
    "Rules:\n"
    "- Length: 50-90 words. Not fewer, not more.\n"
    "- Grounding: only use facts present in the supplied spec. Do not invent "
    "features, numbers, prices, or capabilities that are not stated. Generic "
    "non-factual phrases (e.g. \"great for any kitchen\") are allowed, but "
    "every concrete claim must trace back to the spec.\n"
    "- Tone: warm, confident, plain language. No hype, no exclamation marks, "
    "no unverifiable superlatives (\"the best\", \"revolutionary\").\n"
    "- Output format: plain prose, no headings, no bullet points, no markdown, "
    "just the description text itself."
)

USER_TEMPLATE = "Product name: {name}\nSpec: {spec}"


def build_messages(name: str, spec: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(name=name, spec=spec)},
    ]


def main() -> None:
    products = pd.read_excel(SOURCE_FILE)

    print(f"Loading {MODEL_NAME} on {DEVICE} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    rows = []
    for i, product in products.iterrows():
        messages = build_messages(product["name"], product["description"])

        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
        input_tokens = inputs["input_ids"].shape[1]

        start = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        latency_ms = (time.perf_counter() - start) * 1000

        generated_ids = output_ids[0][input_tokens:]
        output_tokens = generated_ids.shape[0]
        generated_description = tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()

        rows.append(
            {
                "id": product["id"],
                "category": product["category"],
                "name": product["name"],
                "source_spec": product["description"],
                "generated_description": generated_description,
                "latency_ms": round(latency_ms, 1),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        )
        print(f"[{i + 1}/{len(products)}] {product['name']} "
              f"({latency_ms:.0f} ms, {output_tokens} out tokens)")

    results = pd.DataFrame(rows)

    for criterion in ["Fluency", "Grammar", "Tone", "Length", "Grounding", "Latency"]:
        results[criterion] = ""
    results["final_score"] = ""

    results.to_excel(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(results)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
