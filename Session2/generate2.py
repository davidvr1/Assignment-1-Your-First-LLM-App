"""
Task 2 (online model variant) - Generate a product description for every
product using Anthropic's Claude Sonnet 5, in Hebrew, and record latency +
token counts alongside each output.

Reads:  electrical_items_dataset.xlsx   (id, category, name, description)
Writes: electrical_items_dataset2.xlsx  (generation results + blank rubric columns)
"""

import os
import re
import time
from pathlib import Path

import pandas as pd
from anthropic import Anthropic

MODEL_NAME = "claude-sonnet-5"
SOURCE_FILE = "electrical_items_dataset.xlsx"
OUTPUT_FILE = "electrical_items_dataset2.xlsx"
KEY_FILE = Path(__file__).parent / "key.txt"


def load_api_key() -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        return api_key
    if KEY_FILE.exists():
        match = re.search(r'ANTHROPIC_API_KEY="([^"]+)"', KEY_FILE.read_text())
        if match:
            return match.group(1)
    raise RuntimeError(
        f"ANTHROPIC_API_KEY not set and not found in {KEY_FILE}"
    )

SYSTEM_PROMPT = (
    "אתה קופירייטר עבור קמעונאי מוצרי חשמל. בהינתן שם מוצר ופסקת מפרט טכני, "
    "כתוב תיאור שיווקי משכנע לעמוד המוצר, בעברית בלבד.\n\n"
    "כללים:\n"
    "- אורך: 50-90 מילים. לא פחות, לא יותר.\n"
    "- היצמדות לעובדות: השתמש רק בעובדות המופיעות במפרט שסופק. אל תמציא "
    "תכונות, מספרים, מחירים או יכולות שלא צוינו. ביטויים כלליים לא-עובדתיים "
    "(למשל \"מתאים לכל מטבח\") מותרים, אך כל טענה קונקרטית חייבת להתבסס על "
    "המפרט.\n"
    "- טון: חם, בטוח, שפה פשוטה. ללא הגזמות, ללא סימני קריאה, ללא סופרלטיבים "
    "שלא ניתן לאמת (\"הכי טוב\", \"פורץ דרך\").\n"
    "- פורמט הפלט: טקסט רגיל בלבד, ללא כותרות, ללא תבליטים, ללא markdown, "
    "רק טקסט התיאור עצמו.\n"
    "- שפה: כל הפלט חייב להיות בעברית בלבד, ללא מילים או תווים בשפות אחרות "
    "(לרבות סינית, אנגלית וכדומה), וללא שורת כותרת בתחילת התשובה."
)

USER_TEMPLATE = "שם המוצר: {name}\nמפרט: {spec}"


def build_prompt(name: str, spec: str) -> str:
    return USER_TEMPLATE.format(name=name, spec=spec)


def main() -> None:
    products = pd.read_excel(SOURCE_FILE)

    client = Anthropic(api_key=load_api_key())

    rows = []
    for i, product in products.iterrows():
        prompt = build_prompt(product["name"], product["description"])

        start = time.perf_counter()
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        latency_ms = (time.perf_counter() - start) * 1000

        generated_description = "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

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
