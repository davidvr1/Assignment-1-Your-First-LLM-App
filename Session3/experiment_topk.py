"""
Task 6, experiment 3 - raise top-K from 5 to 10.

Isolates ONE variable from the Task 5 baseline: same faiss_index/ (same
chunk_size=1000/overlap=150/title-prefix-on), only k changes. Re-runs the
full judge pipeline (not just the free hit-rate proxy) because sweep_hitrate.py
showed this is the one hypothesis worth spending the API budget on -- the
other two candidates were refuted by the free sweep and don't get a full run.

Reads:  eval_set.json, baseline_results.xlsx, faiss_index/
Writes: assignment_03_exp3_topk10.xlsx, exp3_topk10_summary.md
"""

import pandas as pd

import evaluate
from evaluate import process_item, write_summary
from rag_pipeline import load_vectorstore

evaluate.K = 10  # the one variable this experiment changes


def main() -> None:
    import json
    with open("eval_set.json", "r", encoding="utf-8") as f:
        eval_set = json.load(f)
    baseline_df = pd.read_excel("baseline_results.xlsx").set_index("id")
    vectorstore = load_vectorstore()

    rows = []
    for item in eval_set:
        print(f"[{item['id']}] k=10 running RAG... ", end="", flush=True)
        try:
            row = process_item(item, vectorstore, baseline_df)
        except Exception as e:
            print(f"FAILED: {e}")
            continue
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_excel("assignment_03_exp3_topk10.xlsx", index=False)
    print(f"\nSaved {len(df)} rows")

    write_summary(df, path="exp3_topk10_summary.md")


if __name__ == "__main__":
    main()
