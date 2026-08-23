"""
Task 6, experiment 2 - hybrid (dense + BM25) retrieval, on top of experiment 1's
winning k=10.

Isolates ONE variable from experiment 1 (assignment_03_exp3_topk10.xlsx): same k=10,
same chunk_size=1000/overlap=150/title-prefix-on chunks, only the retriever itself
changes from dense-only FAISS to HybridRetriever (RRF fusion of FAISS + BM25). Free
proxy in sweep_hybrid.py showed a large fact-hit-rate jump (50% -> 68%), which is why
this one gets the full judge run.

Reads:  eval_set.json, baseline_results.xlsx, faiss_index/, corpus/*
Writes: assignment_03_exp4_hybrid.xlsx, exp4_hybrid_summary.md
"""

import json

import pandas as pd

import evaluate
from evaluate import process_item, write_summary
from rag_pipeline import load_vectorstore
from build_index import load_documents, chunk_documents
from hybrid_retriever import HybridRetriever

evaluate.K = 10  # same as experiment 1, so only the retriever type differs


def main() -> None:
    with open("eval_set.json", "r", encoding="utf-8") as f:
        eval_set = json.load(f)
    baseline_df = pd.read_excel("baseline_results.xlsx").set_index("id")

    vectorstore = load_vectorstore()
    docs = load_documents()
    chunks = chunk_documents(docs)  # same baseline chunking as the on-disk index
    hybrid = HybridRetriever(chunks, vectorstore)

    rows = []
    for item in eval_set:
        print(f"[{item['id']}] hybrid k=10 running RAG... ", end="", flush=True)
        try:
            row = process_item(item, hybrid, baseline_df)
        except Exception as e:
            print(f"FAILED: {e}")
            continue
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_excel("assignment_03_exp4_hybrid.xlsx", index=False)
    print(f"\nSaved {len(df)} rows")

    write_summary(df, path="exp4_hybrid_summary.md")


if __name__ == "__main__":
    main()
