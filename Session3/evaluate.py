"""
Task 5 - Evaluate: RAG vs. baseline.

Reads:  eval_set.json, baseline_results.xlsx (Task 1), faiss_index/ (Task 3)
Writes: assignment_03.xlsx (one row per question, baseline + RAG + all metrics)
        eval_summary.md (baseline-vs-RAG table, easy/hard slices)

Runs the full RAG pipeline over all 30 eval questions, computes hit-rate@K in
code, runs the 4 Sonnet judges (context relevance / faithfulness / answer
relevance / correctness) on both the RAG answer, and does refusal-correctness
bookkeeping in code. Baseline numbers are pulled from Task 1's
baseline_results.xlsx rather than re-run.
"""

import json
import time

import pandas as pd

from rag_pipeline import _run_rag, load_vectorstore, REFUSAL_SENTENCE
from judges import (
    judge_context_relevance,
    judge_faithfulness,
    judge_answer_relevance,
    judge_correctness,
)

K = 5


def hit_at_k(chunks, evidence_doc: str) -> bool:
    """Did any retrieved chunk come from evidence_doc? For multi-hop
    questions evidence_doc lists several files joined with ' + ' -- a hit
    counts if the retrieved set covers at least one of them (matching the
    spirit of the metric: "did retrieval surface a document that actually
    holds evidence", not "did it surface ALL of them" -- that stricter
    version is worth calling out separately for the 2 multi-hop questions).
    """
    evidence_docs = {d.strip() for d in evidence_doc.split("+")}
    retrieved_docs = {c.metadata.get("doc_name") for c in chunks}
    return bool(evidence_docs & retrieved_docs)


def chunks_to_text(chunks) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[{i}] ({c.metadata.get('doc_name')}, עמ' {c.metadata.get('page')}): {c.page_content}")
    return "\n---\n".join(parts)


def main() -> None:
    with open("eval_set.json", "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    baseline_df = pd.read_excel("baseline_results.xlsx").set_index("id")

    vectorstore = load_vectorstore()

    rows = []
    for item in eval_set:
        qid = item["id"]
        print(f"[{qid}] running RAG... ", end="", flush=True)

        try:
            row = process_item(item, vectorstore, baseline_df)
        except Exception as e:
            print(f"FAILED: {e}")
            continue
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_excel("assignment_03.xlsx", index=False)
    print(f"\nSaved {len(df)} rows to assignment_03.xlsx")

    write_summary(df)


def process_item(item: dict, vectorstore, baseline_df: pd.DataFrame) -> dict:
    qid = item["id"]
    rag = _run_rag(item["question"], K, vectorstore)
    ga = rag["grounded_answer"]
    chunks_text = chunks_to_text(rag["chunks"])

    hit = hit_at_k(rag["chunks"], item["evidence_doc"]) if item["answerable"] else None

    # Refusal correctness (code): compare what the system DID (answered
    # or refused) against what it SHOULD have done (item["answerable"]).
    false_refusal = item["answerable"] and not ga.answered
    false_answer = (not item["answerable"]) and ga.answered
    refusal_correct = not false_refusal and not false_answer

    print("judging... ", end="", flush=True)
    cr = judge_context_relevance(item["question"], chunks_text)
    fa = judge_faithfulness(ga.answer, chunks_text)
    ar = judge_answer_relevance(item["question"], ga.answer)
    co = judge_correctness(ga.answer, item["reference_answer"])
    print("done")

    baseline_row = baseline_df.loc[qid]

    return {
        "id": qid,
        "question": item["question"],
        "reference_answer": item["reference_answer"],
        "evidence_doc": item["evidence_doc"],
        "evidence_page": item["evidence_page"],
        "answerable": item["answerable"],
        "difficulty": item["difficulty"],

        "baseline_answer": baseline_row["baseline_answer"],
        "baseline_class": baseline_row["baseline_final_class"],
        "baseline_latency_ms": baseline_row["latency_ms"],
        "baseline_input_tokens": baseline_row["input_tokens"],
        "baseline_output_tokens": baseline_row["output_tokens"],

        "rag_answer": ga.answer,
        "rag_sources": " | ".join(ga.sources),
        "rag_evidence": " | ".join(ga.evidence),
        "rag_answered": ga.answered,
        "invalid_citations": str(rag["invalid_citations"]),
        "retrieved_docs": " | ".join(sorted({c.metadata.get("doc_name") for c in rag["chunks"]})),

        "hit_at_k": hit,
        "false_refusal": false_refusal,
        "false_answer": false_answer,
        "refusal_correct": refusal_correct,

        "judge_context_relevance_verdict": cr.verdict,
        "judge_context_relevance_explanation": cr.explanation,
        "judge_faithfulness_verdict": fa.verdict,
        "judge_faithfulness_explanation": fa.explanation,
        "judge_answer_relevance_verdict": ar.verdict,
        "judge_answer_relevance_explanation": ar.explanation,
        "judge_correctness_verdict": co.verdict,
        "judge_correctness_explanation": co.explanation,

        "rag_latency_ms": round(rag["latency_ms"], 1),
        "rag_input_tokens": rag["input_tokens"],
        "rag_output_tokens": rag["output_tokens"],
    }


def pct(series) -> str:
    series = series.dropna()
    if len(series) == 0:
        return "n/a"
    return f"{100 * series.mean():.0f}%"


def write_summary(df: pd.DataFrame, path: str = "eval_summary.md") -> None:
    lines = ["# Task 5 summary: baseline vs. RAG\n"]
    for slice_name, sub in [("ALL", df), ("EASY", df[df.difficulty == "easy"]), ("HARD", df[df.difficulty == "hard"])]:
        lines.append(f"\n## {slice_name} (n={len(sub)})\n")
        lines.append("| metric | baseline | RAG |")
        lines.append("|---|---|---|")
        lines.append(f"| correctness (judge) | n/a* | {pct(sub['judge_correctness_verdict'])} |")
        lines.append(f"| context relevance (judge) | n/a | {pct(sub['judge_context_relevance_verdict'])} |")
        lines.append(f"| faithfulness (judge) | n/a | {pct(sub['judge_faithfulness_verdict'])} |")
        lines.append(f"| answer relevance (judge) | n/a | {pct(sub['judge_answer_relevance_verdict'])} |")
        base_correct = (sub["baseline_class"] == "correct").mean() if len(sub) else float("nan")
        lines.append(f"| correctness (baseline, from Task 1 classification) | {100*base_correct:.0f}% | (see above) |")
        lines.append(f"| refusal correct (code) | n/a | {pct(sub['refusal_correct'])} |")
        hit_sub = sub[sub["hit_at_k"].notna()]
        lines.append(f"| hit-rate@{K} (code, answerable only, n={len(hit_sub)}) | n/a | {pct(hit_sub['hit_at_k'])} |")
        lines.append(f"| avg latency (ms) | {sub['baseline_latency_ms'].mean():.0f} | {sub['rag_latency_ms'].mean():.0f} |")
        lines.append(f"| avg input tokens | {sub['baseline_input_tokens'].mean():.0f} | {sub['rag_input_tokens'].mean():.0f} |")
        lines.append(f"| avg output tokens | {sub['baseline_output_tokens'].mean():.0f} | {sub['rag_output_tokens'].mean():.0f} |")

    lines.append("\n\\* baseline has no judge-based correctness re-run in this script; "
                  "Task 1's baseline_final_class (refused/correct/hallucinated) is the "
                  "baseline correctness signal, shown in the row below it.")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
