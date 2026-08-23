"""
Task 6, experiment 2, step 0 - free fact-hit check for hybrid retrieval
(zero API calls), same proxy metric as sweep_hitrate.py.

Reads:  corpus/*, eval_set.json, faiss_index/ (Task 3's dense index, prefix on,
        chunk_size=1000/150 -- same corpus chunks the hybrid retriever indexes)
Writes: sweep_hybrid_results.md
"""

import json

from build_index import load_documents, chunk_documents
from rag_pipeline import load_vectorstore
from hybrid_retriever import HybridRetriever
from sweep_hitrate import fact_hit


def evaluate_config(retriever, eval_set, k) -> float:
    hits = []
    for item in eval_set:
        if not item["answerable"]:
            continue
        chunks = retriever.similarity_search(item["question"], k=k)
        h = fact_hit(chunks, item["reference_answer"])
        if h is not None:
            hits.append(h)
    return sum(hits) / len(hits) if hits else float("nan")


def main() -> None:
    with open("eval_set.json", "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    print("Loading dense index + rebuilding chunk list for BM25...")
    vectorstore = load_vectorstore()
    docs = load_documents()
    chunks = chunk_documents(docs)  # baseline config: 1000/150, prefix on
    hybrid = HybridRetriever(chunks, vectorstore)

    results = []
    for label, retriever, k in [
        ("dense-only, k=10 (exp.1 winner)", vectorstore, 10),
        ("hybrid (dense+BM25 RRF), k=10", hybrid, 10),
    ]:
        rate = evaluate_config(retriever, eval_set, k)
        results.append((label, rate))
        print(f"{label:35s} fact_hit_rate={rate:.0%}")

    # Also check the id-24 identifier question specifically, since that's the
    # concrete regression this experiment targets.
    id24 = next(i for i in eval_set if i["id"] == 24)
    print(f"\nid=24 (exact identifier) question: {id24['question']}")
    for label, retriever in [("dense k=10", vectorstore), ("hybrid k=10", hybrid)]:
        chunks_ret = retriever.similarity_search(id24["question"], k=10)
        hit = fact_hit(chunks_ret, id24["reference_answer"])
        print(f"  {label}: fact_hit={hit}")

    with open("sweep_hybrid_results.md", "w", encoding="utf-8") as f:
        f.write("# Task 6 exp.2 free fact-hit check: dense vs. hybrid (zero API calls)\n\n")
        f.write("| config | fact-level hit-rate |\n|---|---|\n")
        for label, rate in results:
            f.write(f"| {label} | {rate:.0%} |\n")
    print("\nSaved sweep_hybrid_results.md")


if __name__ == "__main__":
    main()
