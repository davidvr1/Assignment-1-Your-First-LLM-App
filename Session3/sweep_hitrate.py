"""
Task 6, step 0 - Free hit-rate sweep (zero API calls).

The assignment's own advice: sweep chunk_size/top-K/other retrieval knobs
against hit-rate before spending judge-call budget on the full re-measurement.
This script builds several candidate FAISS indices IN MEMORY (never touching
the Task 3 faiss_index/ on disk) and scores each against a FACT-LEVEL hit
metric, not Task 5's doc-level one.

Why not reuse Task 5's hit_at_k? Task 5 found it saturated at 96% (doc-level:
"did any retrieved chunk come from the right FILE") while correctness sat at
33% -- the metric was too easy to move the needle on and was hiding the real
failure (right file, wrong chunk). fact_hit() below checks whether a
retrieved CHUNK actually contains the distinguishing fact from the reference
answer (a number, a plan code) -- much closer to "could the generator have
answered from this," and the numbers move a lot more, which is what makes it
useful for choosing between configs.

Reads:  corpus/*, eval_set.json
Writes: sweep_results.md
"""

import json
import re

from build_index import (
    load_documents,
    chunk_documents,
    BgeQueryPrefixEmbeddings,
    EMBEDDING_MODEL,
)
from langchain_community.vectorstores import FAISS

# Salient-token pattern: numbers (with optional currency/percent marks) and
# plan-code-like tokens (letters-dash-digits, e.g. MZ-2025, MA-ADIF-24).
# These are exactly the things embeddings are weak at (per the assignment's
# "exact identifier" warning) and exactly what a reference answer's real
# content is made of in this corpus.
TOKEN_PATTERN = re.compile(r"\d[\d,.]*%?|\b[A-Z]{2,}-[A-Z0-9-]+\b")


def salient_tokens(text: str) -> set[str]:
    return {t for t in TOKEN_PATTERN.findall(text) if len(t) >= 2}


def fact_hit(chunks, reference_answer: str) -> bool:
    """True if ANY retrieved chunk contains at least one salient token
    (number or plan code) from the reference answer. A crude but free proxy
    for 'the answer-bearing chunk was actually retrieved', not just
    'a chunk from the right document was retrieved'.
    """
    tokens = salient_tokens(reference_answer)
    if not tokens:
        return None  # reference answer has no numbers/codes to check against
    combined = "\n".join(c.page_content for c in chunks)
    return any(tok in combined for tok in tokens)


def evaluate_config(vectorstore, eval_set, k) -> float:
    hits = []
    for item in eval_set:
        if not item["answerable"]:
            continue
        chunks = vectorstore.similarity_search(item["question"], k=k)
        h = fact_hit(chunks, item["reference_answer"])
        if h is not None:
            hits.append(h)
    return sum(hits) / len(hits) if hits else float("nan")


def main() -> None:
    with open("eval_set.json", "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    print("Loading embedding model once, reused across all configs...")
    embeddings = BgeQueryPrefixEmbeddings(model_name=EMBEDDING_MODEL)
    docs = load_documents()

    configs = [
        # (label, chunk_size, chunk_overlap, add_title_prefix, k)
        ("baseline (Task 3/5: 1000/150, prefix on, k=5)", 1000, 150, True, 5),
        ("no title prefix, k=5", 1000, 150, False, 5),
        ("no title prefix, k=8", 1000, 150, False, 8),
        ("chunk_size=500/75, prefix on, k=5", 500, 75, True, 5),
        ("chunk_size=500/75, no prefix, k=5", 500, 75, False, 5),
        ("chunk_size=300/50, no prefix, k=5", 300, 50, False, 5),
        ("prefix on, k=8 (top-K only, isolated)", 1000, 150, True, 8),
        ("prefix on, k=10", 1000, 150, True, 10),
    ]

    results = []
    for label, chunk_size, overlap, prefix, k in configs:
        chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=overlap, add_title_prefix=prefix)
        vs = FAISS.from_documents(chunks, embeddings)
        rate = evaluate_config(vs, eval_set, k)
        results.append((label, len(chunks), rate))
        print(f"{label:45s} n_chunks={len(chunks):3d}  fact_hit_rate={rate:.0%}")

    with open("sweep_results.md", "w", encoding="utf-8") as f:
        f.write("# Task 6 free hit-rate sweep (fact-level, zero API calls)\n\n")
        f.write("| config | n_chunks | fact-level hit-rate |\n|---|---|---|\n")
        for label, n_chunks, rate in results:
            f.write(f"| {label} | {n_chunks} | {rate:.0%} |\n")
    print("\nSaved sweep_results.md")


if __name__ == "__main__":
    main()
