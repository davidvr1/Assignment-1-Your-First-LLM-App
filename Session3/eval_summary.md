# Task 5 summary: baseline vs. RAG


## ALL (n=30)

| metric | baseline | RAG |
|---|---|---|
| correctness (judge) | n/a* | 33% |
| context relevance (judge) | n/a | 23% |
| faithfulness (judge) | n/a | 97% |
| answer relevance (judge) | n/a | 97% |
| correctness (baseline, from Task 1 classification) | 0% | (see above) |
| refusal correct (code) | n/a | 37% |
| hit-rate@5 (code, answerable only, n=27) | n/a | 96% |
| avg latency (ms) | 1129 | 2275 |
| avg input tokens | 171 | 4072 |
| avg output tokens | 42 | 137 |

## EASY (n=20)

| metric | baseline | RAG |
|---|---|---|
| correctness (judge) | n/a* | 25% |
| context relevance (judge) | n/a | 25% |
| faithfulness (judge) | n/a | 95% |
| answer relevance (judge) | n/a | 100% |
| correctness (baseline, from Task 1 classification) | 0% | (see above) |
| refusal correct (code) | n/a | 30% |
| hit-rate@5 (code, answerable only, n=20) | n/a | 100% |
| avg latency (ms) | 1227 | 2488 |
| avg input tokens | 167 | 4112 |
| avg output tokens | 42 | 144 |

## HARD (n=10)

| metric | baseline | RAG |
|---|---|---|
| correctness (judge) | n/a* | 50% |
| context relevance (judge) | n/a | 20% |
| faithfulness (judge) | n/a | 100% |
| answer relevance (judge) | n/a | 90% |
| correctness (baseline, from Task 1 classification) | 0% | (see above) |
| refusal correct (code) | n/a | 50% |
| hit-rate@5 (code, answerable only, n=7) | n/a | 86% |
| avg latency (ms) | 931 | 1850 |
| avg input tokens | 180 | 3991 |
| avg output tokens | 43 | 122 |

\* baseline has no judge-based correctness re-run in this script; Task 1's baseline_final_class (refused/correct/hallucinated) is the baseline correctness signal, shown in the row below it.
