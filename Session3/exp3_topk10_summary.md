# Task 5 summary: baseline vs. RAG


## ALL (n=30)

| metric | baseline | RAG |
|---|---|---|
| correctness (judge) | n/a* | 40% |
| context relevance (judge) | n/a | 33% |
| faithfulness (judge) | n/a | 97% |
| answer relevance (judge) | n/a | 93% |
| correctness (baseline, from Task 1 classification) | 0% | (see above) |
| refusal correct (code) | n/a | 43% |
| hit-rate@10 (code, answerable only, n=27) | n/a | 96% |
| avg latency (ms) | 1129 | 2635 |
| avg input tokens | 171 | 6710 |
| avg output tokens | 42 | 167 |

## EASY (n=20)

| metric | baseline | RAG |
|---|---|---|
| correctness (judge) | n/a* | 40% |
| context relevance (judge) | n/a | 40% |
| faithfulness (judge) | n/a | 95% |
| answer relevance (judge) | n/a | 100% |
| correctness (baseline, from Task 1 classification) | 0% | (see above) |
| refusal correct (code) | n/a | 45% |
| hit-rate@10 (code, answerable only, n=20) | n/a | 100% |
| avg latency (ms) | 1227 | 3046 |
| avg input tokens | 167 | 6769 |
| avg output tokens | 42 | 192 |

## HARD (n=10)

| metric | baseline | RAG |
|---|---|---|
| correctness (judge) | n/a* | 40% |
| context relevance (judge) | n/a | 20% |
| faithfulness (judge) | n/a | 100% |
| answer relevance (judge) | n/a | 80% |
| correctness (baseline, from Task 1 classification) | 0% | (see above) |
| refusal correct (code) | n/a | 40% |
| hit-rate@10 (code, answerable only, n=7) | n/a | 86% |
| avg latency (ms) | 931 | 1814 |
| avg input tokens | 180 | 6591 |
| avg output tokens | 43 | 118 |

\* baseline has no judge-based correctness re-run in this script; Task 1's baseline_final_class (refused/correct/hallucinated) is the baseline correctness signal, shown in the row below it.