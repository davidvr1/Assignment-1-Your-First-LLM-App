# Task 5 summary: baseline vs. RAG


## ALL (n=30)

| metric | baseline | RAG |
|---|---|---|
| correctness (judge) | n/a* | 57% |
| context relevance (judge) | n/a | 63% |
| faithfulness (judge) | n/a | 100% |
| answer relevance (judge) | n/a | 100% |
| correctness (baseline, from Task 1 classification) | 0% | (see above) |
| refusal correct (code) | n/a | 57% |
| hit-rate@10 (code, answerable only, n=27) | n/a | 93% |
| avg latency (ms) | 1129 | 3041 |
| avg input tokens | 171 | 6887 |
| avg output tokens | 42 | 213 |

## EASY (n=20)

| metric | baseline | RAG |
|---|---|---|
| correctness (judge) | n/a* | 50% |
| context relevance (judge) | n/a | 65% |
| faithfulness (judge) | n/a | 100% |
| answer relevance (judge) | n/a | 100% |
| correctness (baseline, from Task 1 classification) | 0% | (see above) |
| refusal correct (code) | n/a | 50% |
| hit-rate@10 (code, answerable only, n=20) | n/a | 90% |
| avg latency (ms) | 1227 | 2802 |
| avg input tokens | 167 | 6896 |
| avg output tokens | 42 | 206 |

## HARD (n=10)

| metric | baseline | RAG |
|---|---|---|
| correctness (judge) | n/a* | 70% |
| context relevance (judge) | n/a | 60% |
| faithfulness (judge) | n/a | 100% |
| answer relevance (judge) | n/a | 100% |
| correctness (baseline, from Task 1 classification) | 0% | (see above) |
| refusal correct (code) | n/a | 70% |
| hit-rate@10 (code, answerable only, n=7) | n/a | 100% |
| avg latency (ms) | 931 | 3521 |
| avg input tokens | 180 | 6869 |
| avg output tokens | 43 | 228 |

\* baseline has no judge-based correctness re-run in this script; Task 1's baseline_final_class (refused/correct/hallucinated) is the baseline correctness signal, shown in the row below it.