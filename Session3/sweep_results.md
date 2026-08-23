# Task 6 free hit-rate sweep (fact-level, zero API calls)

| config | n_chunks | fact-level hit-rate |
|---|---|---|
| baseline (Task 3/5: 1000/150, prefix on, k=5) | 92 | 45% |
| no title prefix, k=5 | 92 | 32% |
| no title prefix, k=8 | 92 | 41% |
| chunk_size=500/75, prefix on, k=5 | 179 | 18% |
| chunk_size=500/75, no prefix, k=5 | 179 | 18% |
| chunk_size=300/50, no prefix, k=5 | 323 | 14% |
| prefix on, k=8 (top-K only, isolated) | 92 | 45% |
| prefix on, k=10 | 92 | 50% |
