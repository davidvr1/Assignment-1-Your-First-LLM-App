# Task 6 — two improvement cycles

Both experiments target failures actually observed in the Task 5 matrix
(`assignment_04.xlsx`, `task5_sliced_table.csv`), re-run only on the task
types each hypothesis targets (`unanswerable` + `multi_hop`), 5 runs each,
per the assignment's "sweep cheaply" note. Runner: `task6_experiments.py`.
Traces: `traces_task6/`. Raw per-run rows: `task6_exp1_results.json`,
`task6_exp2_results.json`.

Task 5 baseline for reference (agent, Haiku, v1 prompt, from
`task5_sliced_table.csv`):

| type | success | avg_tool_calls | avg_tokens | p95 | cap_breaches |
|---|---|---|---|---|---|
| unanswerable | 9/15 | 3.07 | 30627 | 15389ms | 0 |
| multi_hop | 17/30 | 3.73 | 68201 | 27777ms | 5 |

---

## Experiment 1 — system-prompt fix (structural, not a model swap)

### Hypothesis (written before the run)

Two Task 5 failures share one root cause: the agent's stopping rule was the
vague "2–4 tool calls, don't keep retrying" (`SYSTEM_PROMPT` rule 5), and its
refusal wording was left to the model's own phrasing.

- **m06** (multi_hop, 0/5 in Task 5) looped: 7 `retrieve_policy_docs` calls
  with re-worded queries, all returning near-identical passages, before
  hitting the 120k token-budget net at 155,787 tokens
  (`traces/m06_agent_run1.jsonl`, `terminal_state=cap_breached`, `steps=8`).
- **u02** (unanswerable, 0/5 in Task 5) refused *in substance* every run —
  "ביטוח נסיעות... הוא מוצר ביטוח מסחרי נפרד... **אינו מופיע במסמכים
  שלי**" — but never in a wording either the agent's own post-hoc
  refusal-marker check or `eval_runner.score_task`'s regex
  (`r"לא (מצאתי|ניתן|יכול)|מצטער"`) recognized, so every run scored as a
  false answer even though the agent correctly declined to guess.

**Hypothesis:** replacing the vague retry rule with a hard numeric cap (max
4 `retrieve_policy_docs` calls) and requiring the exact verbatim refusal
sentence already used by the RAG pipeline (`REFUSAL_SENTENCE`) will (a) cut
off m06-style loops before they burn the token budget, and (b) make refusal
wording consistent and exact-string-checkable, the same way Task 5 already
checks RAG's refusals.

### Change (one variable: `SYSTEM_PROMPT` → `SYSTEM_PROMPT_V2` in `agent.py`)

Old rule 5: *"סרב/י בנימוס... אחרי חיפוש סביר (2-4 קריאות כלי לכל היותר)..."*
New rules 5–6: hard "no more than 4 `retrieve_policy_docs` calls, ever" +
mandatory verbatim `"לא מצאתי את התשובה במסמכים שסופקו."` on refusal.
Model, tools, nets, guardrails flag: all held constant.

### Results (baseline v1 prompt vs. v2 prompt, Haiku, 5 runs × 9 tasks)

| type | config | success | avg_tool_calls | avg_tokens | p50 | p95 | cap_breaches |
|---|---|---|---|---|---|---|---|
| unanswerable | v1 (baseline) | 9/15 | 3.27 | 33,900 | 13,424ms | 14,396ms | 0 |
| unanswerable | **v2** | **15/15** | 3.20 | 33,108 | 13,744ms | 14,870ms | 0 |
| multi_hop | v1 (baseline) | 17/30 | 3.80 | 69,581 | 13,164ms | 27,166ms | **5** |
| multi_hop | **v2** | 17/30 | 3.37 | **42,062** | 12,502ms | **14,841ms** | **0** |
| **TOTAL** | v1 | 26/45 | 3.62 | 57,687 | 13,424ms | 27,022ms | 5 |
| **TOTAL** | **v2** | **32/45** | 3.31 | **39,077** | 12,506ms | **14,869ms** | **0** |

**m06 specifically**, v2 (`traces_task6/m06_exp1_v2_haiku_run1.jsonl`):
4 retrieval calls, then a clean refusal — `steps=5`, `tokens=41,378`,
`terminal_state=refused` — versus v1's 8 steps, 155,787 tokens, and a
`cap_breached` non-answer. Every one of the 5 v2 runs on m06 stopped at
exactly 4 calls and refused; none hit the cap.

### Conclusion

**Confirmed, ship it.** unanswerable success 9/15 → 15/15 (+40pp, the exact
fix predicted — u02's substance-correct-but-unrecognized refusals now match
the canonical sentence). multi_hop success is flat (17/30 → 17/30, m06
still fails all 5 runs) — the hard cap doesn't make the agent *find* the
fact any more than the loop did, it just stops it from burning budget while
failing. But that "flat success, cheaper failure" is itself the win: zero
cap breaches (was 5/30), p95 latency -47% (27.2s → 14.8s), average tokens
-32% overall and -40% on multi_hop specifically. This is a pure
structural/prompt fix — no model change — and it's the one required to run
first. Kept as the new baseline for Experiment 2.

---

## Experiment 2 — model upgrade (Haiku → Sonnet), multi_hop only

### Hypothesis (written before the run, after Experiment 1)

Per the assignment's lever ordering, a model swap is only justified once
the cheap structural fixes are ruled out — Experiment 1 already fixed the
loop/refusal-wording failures, so the two multi_hop tasks still failing 0/5
after Experiment 1 (**m01**, **m06**) are candidates for a genuine
reasoning/retrieval-quality gap rather than a stopping-rule problem.
**Hypothesis:** claude-sonnet-5, same v2 prompt/tools/nets, will solve some
of what Haiku still can't on this slice, at a cost/latency premium.

### Change (one variable: `model` argument, `claude-haiku-4-5` →
`claude-sonnet-5`; v2 prompt, tools, nets held constant)

Hit one real bug on the way: `ChatAnthropic(temperature=0, ...)` raises
`400: temperature is deprecated for this model` on `claude-sonnet-5`.
Fixed in `agent.py`'s `_build_agent` by only passing `temperature` for
models that still accept it — not a result, just noted so the fix is
traceable if this file is read later.

### Results (v2 prompt, multi_hop only, 5 runs × 6 tasks)

| model | success | avg_tool_calls | avg_tokens | p50 | p95 |
|---|---|---|---|---|---|
| Haiku (exp1 baseline) | 17/30 | 3.37 | 42,062 | 12,502ms | 14,841ms |
| **Sonnet** | **15/30** | **2.53** | **30,627** | 15,136ms | **21,605ms** |

Per-task success (5 runs each):

| task | Haiku | Sonnet |
|---|---|---|
| m01 | 0/5 | 0/5 |
| m02 | 2/5 | **0/5** |
| m03 | 5/5 | 4/5 |
| m04 | 5/5 | 5/5 |
| m05 | 5/5 | 5/5 |
| m06 | 0/5 | **1/5** |

### Conclusion

**Not confirmed — do not ship the swap.** Sonnet is *cheaper per call*
(fewer, more decisive tool calls: 2.53 vs 3.37; -27% tokens) but *slower*
(p95 +46%, 14.8s → 21.6s) and **less reliable overall** (15/30 vs 17/30).
It regressed m02 from 2/5 to 0/5 and only marginally helped the hardest
task, m06 (0/5 → 1/5, still failing 4 of 5 runs) — for the same underlying
reason Haiku failed it: `retrieve_policy_docs` keeps surfacing the same
document-intro chunk regardless of query rewording
(`traces_task6/m06_exp1_v2_haiku_run1.jsonl` and the equivalent Sonnet
trace both show 3–4 near-duplicate top-1 results before giving up). That's
a **retrieval-quality problem**, not a reasoning-capacity one — a bigger
model can't out-reason a corpus chunk it never receives. This is exactly
the ordering the assignment's lever table predicts: model upgrade is for
"genuine reasoning failures... after you've ruled out" the cheaper fixes,
and here it wasn't one. The actionable next lever (not run, out of scope
for this assignment) would be improving `retrieve_policy_docs`'s chunking
or re-ranking so a rephrased query stops returning the same chunk — a
retrieval-layer fix, the same category of problem Assignment 3's k/hybrid
experiments addressed, not an agent-layer one.

---

## Summary

| experiment | lever | verdict |
|---|---|---|
| 1 | system prompt: hard retrieval cap + canonical refusal sentence | **Ship.** unanswerable +40pp, cap breaches 5→0, p95 -47%, tokens -32%, multi_hop success unchanged |
| 2 | model upgrade: Haiku → Sonnet on multi_hop | **Don't ship.** Success -2/30, p95 +46%; the remaining failures are a retrieval gap, not a reasoning gap |

Net effect of both experiments together vs. the Task 5 agent baseline on
these two slices: success 26/45 → 32/45 (using v2-Haiku, the version
actually kept), cap breaches 5→0, p95 latency roughly halved. The model
swap was tried, as required, and correctly rejected with numbers rather
than assumed to help.
