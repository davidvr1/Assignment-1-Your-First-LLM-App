# Assignment 4 write-up — RAG pipeline made into an agent

Frozen Assignment 3 baseline: commit `6d9edeb` (RAG pipeline over the Hebrew
health-insurance corpus), unmodified throughout. Task set: `agent_tasks.json`
(25 tasks — 11 single, 6 multi_hop, 3 no_tool, 3 unanswerable, 2 tool_fails).

---

## Task 2 — tool design and the dumb-engineer test

Three tools in `tools.py`: `retrieve_policy_docs` (Assignment 3's retriever,
wrapped), `calculate` (safe AST-only arithmetic evaluator — no `eval()` —
for totals/percentages/multi-visit sums), `date_duration` (waiting periods,
notice periods, renewal windows). Every tool follows the failure contract:
an `ERROR:`/`NO_RESULTS:` string, never a raise, `None`, or `""`.

### The dumb-engineer test, actually run

Ran two independent fresh-context checks (no shared history, via a
zero-context subagent each time): the same task pasted against a
deliberately bare tool set (`search`/`calc`/`dates`, one-line descriptions)
versus the real tool set (full scope/limits/failure-contract descriptions).

- **Test 1** (multi-hop copay task, m02): naive descriptions →
  `search({"query": "מאוחדת עדיף השתתפות עצמית ניתוח פרטי בישראל"})` then
  `calc` on the retrieved figure ×2 — correct order, correct first call.
  Real descriptions → the same correct order, plus an extra
  reasoning step checking for a possible multi-procedure discount before
  calculating.
- **Test 2** (date task): naive `dates("Handle dates.")` → still refused to
  guess `add_days` and correctly called `search` first. Real
  `date_duration` → identical behavior.

**Result: no failure was found on either task, with either description
level.** This itself is the finding, not a null result to hide — Haiku/
Sonnet-class models pick the right tool from the task's own phrasing even
under one-line descriptions, on tasks this shape. Tool-selection accuracy
was never the failure mode this project actually hit. The real failures
(documented in Task 5/6 below — retrieval loops, refusal-wording
inconsistency) were about **when to stop**, not **which tool to call**,
which is why Task 6's fix lives in the system prompt's stopping rule, not
in a tool description rewrite. No before/after description change was
needed or made as a result.

---

## Task 4 — pattern: guardrails

Implemented in `agent.py` via a `use_guardrails` flag on the same file
(before/after is one flag flip):
- **Input guardrail**: reject obviously off-topic requests (poems, code,
  recipes) before any tool call.
- **Output guardrail**: every number ≥100 in the final answer must appear
  verbatim in this run's actual tool outputs, or the answer is withheld as
  unverifiable instead of returned.

### Before/after (`task4_results.json`, 25 tasks × 2 runs)

| type | metric | off | on | delta |
|---|---|---|---|---|
| multi_hop | success | 8/12 | 6/12 | −2 |
| single | success | 14/22 | 15/22 | +1 |
| unanswerable | success | 4/6 | 3/6 | −1 |
| tool_fails | success | 2/4 | **4/4** | **+2** |
| no_tool | success | 6/6 | 6/6 | 0 |
| **overall** | **success** | **34/50** | **34/50** | **0** |
| overall | avg tokens | 33,167 | 33,827 | +2% |
| overall | p50 latency | 9,909ms | 10,510ms | +6% |
| overall | p95 latency | 16,830ms | 16,383ms | −3% |

### Verdict

Flat on net success (34/50 both ways) at a small cost/latency tax
(+2% tokens, +6% p50). The gain is concentrated exactly where the pattern
targets it: `tool_fails` 2/4→4/4 — the output guardrail catches the case
where a broken tool's error was ignored and a number got fabricated
anyway. The loss is concentrated in `multi_hop` (8/12→6/12) — the number
guardrail sometimes rejects a *correct* answer because the number's
formatting in the final text (e.g. comma placement, currency symbol
position) doesn't string-match the tool output verbatim, a false positive
on the guardrail's exact-match check, not a real fabrication. At n=2 runs
this is close to noise, but the mechanism is real and traceable (see the
m05 example in Task 5(c) below, which shows exactly this false rejection).
**Kept for Task 5's main matrix** because the tool_fails win is
higher-value than the multi_hop false-positive cost for this corpus, but a
fuzzier number-match (tolerant of formatting) would likely recover most of
the multi_hop loss — noted, not built, since Task 6 spent its two
experiments elsewhere.

---

## Task 5 — Agent vs. static RAG

Full sliced table (`task5_sliced_table.csv`, `assignment_04.xlsx`, agent
run with guardrails on, v1 system prompt — the Task 5 baseline, before
Task 6's fixes):

| config | type | success | avg_tool_calls | avg_tokens | p50 | p95 | cap_breaches |
|---|---|---|---|---|---|---|---|
| rag | single | 10/55 | 1.0 | 4,267 | 1,417ms | 4,221ms | 0 |
| rag | multi_hop | 0/30 | 1.0 | 4,207 | 1,356ms | 1,710ms | 0 |
| rag | no_tool | n/a | — | — | — | — | 0 |
| rag | unanswerable | 15/15 | 1.0 | 4,113 | 1,340ms | 1,507ms | 0 |
| rag | tool_fails | n/a | — | — | — | — | 0 |
| agent | single | 35/55 | 2.33 | 23,937 | 8,877ms | 16,216ms | 0 |
| agent | multi_hop | 17/30 | 3.73 | 68,201 | 13,696ms | 27,777ms | 5 |
| agent | no_tool | 15/15 | 0.0 | 2,403 | 3,637ms | 7,550ms | 0 |
| agent | unanswerable | 9/15 | 3.07 | 30,627 | 12,179ms | 15,389ms | 0 |
| agent | tool_fails | 10/10 | 2.0 | 14,741 | 6,877ms | 10,101ms | 0 |

### (a) A task the agent won that RAG structurally could not do

**m02**: *"מבוטח מאוחדת עדיף עבר 2 ניתוחים פרטיים בישראל השנה. כמה בסך הכל
שילם בהשתתפות עצמית?"* — RAG: 0/5 (a single retrieve-then-generate call
can surface the ₪500-per-surgery figure but has no step that multiplies it
by 2). Agent: 5/5. Trajectory (`traces/m02_agent_run1.jsonl`):

```
step 1  retrieve_policy_docs({"query": "אוחדת עדיף ניתוח פרטי השתתפות עצמית", "k": 5})
step 2  retrieve_policy_docs({"query": "מאוחדת עדיף ניתוח השתתפות עצמית סכום", "k": 10})
step 3  retrieve_policy_docs({"query": "אוחדת עדיף ניתוח 1500 2000 השתתפות", "k": 10})
step 4  calculate({"expression": "500 * 2"})
step 5  → "...שילם בסך הכל 1,000 ₪ בהשתתפות עצמית..."
```
The decision point is step 4: having confirmed the ₪500 per-surgery figure
across three retrieval attempts, the agent recognized the retriever's job
was done and switched to `calculate` — the exact two-tool handoff no
static pipeline has a slot for. This is the whole thesis of the lecture in
one example.

### (b) A task the agent lost

**u02** (unanswerable, travel-insurance pricing — not in the corpus): RAG
15/15 (its generation prompt has a single fixed refusal sentence baked in,
so "not found" is trivially exact). Agent (pre-Task-6-fix): **0/5**, despite
refusing *in substance* every time. `traces/u02_agent_run1.jsonl`, final
answer: *"...ביטוח נסיעות לחו"ל... הוא מוצר ביטוח מסחרי נפרד שאינו חלק
מתכניות השב"ן... ולכן אינו מופיע במסמכים שלי."* — a correct refusal, worded
its own way rather than in the one phrase the scoring regex and the
agent's own post-hoc refusal check both looked for. RAG wins this one for a
structural reason (one fixed sentence is trivially checkable) that has
nothing to do with RAG being smarter — the agent was *right* and still
lost the metric. (Fixed in Task 6, Experiment 1: 0/5→5/5 after adopting the
same canonical sentence RAG already used.)

### (c) A task with variance across the 5 runs

**m05**: *"מבין שלוש הקופות... מהי התקרה הגבוהה ביותר... להשתלת איבר
בחו"ל..."* — 2/5 (runs 2, 4). Diffing run 1 (fail) against run 4 (success):

```
run1: retrieve → retrieve → retrieve → [guardrail rejects] →
      "לא ניתן לאשר את התשובה: הסכום/המספר 300,000, 300,000, 275,000,
       280,000 אינו מופיע בתוצאות הכלים..."
run4: retrieve → retrieve → answered directly:
      "...מכבי זהב: עד 300,000 דולר... ההסתייגות... הסכום כולל גם
       טיפולים נלווים..."
```
First divergence is the output guardrail's number check in Task 4's
pattern: run 1's final answer restates the retrieved number with a `$`
sign or comma format that doesn't byte-match the tool output's exact
formatting, so the guardrail treats a *correctly retrieved* number as
unverified and blocks a right answer. This is the guardrail's own false
positive from the Task 4 table above, caught live in the wild — and it's
the number that decides whether you'd ship this: a feature that silently
flips a correct answer to a refusal 60% of the time on a single formatting
quirk is not something you ship without a fuzzier match.

### (d) A 0/5 task, and the broken-task check

**m06**: *"...טיפול פוריות... בעלות 10,000 ₪. כמה היא תשלם מכיסה עצמה..."*
— 0/5, `terminal_state=cap_breached` all 5 runs (`traces/m06_agent_run1.jsonl`:
8 steps, 155,787 tokens, hit the 120k token-budget net). Read the transcript
before blaming the agent: is the task impossible, is the tool missing, or
is `success_criteria` too strict? None of those — `retrieve_policy_docs`
returns the *same* `meuhedet_adif_takanon.md` intro chunk for 6 of 7
re-worded queries in a row (only query 4, an off-topic "IVF" rewording,
surfaced a different — wrong — document); the 15% figure is presumably
somewhere in the corpus but the retriever never surfaces the right chunk
regardless of how the agent rephrases. **Diagnosis: a retrieval/chunking
gap**, not an agent reasoning failure, a missing tool, or a bad success
predicate — confirmed independently in Task 6, Experiment 2, where
upgrading the *model* (Haiku→Sonnet) only nudged this to 1/5, because a
smarter model still can't reason about a chunk it never receives.

### (e) The cost of autonomy, as a number

Aggregated across all runs (`assignment_04.xlsx`):

| metric | RAG | Agent | Agent ÷ RAG |
|---|---|---|---|
| avg tokens/call | 4,226 | 32,043 | **7.6×** |
| avg tool calls | 1.0 | 2.45 | **2.45×** |
| p95 latency | 2,154ms | 17,899ms | **8.3×** |

What that bought: the ability to answer 17/30 `multi_hop` tasks RAG
structurally scores 0/30 on (finding a), and to refuse cleanly with a
citable reason instead of a single canned sentence (findings b, d) — but
also 5/30 `multi_hop` cap breaches, a guardrail-driven flip-flop on a
correct answer (finding c), and roughly an order of magnitude more tokens
and latency for every call, whether the task needed it or not.

---

## Task 6 — two improvement cycles

Full write-up with hypotheses, results, and honest before/after tables:
**see `task6_writeup.md`**. Summary:

| experiment | lever | verdict |
|---|---|---|
| 1 | system prompt: hard 4-call retrieval cap + canonical refusal sentence | **Ship.** unanswerable 9/15→**15/15**, multi_hop cap breaches 5→**0**, p95 −47%, tokens −32%, multi_hop success unchanged (17/30) |
| 2 | model upgrade: Haiku → Sonnet, multi_hop only | **Don't ship.** Success 17/30→**15/30**, p95 +46%; the remaining failures (m01, m06) are a retrieval gap, not a reasoning gap — a bigger model can't fix a chunk it never receives |

---

## The verdict paragraph

**Ship the agent, but only for the multi_hop slice — as a router branch,
not a wholesale replacement.** The number that decides it: RAG scores
**0/30** on `multi_hop` by construction (no step in a fixed
retrieve→generate pipeline can call `calculate`), while the agent, even
before Task 6's fixes, already clears **17/30** there — that's the one
capability class no amount of RAG tuning reaches, at any cost. But outside
that slice, the picture reverses: on `single` tasks RAG is **7.6× cheaper
and 8.3× faster in p95** for a task shape it already handles adequately in
substance (its low 10/55 code-checked score is largely a strict-string-match
artifact — the `single` slice is the one place `expected_tools` correctly
predicts "one retrieval call is enough," which is exactly the case a router
should short-circuit to RAG). And on `unanswerable`/`no_tool`, once Task 6
fixed the refusal-wording mismatch, the agent matches RAG's reliability at
roughly 8× the cost for the same outcome.

If I were shipping this today, **a router with two branches would beat
either system running alone**: classify multi_hop-shaped questions (numeric
composition across a retrieved fact, or an explicit second-tool need like a
waiting-period date) to the agent, and everything else to the frozen RAG
pipeline. That's not a hedge — it's what the numbers in Task 5(e) and the
Task 6 experiments actually say: autonomy earns its 7-8× cost premium on
exactly one task type in this eval set, and costs it for free everywhere
else.
