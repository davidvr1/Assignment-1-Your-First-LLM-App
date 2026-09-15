# Assignment 5 — Multi-Agent Systems: write-up

> **The wall sentence, written before any team code existed:**
> **Context bloat.** Assignment 4's `m06` (a multi-hop copay task) blew the
> 120k token budget and hit a cap breach re-querying the same retriever
> inside one ever-growing single-agent context (8 steps, 155,787 tokens,
> `task6_writeup.md`). More generally, Assignment 4's own verdict paragraph
> recommended routing multi-hop work to the agent and everything else to a
> cheaper static pipeline — i.e. *"one agent with better routing beats a
> monolith,"* which is exactly the split this assignment asks whether a
> team can deliver on honestly.

Frozen Assignment 4 baseline: commit `515034d` (agent.py/tools.py/
agent_tasks.json, first committed this session — Assignment 4 was built
but never committed until Task 1 required a pinned hash). System prompt
used for the frozen baseline in Task 6's matrix is `SYSTEM_PROMPT_V2` with
guardrails on — the *kept* Task-6 improvement, per the assignment's "freeze
the final version" instruction.

---

## Task 2 — the team design

Three agents, split by **tool domain**, directly against the context-bloat
wall (not job titles):

| Agent | Scope (no "and") | Inputs | Outputs | Tools | Model |
|---|---|---|---|---|---|
| **researcher** | Searches the שב"ן corpus and returns cited passages | a search question | cited facts, or a reported failure | `retrieve_policy_docs` (1) | haiku-4-5 |
| **analyst** | Computes a number or date from facts it's given | facts + the arithmetic/date question | a computed value | `calculate`, `date_duration` (2) | haiku-4-5 |
| **writer** | Synthesizes the final answer under the handoff's constraints | facts + constraints | final answer text | none (0) | haiku-4-5 (sonnet-5 held in reserve, Task 7 lever) |

**Topology: orchestrator-worker.** Two sentences of justification: it's the
required pattern for a three-agent project (hierarchical is explicitly out
of scope at this scale, swarm/network don't have an obvious single-hub
readable trace), and it maps directly onto the wall — a single hub keeps
each worker's context small and bounded by its own payload, rather than
letting a growing shared history re-accumulate the exact bloat the split
exists to avoid. Diagram: `team_diagram.md`.

## Task 3 — the contract layer

**Decision: hybrid** (written in `contracts.py`'s module docstring). A
small `TeamState` travels through the graph (`task_id`, `last_active`,
`route`, `handoff_log`, turn/token/time counters, `facts`, `constraints`),
but the *content* handed from one agent to the next is never the raw
conversation — it's always an explicit `Handoff` the orchestrator builds.
Pure shared state would re-bloat every context (the wall); pure message
passing would force every worker to forward everything "just in case,"
since `facts` need to survive across an arbitrary number of hops.

Typed handoff schema (`contracts.py`):
```python
class HandoffPayload(BaseModel):
    summary: str
    constraints: list[str]
    facts: dict[str, str]
    open_question: str

class Handoff(BaseModel):
    destination: Literal["researcher","analyst","writer","direct_answer"]
    payload: Optional[HandoffPayload]
    reason: str
```
**Ownership:** `TeamState.last_active` is set on every transition (never
left unset once a run starts); workers never call each other and always
return to the orchestrator, except the writer, whose output always
terminates the run — it's the synthesis step, so bouncing back would only
recreate the deadlock the assignment warns about.

**A real engineering fight worth recording:** the orchestrator's tool
schema originally nested `HandoffPayload` inside `Handoff`. Anthropic's
tool calling occasionally double-JSON-encodes a nested object as a
*string* rather than a native object, and when that string echoes source
text containing a literal embedded quote (both `unanswerable` tasks
contain the Hebrew abbreviation `חו"ל`), the nested JSON's own escaping
breaks — the payload string isn't even valid JSON. Root-caused via
`result["parsing_error"]` after several rounds of retries didn't help (see
git history, commit `d9c5f41`). Fix: the tool schema shown to the model is
now **flat** (`_HandoffFlat` in `team.py`) — no nested object for the model
to self-serialize — and reassembled into the nested `Handoff` contract in
Python. 120/120 rows completed cleanly after this fix.

## Task 4 — build

`team.py`: orchestrator dispatches via one structured-output call per turn
(one retry on a parse miss); workers are `create_react_agent` instances
with their own scope-contract system prompt and ≤2 tools, no ability to
call another worker. Four safety nets, all recorded outcomes with reason
codes, never crashes: `MAX_AGENT_TURNS=8`, `TOKEN_BUDGET=150_000`,
`WALLCLOCK_TIMEOUT_S=120`, and loop detection (same `(from,to)` handoff
pair twice in a row → `loop_detected`). Tracing is JSONL, one line per
tool call / handoff event plus a summary line per run (`route`,
`per_agent_turns`, `per_agent_tokens`, `terminal_state`).

## Task 5 — procedural memory, measured

`AGENTS.md` loaded into every agent's system prompt (domain/corpus
boundaries, house rules, output conventions, "what went wrong last time").
With/without measurement, 20 tasks × 1 run, team config, code-checkable
scoring only (no judge, to isolate the effect and keep this ablation
cheap — `task5_memory_ablation.py` / `.csv`):

| | success (code-checked) | avg tokens |
|---|---|---|
| memory **on** | 35% (7/20) | 34,280 |
| memory **off** | 45% (9/20) | 24,067 |

**This is not the result I expected, and it's a real finding, not a null
one.** At n=1 run per arm it's inside the noise floor for a 20-task set (a
couple of tasks flipping easily explains a 2-task/10-point swing), but the
direction is consistent with a mechanism visible in the traces: `c06`
under memory-on hit `loop_detected` at 134,608 tokens (the duplicated-work
failure mode below), while the same task under memory-off completed in
48,317 tokens without looping. `AGENTS.md`'s house rule about bounded
retrieval ("2-4 calls, stop and report") is advisory text competing with
the orchestrator's own retry logic (rule 7) for the same decision, and in
at least this one case the redundant instruction correlated with *more*
looping, not less, because the worker second-guessed when to stop instead
of following the orchestrator's explicit one-retry cap. **Kept anyway**:
the house rules about refusal-sentence wording and never restating a
number a tool didn't produce are load-bearing elsewhere in the matrix (no
fabricated numbers anywhere in 120 rows), and a single n=1 run isn't
grounds to cut a memory file whose other rules are doing real work — but
the `c06`-style overlap between `AGENTS.md`'s retrieval guidance and the
orchestrator's own rule 7 is a duplicate-authority problem worth trimming
next iteration.

---

## Task 6 — evaluate: one agent vs. the team

Full matrix: 20 tasks × 2 configs × 3 runs = 120 rows, **0 errors, 0 cap
breaches** in the final run (`assignment_05.xlsx`, `traces5_single/`,
`traces5_team/`). Reduced from the spec's 30-45 tasks / 5 runs by explicit
agreement (real API cost) — every required task type and its floor count
is still present (8 `cross_domain`, 4 `misroute_bait`, 3 `no_tool`, 2
`handoff_stress`, 2 `unanswerable`, 1 control).

### Table 1 — head to head, sliced by type (3 runs)

| type | config | success | agent turns (avg) | tool calls (avg) | p50 latency | p95 latency |
|---|---|---|---|---|---|---|
| cross_domain | single | 13/24 (54%) | 1.00 | 3.50 | 13,114ms | 17,540ms |
| cross_domain | team | 13/24 (54%) | 1.54 | 3.67 | 22,038ms | 27,501ms |
| handoff_stress | single | 0/6 (0%) | 1.00 | 2.00 | 5,221ms | 7,369ms |
| handoff_stress | team | 0/6 (0%) | 1.50 | 2.00 | 13,557ms | 17,454ms |
| misroute_bait | single | 3/12 (25%) | 0.75 | 1.83 | 8,944ms | 13,440ms |
| misroute_bait | team | 3/12 (25%) | 1.00 | 2.42 | 19,679ms | 23,385ms |
| no_tool | single | 9/9 (100%) | 0.00 | 0.00 | 7,339ms | 7,797ms |
| no_tool | team | 9/9 (100%) | 0.00 | 0.00 | 6,812ms | 7,198ms |
| single (control) | single | 0/3 (0%) | 1.00 | 4.00 | 13,473ms | 13,498ms |
| single (control) | team | 3/3 (100%) | 1.00 | 2.00 | 13,337ms | 13,681ms |
| unanswerable | single | 6/6 (100%) | 1.00 | 2.00 | 8,433ms | 10,956ms |
| unanswerable | team | 6/6 (100%) | 0.50 | 1.50 | 10,798ms | 21,568ms |

Cap breaches: **0** everywhere. Loops: **0** in this matrix (loop detection
fired once, in the Task 5 memory-ablation run, not the main matrix — see
Task 7 below).

**A scoring bug worth documenting, not hiding**: `handoff_stress` success
was initially computed asymmetrically — the team config required *both* a
format check (language/word-count, code) *and* a judge verdict on
substance, but the single-agent path only checked format. A clean refusal
("לא מצאתי...") is short and grammatically Hebrew, so it was passing the
format check and being counted as a *success* on `h01` even though it
never actually summarized the cancellation policy. Fixed to require both
symmetrically (`eval_runner_team.py`, `git log` for the fix) and re-judged
the 6 already-collected single-agent `h01`/`h02` answers rather than
re-running the agent. Corrected table above: `handoff_stress` is **0/6 for
both configs** — a genuinely hard slice (the cancellation-policy content
isn't well-surfaced by the retriever for either config), not a
team-specific weakness. Left uncorrected, the original number would have
made the team look worse than it is on exactly the slice this assignment
cares most about.

### Table 2 — per-agent attribution (team config)

| agent | turns taken | per-agent success rate (judge) | failures attributed |
|---|---|---|---|
| researcher | 45 | 80% | b02, b03, c03, c04, c06 |
| analyst | 16 | 69% | b04, c07 |
| writer | 3 | 100% | (none) |

This is the table that names what to fix first: **researcher**, not
because its rate is lowest (analyst's is), but because it takes 3× the
turns and its failures cluster on the corpus's own weak spots (c03/c04/c06
all fail to surface a specific copay figure that A4's single agent also
struggled with on some of the same facts — see (e) below). The analyst's
two failures (b04, c07) are cases where the judge scored the *computation*
against a stricter reading of the task than the code checker used (both
still pass the code-checkable success metric) — worth a second look before
trusting the judge's per-agent number over the code one on those two.

### (a) A task the team won that the soloist couldn't

**c07**: *"מבוטח הצטרף לתכנית מאוחדת עדיף ב-2026-01-01. מתי מסתיימת תקופת
האכשרה של 9 חודשים...?"* — single: **0/3**. team: **3/3**.
`traces5_team/c07_team_run1.jsonl`:
```
HANDOFF orchestrator -> researcher | צריך לחפש את תנאי האכשרה לניתוח פרטי - מידע זה חייב להיות מצוטט מהמסמכים
  tool: retrieve_policy_docs -> ...תקופת אכשרה כללית לרוב השירותים: 3 חודשים... (חריגים מפורטים בכל פרק)
HANDOFF researcher -> analyst | יש עובדות שנאספו (תאריך התחלה: 2026-01-01, תקופת אכשרה: 9 חודשים), דורש חישוב תאריך סיום
  tool: date_duration -> 2026-09-28
HANDOFF analyst -> direct_answer | כל העובדות הנדרשות כבר נאספו וחושבו
SUMMARY answered: "...מסתיימת ב-2026-09-28."
```
The decision point is the second handoff: the analyst used the **9
months** the researcher extracted (not a number the analyst looked up
itself) to call `date_duration` — the exact cross-agent handoff this
assignment is about. The single agent, on the *same underlying fact*,
failed all 3 runs — its v2 system prompt's hard 4-call retrieval cap
combined with a slightly different query phrasing missed the fact inside
its own single growing context (see (e) below for the general pattern:
this corpus's retrieval quality is itself brittle to query wording,
independent of the agent architecture).

### (b) A task the team lost

**b01** *("למה החשבונית שלי כל כך מבולגנת...")* before its Task-7 fix:
team **0/3**, `route=[]` — the orchestrator answered *directly*, paying a
full dispatch-classification call and getting the task wrong, instead of
consulting the researcher once. See Task 7 for the full diagnosis and fix.
Framed as the assignment's suggested pattern ("two calls to say hello"):
this cost one orchestrator call for nothing, on a task the single agent
solved (with its own single retrieval call) 2/4 of the time — see
`misroute_bait` in Table 1, where the single agent is directionally ahead
of a pre-fix team on exactly this task type (both land at 25% aggregate
because the *other* three misroute_bait tasks are a real retrieval-gap
problem shared by both configs, not a routing one).

### (c) A misroute

**b01** (the same task, `misroute_bait`, `capable_agents=["researcher"]`).
`routing_correct=0.0` for all 3 pre-fix runs — the orchestrator classified
the request as `direct_answer` ("שיחת חולין"). Diagnosis: **it was the
router's own prompt, not a worker's description** — rule 2 of
`ORCHESTRATOR_SYSTEM` read *"if the request doesn't need a worker
(small-talk, a question about your capabilities, thanks), answer
directly"*, and a vaguely-phrased complaint ("it's confusing, what do I
do") pattern-matched "small talk" closely enough to skip dispatch
entirely, even though `capable_agents` says a researcher lookup was the
right move. Fix and re-measurement in Task 7.

### (d) A lossy handoff — and a checker limitation caught red-handed

**h02** (*"...one English sentence"*): payload correctly carried
`constraints=["language=en","sentences<=1"]` to the writer
(`traces5_team/h02_team_run1.jsonl`, `payload_keys` includes
`constraints` on the `researcher -> writer` handoff), and the writer's
output — *"The copay for private analysis (ניתוח פרטי) in Meuhedet Adif is
500 ₪ per procedure."* — **is** one English sentence. My own
`HANDOFF_CHECKS['h02']` code check flagged it as failing anyway, because it
tests `not _is_hebrew(answer)` against the *whole* string, and the answer
embeds an untranslated Hebrew gloss `(ניתוח פרטי)` for the technical term —
reasonable writing, not a lossy handoff. **The honest finding here isn't a
lossy handoff at all**: the schema worked, the writer honored the
constraint, and the bug was in my own success-checking regex being
stricter than the actual requirement. Reported as a checker limitation
rather than forced into a false positive for the sake of having a clean
(d) finding.

### (e) Variance across the 5 (reduced: 3) runs

**c04** (*mirrors A4's m02: 2 private surgeries at Meuhedet Adif, 500 ₪
each*): single config — run1 ✅, run2 ✅, run3 ❌ (`refused`, citing the
same "couldn't find the specific copay figure" pattern seen on c01/c03/c06
above). First divergence: run 3's researcher-equivalent retrieval call
inside the single agent's own loop simply didn't surface the passage the
first two runs did, despite `temperature=0` — not fully deterministic tool
selection/retrieval under real load. Team, for the same task, refused
**0/3** (`traces5_team/c04_team_run*.jsonl`) — worse than single's 2/3, and
the two configs' failures on this task are **the same underlying retrieval
brittleness**, not an artifact of either architecture. This is the number
that would decide whether to ship: on a task the corpus itself makes
borderline, neither config is reliable enough at n=3 to trust blind, and
the team doesn't make it worse in kind, only in degree (team's researcher
gets fewer effective retrieval attempts per dispatch than the single
agent's own 4-call budget, per Table 2's turn counts).

### (f) The cost of coordination, as a number

| | team ÷ single |
|---|---|
| tokens | **1.12×** |
| agent turns | **1.33×** |
| p95 latency | **1.84×** |

What that bought: a fix for one A4-visible wall (cross_domain success:
54% for both, but the team never risks a single-context budget blowout —
0 cap breaches on cross_domain vs. A4's 5/30 on the equivalent
`multi_hop` slice) and a correctly-attributed `single`-control win (team
3/3 vs. single 0/3, on the same v2-prompt agent that scored this fact
correctly in A4's own writeup — a reproducibility gap in the single
agent's current behavior, not a team advantage per se). It cost roughly
12% more tokens, a third more agent turns, and **p95 latency nearly
doubled** — exactly the "latency is a sum, not a max" warning the
assignment gives, confirmed in this data.

---

## Task 7 — two failure modes, hunted and fixed

### 1. Duplicated work (`c06`)

**Reproduction**: observed live in an early partial run (before the fix
below was already in place for the full matrix — see commit history) — the
orchestrator dispatched `researcher` a second time after a failed 3-call
search, with near-identical query terms:
```
seq1-3: retrieve_policy_docs("השתתפות עצמית ניתוח פרטי בישראל מכבי זהב") / ("ניתוח פרטי השתתפות עצמית") / ("מכבי זהב ניתוח פרטי בישראל כיסוי")
HANDOFF researcher -> researcher | "זה הניסיון ה-5..." (a genuine re-try, but:)
seq4-6: retrieve_policy_docs("השתתפות עצמית ניתוח פרטי בישראל מכבי זהב") / ("ניתוח פרטי השתתפות עצמית") / ("מכבי זהב ניתוח פרטי בישראל")
HANDOFF researcher -> direct_answer | refused, having gained zero new information
```
Six tool calls, two of them nearly byte-identical to earlier ones, for one
refusal. Also observed independently in the Task 5 memory-ablation run at
134,608 tokens with `terminal_state=loop_detected` (the same-agent retry
eventually escalated into a same-pair-twice loop-net breach).

**Diagnosis**: the orchestrator's re-dispatch reasoning ("try a more
focused search") wasn't backed by any requirement that the *retry itself*
actually differ from the first attempt — a prompt/logic gap, not a model
capability gap.

**Mitigation** (prompt/deterministic-logic fix, **not** a model swap):
`ORCHESTRATOR_SYSTEM` rule 7 — a retry dispatch to the same agent is
allowed once, and only if `open_question` names an explicitly different
search term; otherwise go straight to the rule-6 refusal.

**Re-measurement**: 6 post-fix `c06` team runs (3 from the clean full
matrix + 3 fresh re-runs today) — **0/6 show a duplicate dispatch**, token
cost for the (still-failing) attempt dropped to a consistent ~50-52k
(vs. 134k+ pre-fix with the loop). **Task success is unchanged at 0/6** —
this is the honest "the fix made the waste go away, not the underlying
retrieval gap" case the assignment specifically asks to report rather than
oversell.

### 2. Hallucinated handoff (`b01`)

**Reproduction**: 3/3 (100%) before the fix — `route=[]`,
`routing_correct=0.0` every run.

**Diagnosis**: `traces5_team/b01_team_run1.jsonl`:
```
HANDOFF orchestrator -> direct_answer | "זו שאלה כללית על בעיה טכנית/אדמיניסטרטיבית... חלק מ'שיחת חולין'"
SUMMARY answered: "אני מסייע בשאלות על תוכן הפוליסות... פנה לחברת הביטוח שלך..."
```
Rule 2 of the orchestrator's own prompt ("if it's small talk, answer
directly") pattern-matched a vaguely-phrased in-scope complaint as
out-of-scope chit-chat, hallucinating that no agent could serve a request
`capable_agents` says the researcher could. The router's own wording was
the cause, not the researcher's tool description (which was never
consulted).

**Mitigation** (prompt/deterministic-logic fix, not a model swap): rule 2
narrowed to explicitly exclude vaguely-worded but substantively in-domain
complaints from the direct-answer fast path — "confusing/unclear" language
about a process, document, payment, or right no longer counts as small
talk just because it doesn't name "policy" or "takanon" explicitly.

**Re-measurement**: 3 post-fix runs — `routing_correct` **0/3 → 3/3**
(every run now dispatches to researcher first). Task success **0/3 →
2/3** — an honest partial win: the targeted bug (never even asking the
capable agent) is fully fixed, but one of the three post-fix runs still
lands on a substantively-wrong final answer per the judge, which is a
downstream researcher/synthesis question, not the hallucinated-handoff bug
this fix targeted.

---

## The verdict paragraph

**Ship the soloist, not the team — for this task set, as measured.** The
number that decides it: on the two slices where the assignment predicted
the team should win outright (`cross_domain`, where a second agent is
supposed to use the first's output, and the `single` control, which
should be a wash), the team ties on `cross_domain` (54% vs. 54%) and only
wins the `single` control because of a reproducibility gap in the current
single-agent prompt (0/3, not a structural single-agent limitation — A4's
own writeup scored this exact fact correctly). Nowhere in Table 1 does the
team **clearly and reproducibly** outperform the soloist on task success,
while it costs **12% more tokens, a third more agent turns, and nearly
double the p95 latency** (finding f) for that flat result. The two Task-7
fixes are real and worth keeping (routing accuracy 0%→100% on `b01`;
duplicated-work waste eliminated on `c06`), but they fix *coordination
overhead the team itself introduced* — they don't produce a case where the
team clears a bar the single agent couldn't already clear on its own, the
way Assignment 4's agent cleared a bar RAG structurally couldn't (0/30 on
`multi_hop` vs. the agent's 17/30). Finding (a)'s `c07` win looks like a
counterexample, but it's explained by the single agent's own retrieval
variance (finding e) on a fact both architectures know how to use, not by
a capability the team has and the soloist doesn't.

If I were shipping today: **one agent, with the v2 system prompt's
retrieval-cap fix stabilized further** (the `c01/c03/c04/c06` failure
cluster suggests the 4-call retrieval budget is sometimes too tight for
this corpus's chunking, independent of architecture) is the better bet
than either team as currently built. The team's infrastructure — typed
handoffs, four safety nets, per-agent attribution — is sound and reusable
(0 errors, 0 cap breaches, 0 unrecovered loops across 120 clean rows), and
would earn its keep the moment a task genuinely needs two *disjoint* tool
domains in one turn where a single 5-tool-cap agent would hit the tool-
overload wall Assignment 4 never actually reached. That wall wasn't hit in
this task set — most of these tasks are answerable by researcher alone or
researcher+analyst, which a well-scoped single agent already does. **If
you don't need a second agent, don't add one** — and honestly, for this
corpus and this task set, I didn't.
