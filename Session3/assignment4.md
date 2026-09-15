# Assignment 4 — Agents

**Lecture 4 · Workflows, Systems & AI Agents**

---

## What this assignment is about

In Assignment 3 you built a RAG pipeline: `query → retrieve → generate`. It runs the same three steps, in the same order, every single time. That's its virtue — and its ceiling.

Some of the questions in your Task 2 eval set could not be answered no matter how you tuned chunk size or K. Not because the model was too weak, but because answering them needs **two lookups**, or **arithmetic**, or the judgement to **skip retrieval entirely**. Your pipeline has no place to make that decision.

This assignment adds the decision. Your retriever doesn't get replaced — it gets **wrapped in a tool** and handed to something that chooses whether to call it, how many times, and in what order.

Here's the arc:

1. **Write the task set first, and freeze your Assignment 3 baseline** — including tasks the static pipeline structurally cannot do.
2. **Design your tools** — the retriever plus two more, specified well enough that a stranger could use them.
3. **Build the ReAct loop** — by hand once, then in LangGraph, with the three safety nets and full tracing.
4. **Add one pattern** — reflection, evaluator–optimizer, or guardrails, and measure whether it earned its cost.
5. **Evaluate** — agent vs. static RAG, on outcomes, cost, latency and reliability across repeated runs.
6. **Improve** — two experiments, one variable at a time.

The through-line from class: **most "agent" problems are workflow problems, and an agent that works once is not an agent that works.** Most of the grade is in whether you can tell the difference — and defend it with numbers.

> **Keep working in your course repo.** You need your Assignment 3 index and retriever, and your Assignment 2 judges. Do not start a new project.

---

## Setup

Add to your existing virtual environment:

```bash
pip install langgraph langchain-anthropic
pip install ddgs                    # web search — or tavily-python
```

Keep everything else from Assignment 3 — your saved FAISS/Chroma index, your embedding model, your judges.

Three roles this time:

| Role | Model | Why |
|---|---|---|
| **Agent** | `claude-haiku-4-5` | you'll run every task **5 times**; a slow agent model makes this assignment expensive |
| **Judge** | `claude-sonnet-5` | stronger, and not the model being judged |
| **Upgrade lever** | `claude-sonnet-5` | held in reserve as a **Task 6 experiment**, not a starting point |

> **Start on Haiku deliberately.** If it picks the wrong tool or loops, that is a **finding**, not a setup error — and the fix is almost never "use a bigger model." Slide 37: tool descriptions first, deterministic logic second, model last. Task 6 enforces that order.

**Cost warning, read it now.** An agent re-sends its entire trajectory on every step, so a 10-step task can cost more than 50 single calls. You are about to run ~25 tasks × 5 runs × 2 configurations. Before you launch the full matrix:

- Set the three safety nets from Task 3 **first**.
- Do a dry run on **3 tasks × 1 run** and read the token counts.
- Multiply by 250. If that number frightens you, shrink the task set — don't remove the nets.

---

## The use case: your RAG system, now deciding

Keep the **domain and corpus from Assignment 3**. What changes is that the system now has more than one thing it can do, and has to choose.

Your agent must have **at least 3 tools**:

1. **The retriever from Assignment 3** — mandatory.
2. **A tool that does something retrieval structurally cannot** — a calculator, a code executor, a live API (weather, currency, stock, your company's REST endpoint), a date/duration utility.
3. **One more of your choosing** — web search, a second corpus, a lookup table, a unit converter, an internal API.

> **At least one tool must be capable of something no amount of retrieval could deliver.** If all three tools are flavours of "find text," the agent's tool choice is never load-bearing and Task 5 will show a flat, uninteresting comparison. Arithmetic is the cheapest way to guarantee a real decision point: no corpus contains *"18% VAT on ₪4,390"*.

---

## The evaluation criteria

Seven things you'll measure. Note the right-hand column, and note what is deliberately **absent**.

| Criterion | What it's asking | Measured by |
|---|---|---|
| **Task success** | Is the **final state / final answer** correct? | **code** where checkable, else judge |
| **Faithfulness** | Is every claim supported by what the tools actually returned? | LLM judge |
| **Refusal correctness** | Did it refuse exactly when it should have? | **code** + judge |
| **Cost** | Tokens, $, **number of tool calls** | **code** |
| **Latency** | Wall-clock, reported as **p50 and p95** | **code** |
| **Reliability** | Same task, **5 runs** — how often does it succeed? | **code** |
| **Trajectory sanity** | Sane steps, no thrash, no loops | judge — **debugging only** |

Four things to notice:

- **There is no "did it call the right tools in the right order" metric, and there must not be.** Grade the outcome, not the path. If your agent solved a task in 2 steps where you expected 5, that's a win you would have marked as a failure. You will record the path — you will not score against it.
- **The one exception is safety.** A hard constraint like *"must never call `execute_code` on an unanswerable question"* is legitimately path-level. Constraints, not prescribed sequences.
- **Reliability is new, and it is the metric most likely to embarrass you.** Assignment 3's pipeline was near-deterministic. An agent is not. A task that passes your demo and fails 2 runs out of 5 is not a working feature.
- **Cost and latency are quality metrics now, not ops trivia.** 95% success at $4 and 90 seconds can be strictly worse than 88% at $0.20 and 8 seconds. Your final recommendation has to weigh them.

---

## Task 1 — The task set, and your frozen baseline

**Goal:** fix the target before you build, and make sure the comparison is honest.

### 1. Freeze the baseline

Your Assignment 3 RAG pipeline is the thing to beat. Pin it: same index, same K, same generation prompt, same model. Write down the commit hash. **Do not improve it while you build the agent** — a moving baseline makes every number in Task 5 meaningless.

### 2. Write 20–30 agent tasks

For each task record:

| Field | What it holds |
|---|---|
| `task` | the request, phrased the way a real user would |
| `success_criteria` | a **checkable predicate** — see below |
| `reference_answer` | the correct answer, written by you |
| `type` | `single` / `multi_hop` / `no_tool` / `unanswerable` / `tool_fails` |
| `expected_tools` | which tools you *think* it needs — **debugging metadata, never scored** |
| `answerable` | `True` / `False` |

**Required composition** — this is where the assignment lives:

- **6+ `multi_hop`** — genuinely need **two or more tool calls**, and at least 3 of them should need **two different tools** (e.g. retrieve a price, then compute a total). These are the tasks your static pipeline cannot do. Carry your Assignment 3 multi-hop questions over.
- **3 `no_tool`** — answerable with no tool at all (*"what can you help me with?"*, *"summarise what you just told me"*). **Success means the agent calls zero tools.** This is your over-retrieval detector, and it will catch more agents than you expect.
- **3 `unanswerable`** — not in the corpus, not derivable from any tool. Success = a clean refusal after a *bounded* search, not a 12-step hunt ending in a guess.
- **2 `tool_fails`** — you will deliberately break a tool for these (Task 3). Success = the agent reports it cannot complete the task. Success is **not** inventing an answer, and **not** retrying forever.
- The rest `single` — one tool call, straightforward. Your control group.

### 3. Make success checkable

`success_criteria` must be something a program or a judge can decide without argument:

- ✅ `answer contains "₪5,180.20" (±0.01)`
- ✅ `answer cites both policy_2026 and contract_2024`
- ✅ `tool_calls == 0`
- ✅ `refused == True`
- ❌ `the answer is good and helpful`

Prefer **code-checkable** criteria. This is the one place agents make evaluation *easier* than Assignment 3: numbers, identifiers, refusal booleans and tool-call counts are all free to check and impossible to argue with. Reach for the judge only when the answer is genuinely open-ended.

> **`expected_tools` is a trap you are being handed on purpose.** You are recording it because it makes failures readable — "it never called the calculator" is a two-second diagnosis. The moment you use it to *score*, you've built an eval that punishes an agent for being cleverer than you. Keep it in the trace, keep it out of the metric.

✅ **Done when:** 20–30 tasks saved as CSV/JSON with checkable success criteria, the required type mix, and a frozen baseline commit hash recorded.

---

## Task 2 — Design your tools

**Goal:** internalise that tool design beats prompt engineering, by doing it properly once.

Build `tools.py` with your three tools. For **each** one, write:

- **A precise name** — `search_product_catalog`, not `search`.
- **A description that states scope, return value and limits.** What corpus/API does it cover? What does it return? What does it explicitly **not** cover?
- **Typed arguments** with descriptions, via Pydantic or the LangChain `@tool` decorator's schema.
- **Explicit failure behaviour** — see below.

Compare:

```python
# ❌
@tool
def search(q: str) -> str:
    """Search."""

# ✅
@tool
def search_docs(query: str, doc_name: str | None = None) -> str:
    """Search the 2026 company policy corpus (8 documents, see /corpus).
    Returns up to 5 relevant passages with document name and page.
    Returns "NO_RESULTS: <query>" if nothing matches.
    Does NOT cover: pricing, live inventory, anything after Jan 2026."""
```

### The failure contract

Every tool must **return an explanatory string on failure — never raise, never return `None`, never return an empty string.**

```python
return "NO_RESULTS: no passage matched 'X'. Try a broader query or a different document."
return "ERROR: the currency API timed out. This tool is unavailable; do not retry more than once."
```

This is the fragility point from slide 30. An exception kills your run; a bare `None` or `""` gets interpreted as "nothing to see here" and the agent confidently improvises around it. An explanatory string is the only version the model can actually recover from — and it's the difference between your two `tool_fails` tasks passing and failing.

### Run the dumb engineer test — for real

Open a **fresh** Claude session with no context about your project. Paste **only** your three tool descriptions and schemas, plus one of your tasks. Ask it which tool it would call and with what arguments.

- If it picks correctly: your descriptions work.
- If it hesitates, picks wrong, or asks a clarifying question: **rewrite the description and try again.** Record both versions.

> Ten minutes here saves you an evening in Task 5. Every "the agent chose the wrong tool" bug you hit later traces back to a description you thought was obvious because you wrote the function.

✅ **Done when:** three tools with precise descriptions, typed args and the failure contract, plus a written before/after for at least one description the dumb engineer test forced you to fix.

---

## Task 3 — The ReAct loop

**Goal:** understand the loop, then let the framework run it — and make sure it can never run away.

### 1. Write it by hand first (~20 lines, ~20 minutes)

Before you `import langgraph`, implement the raw loop from class:

```python
messages = [system, user_task]
while steps < 10:
    resp = llm(messages, tools=[...])
    if resp.tool_calls:
        for call in resp.tool_calls:
            result = TOOLS[call.name](**call.args)     # your code runs it
            messages += [resp, tool_message(result)]
    else:
        return resp.text                              # the model says it's done
```

Run **one multi-hop task** through it and print every message. Commit this file as `raw_loop.py`. You will not use it again — but you'll never again think an agent framework is doing something mysterious.

### 2. Now build it in LangGraph

Build `agent.py` using LangGraph (`create_react_agent` is fine). Same three tools, same system prompt.

Your system prompt should be short and should include stopping rules:

- answer only from what the tools return
- if a tool fails twice, stop and say you cannot complete the task
- do not call a tool when you already have the answer
- refuse cleanly when the answer isn't available

### 3. Wire the three safety nets — before your first full run

| Net | Behaviour on breach |
|---|---|
| **Max iterations** (start at 10–12) | return a clean "I couldn't complete this within the step limit" |
| **Token budget** per task | same — a clean, recorded failure |
| **Wall-clock timeout** | same |

A breach must be a **recorded outcome**, not a crash and not a silent truncation. Count breaches per configuration and report them in Task 5 — an agent that hits the cap on 3 of 25 tasks is telling you something specific.

### 4. Trace everything

Every run writes JSONL, one line per step:

```json
{"task_id": "t07", "run": 2, "step": 3, "thought": "...", "tool": "search_docs",
 "input": {"query": "refund window"}, "output": "...", "duration_ms": 412,
 "input_tokens": 3180, "output_tokens": 96}
```

Plus one summary line per run: total steps, total tokens, total wall-clock, terminal state (`answered` / `refused` / `cap_breached` / `error`).

> **This is not optional and it is not for later.** Agent bugs are emergent — nothing throws, the answer is just wrong. Without the trace your only debugging tool is re-running it and hoping. And these trace files *are* your eval transcripts in Task 5, so the instrumentation pays for itself twice.

### 5. Break a tool on purpose

For your two `tool_fails` tasks, add a flag that makes one tool return its error string (`"ERROR: service unavailable"`). Run them. Does the agent report the failure, or invent an answer? Does it retry forever?

If it loops or fabricates, fix the system prompt and the tool's error string **now** — before you evaluate 25 tasks.

✅ **Done when:** `raw_loop.py` committed, the LangGraph agent runs end to end, all three nets fire cleanly and are counted, JSONL traces are written, and the two `tool_fails` tasks behave.

---

## Task 4 — Add one pattern

**Goal:** add one production pattern and find out whether it was worth it.

Pick **one** and wire it in:

| Pattern | What it does | Notes |
|---|---|---|
| **Guardrails** | validate input and output — schema, scope, refusal on out-of-domain | fastest to implement well; most production-relevant |
| **Evaluator–Optimizer** | a **separate** judge scores the draft against your Assignment 2 rubric; reject → revise, **max 2 rounds** | most satisfying — your old rubric drops straight in |
| **Reflection** | the same model critiques and revises its own answer, **1–2 rounds** | cheapest; expect a small effect, and say so honestly |

Then measure it. Re-run your task set **with** and **without** the pattern and report the delta on:

- task success (overall, and on the slice you expected it to help)
- cost — tokens and tool calls
- latency — p50 and p95

> **One pattern done properly beats three bolted on.** And the required deliverable is the **before/after table**, not the pattern itself. *"+38% latency and +22% tokens for +2 points of success, which is inside my noise band — I would not ship this"* is a **strong** answer. Adding a pattern and asserting it helped, with no numbers, is the weakest thing you can submit.

✅ **Done when:** one pattern implemented, and a before/after table on success, cost and latency with a one-paragraph verdict.

---

## Task 5 — Evaluate: Agent vs. static RAG

**Goal:** the head-to-head, run properly — then the parts it hides.

### 1. Run the matrix

Every task, on **both** configurations, **5 times each**:

- **A:** your frozen Assignment 3 RAG pipeline
- **B:** your agent (with the Task 4 pattern, if you kept it)

For RAG, the `no_tool` and `tool_fails` tasks may be structurally meaningless — record that explicitly as `n/a`, don't silently drop the rows.

### 2. Score the outcomes

- **Task success** — code-checked against `success_criteria` wherever possible; Sonnet judge otherwise, using the Assignment 2 pattern (rubric in the prompt, Pydantic schema, `explanation` **before** `verdict`).
- **Faithfulness** — judge, given the answer **and the tool outputs from the trace**. This is the agent version of Assignment 3's faithfulness: is every claim supported by what the tools actually returned?
- **Refusal correctness** — code. Count **false refusals** and **false answers** separately.
- **Cost, latency, tool calls, cap breaches** — from your traces.

### 3. The table

One table: **RAG vs. Agent**, per metric, **sliced by task type** (`single` / `multi_hop` / `no_tool` / `unanswerable` / `tool_fails`).

Report success as a rate over 5 runs (`4/5`), not a boolean. Report latency as **p50 and p95**, not the mean — agent latency has a long right tail by construction, and the mean flatters it.

### 4. Now the interesting part

Five things you are **required** to find and write up, each with the actual trace pasted in:

- **a. A task the agent won that RAG structurally could not do.** Paste the full trajectory. Show the two tool calls and the point where the agent decided it needed the second one. This is the whole thesis of the lecture in one example.
- **b. A task the agent lost.** Slower, more expensive, or wrong. The best version is an **over-retrieval** case — one of your `no_tool` tasks where the agent called the retriever anyway and dragged an irrelevant chunk into a perfectly good answer.
- **c. A task with variance across the 5 runs** — same input, different outcome. Diff the trajectories. What differed at the first divergence? This is the number that decides whether you could ship this.
- **d. A 0/5 task — and the broken-task check.** If any task failed all 5 runs, **read one transcript before you blame the agent.** Is the task impossible, is the tool missing, is your `success_criteria` string-matching something the agent phrased differently? Report which it was. (If nothing failed 0/5, say so and pick your worst task instead.)
- **e. The cost of autonomy, as a number.** Agent ÷ RAG for tokens, tool calls and p95 latency. One sentence: what did that buy you?

> **Read the trace before you theorise.** Every claim in (a)–(d) must be backed by trajectory lines pasted into your write-up. *"I think it got confused"* is not analysis; *"here is step 4, where the tool returned NO_RESULTS and it rephrased the same query three times"* is.

✅ **Done when:** the sliced RAG-vs-Agent table over 5 runs, plus (a)–(e) with traces as evidence.

---

## Task 6 — Two improvement cycles

**Goal:** the EDD loop, applied to a system whose control flow you no longer fully own.

Run **two** experiments. For each: a hypothesis written **before** the run, tied to a failure you actually saw in Task 5; **one** changed variable; every metric re-measured on the **same** task set; and an honest conclusion.

| Lever | What it tends to fix |
|---|---|
| **Tool descriptions** | wrong tool chosen; tool ignored when it was needed |
| **Consolidate or remove a tool** | selection accuracy degrading as tool count grows |
| **Replace a decision with deterministic logic** | over-retrieval — e.g. a code-level check that skips the retriever for greetings |
| **System prompt / stopping rules** | loops, over-calling, refusing when the answer was there |
| **Max iterations** | cap breaches, or an agent quitting one step early |
| **Model upgrade** (Haiku → Sonnet) | genuine reasoning failures — *after* you've ruled out the four above |
| **Skeleton swap** (plan-and-execute or ReWOO on the fan-out tasks) | latency and tokens on tasks whose lookups are independent |

**Your first experiment must not be the model swap.** It must be a tool-description change, a tool removal, or a deterministic-logic change. That's slide 37's ordering, and it's a rule for this assignment: prove you tried the cheap structural fix before you bought a bigger model.

> **Change one thing at a time.** Rewrite two descriptions *and* raise max-iterations and you've learned nothing about either.
>
> **Watch the noise — it's worse here than in Assignment 3.** With 25 tasks × 5 runs you're reading a rate, not a boolean, and the run-to-run variance you documented in 5(c) is your noise floor. If a config changes success from 78% to 81%, that is one task changing its mind. Say so.
>
> **Sweep cheaply.** Tool-call count and cap breaches come straight from the traces and need no judge. Use them to triage configurations before you spend money on a full judged run.

✅ **Done when:** two documented experiments — pre-registered hypothesis, one variable, re-measured metrics as deltas, honest conclusion — with the first one being a structural fix rather than a model swap.

---

## Task 7 — Bonus (optional)

Pick one if you want to go further:

- **ReWOO on your fan-out tasks.** Hand-roll a planner that emits the whole plan with `#E1`/`#E2` placeholders, fire the independent tool calls in parallel, then synthesise. Compare tokens and p95 latency against ReAct on the same `multi_hop` slice. Where does it break?
- **Reflexion-style memory.** On failure, have the agent write a one-line lesson to a store, and re-inject it on the retry. Measure pass@2 with and without the memory. Does it actually learn, or just apologise more fluently?
- **Human-in-the-loop interrupt.** Add a genuinely consequential tool (send an email, write a file, hit a POST endpoint) and use LangGraph's interrupt to require approval. Then answer the design question from class: *what does the human actually see* — and would they still be reading by the twentieth approval?
- **CodeAct.** Replace your calculator with a sandboxed Python executor. Measure steps-per-task against the tool-call version, and write down exactly what your sandbox does and does not prevent.
- **A trajectory judge — and whether it's worth anything.** Score trajectory quality with an LLM judge, then check whether it correlates with outcome success across your 125 runs. If it doesn't correlate, you've just demonstrated slide 41 empirically.

---

## What to submit

1. **Your code**, committed and pushed:
   - `raw_loop.py` (Task 3.1 — the hand-written loop)
   - `tools.py` (Task 2)
   - `agent.py` — the LangGraph agent with nets and tracing (Task 3)
   - your evaluation runner (Task 5) and experiment runner (Task 6)
2. **Your task set** as CSV/JSON, and your **raw JSONL traces** (or a representative sample if they're large).
3. **`assignment_04.xlsx`** — one row per **(task × config × run)**:
   - `task_id`, `task`, `type`, `answerable`, `success_criteria`
   - `config` (`rag` / `agent`), `run` (1–5)
   - `answer`, `success`, `refused`, `terminal_state`
   - `tool_calls`, `tools_used` (ordered list), `steps`
   - `faithfulness` verdict + explanation
   - `latency_ms`, `input_tokens`, `output_tokens`
4. **Three annotated traces**, in full: your best multi-hop win, your worst failure, and one `tool_fails` run. Annotate them — mark the step where it went right or wrong.
5. **A write-up** (markdown) containing:
   - **Task 2:** your three tools, and the before/after of the description the dumb engineer test made you fix
   - **Task 4:** the pattern you chose and its before/after table, with your verdict
   - **Task 5:** the sliced RAG-vs-Agent table over 5 runs, plus (a)–(e) with traces
   - **Task 6:** two experiments — hypothesis, change, deltas, conclusion
   - **The verdict paragraph** *(this is the one that carries the assignment)*: **would you ship the agent or the workflow?** Name the number that decided it. If your honest answer is "a router with two branches would have beaten both," say that — and show the evidence.

---

## The takeaway

You'll finish with a system that decides for itself when to retrieve, when to compute, and when to say it can't. That's the visible result.

The durable one is the judgement. Building a ReAct agent is a forty-line file — LangGraph will hand you one in a single call. Knowing whether you should have is the engineering, and it only comes from the numbers you generated in Task 5: what autonomy cost you in tokens and p95 latency, what it bought you on the multi-hop slice, and how often the same input produced a different outcome.

There is a real chance your agent loses on aggregate. **That is not a failed assignment** — reported honestly, with the traces to explain it, it's a better submission than a lucky win you can't account for. The teams who ship agents are the ones who can answer *"why did it do that?"* in under a minute, and the ones who know when not to.

**If you don't need an agent, don't use an agent.**
