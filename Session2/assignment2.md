# Assignment 2 — Evaluation

**Lecture 2 · Evaluation-Driven Development**

---

## What this assignment is about

In Assignment 1 you made an LLM produce text. The obvious next question is the one nobody can answer by looking at a single output: **is it any good?**

That question is genuinely hard, because the task you'll work on has **no single correct answer**. Two product descriptions can be completely different and both be excellent. There's no string to compare against, so exact-match, BLEU, and friends won't save you. You have to *define* what "good" means before you can measure it.

That's what you'll build here, end to end:

1. **Define a rubric** — turn a vague notion of quality into explicit, repeatable rules.
2. **Generate** — produce a description for every product with a deliberately small model, and record latency and token counts alongside each output.
3. **Evaluate by hand** — score a sample yourself using your rubric. This is your **baseline** and your **ground truth**.
4. **Improve** — change one thing at a time, re-measure, and see whether the number actually moved.
5. **Build a judge** — get an LLM to apply *your* rubric automatically.
6. **Validate the judge** — check the judge against your own human scores, and decide which you'd trust in production.

The through-line is the EDD loop from class: **write the eval first, change one thing, re-run the eval, inspect the delta, repeat.** The rubric is the eval. Everything else is the loop.

> **Keep working in your course repo.** Same repo as Assignment 1. Commit as you go and push at the end.

---

## The business use case: generating product descriptions

You're given a dataset of e-commerce products. Each product has:

- a **name**
- a set of **structured attributes** (material, colour, size, features, and so on)

Your job is to make a language model turn those raw fields into a **persuasive, 50–90 word product description** — the kind of copy a human marketer would write for a product page.

> **Prefer your own domain?** You may swap in your own dataset or use case instead — ideally the theme you picked in Assignment 1. A good alternative shape is a **grounded Q&A bot**: a list of user questions plus a knowledge base, where the model must answer *from your data*, in your product's voice (e.g. warm, helpful, never inventing facts). Everything below still applies — you'll just adjust the criteria to fit (e.g. "Length: 50–90 words" might become "answers in 1–3 sentences").
>
> If you go this route, say so at the top of your submission and describe your dataset in two or three sentences.
> (You can also have claude generate a dataset for you)

---

## The evaluation criteria

These are the six dimensions you'll score every output on. You define what each rating *means* in Task 1.

| Criterion | What it's asking | Ratings |
|---|---|---|
| **Fluency** | Do the sentences read naturally, like a human wrote them? | good / ok / bad |
| **Grammar** | Correct spelling, punctuation, agreement? | good / ok / bad |
| **Tone** | Does it match a friendly, credible sales voice? | good / ok / bad |
| **Length** | Is it within 50–90 words? | good / ok / bad |
| **Grounding** | Does it stick to the supplied attributes, inventing nothing? | good / ok / bad |
| **Latency** | Time per call (time to first byte / full response) | good / ok / bad |

Two things to notice about this table:

- **Grounding is the safety criterion.** A beautifully written description that claims the shirt is waterproof when the attributes never said so is a *worse* failure than a clumsy but truthful one. Fluency problems embarrass you; grounding problems get you sued.
- **Latency is measured, not judged.** You get it for free from a timer in Task 2. It never goes to the judge model in Task 5 — an LLM cannot know how long its own call took.

---

## Task 1 — Define your rubric

**Goal:** turn "good description" into rules precise enough that a stranger applying them reaches the same verdicts you would.

Do this **before** you generate or evaluate anything. If you write the rubric after seeing the outputs, you'll unconsciously write it to fit what the model happened to produce — and you'll have no honest baseline.

### 1a. Define each rating band

For **each** of the six criteria, write explicit definitions of *good*, *ok*, and *bad*. Push as much subjectivity out as you can: prefer countable, checkable conditions over adjectives.

Worked example for **Length**:

| Rating | Definition |
|---|---|
| good | 50–90 words |
| ok | 40–49 or 91–110 words |
| bad | fewer than 40 or more than 110 words |

Now do the same for the soft ones. "Fluency: good = reads well" is not a rubric — it's a feeling. Something like *"good = no awkward phrasing; ok = 1–2 clunky phrases but the meaning is clear; bad = 3+ awkward phrases, or a sentence you have to re-read to parse"* is a rubric: it's countable, and two people will mostly agree.

**Grounding deserves the most care.** Decide up front how you treat generic sales filler ("perfect for everyday use") that isn't in the attributes but isn't a factual claim either — is that acceptable, or is it ungrounded? There's no universally right answer; there's only *your* answer, written down and applied consistently.

For **Latency**, pick thresholds in milliseconds (e.g. good ≤ 2000 ms, ok 2001–5000 ms, bad > 5000 ms). Run a couple of calls first to see the rough scale on your machine.

### 1b. Define pass / fail

Six per-criterion ratings don't ship a product. You need one bit: does this description go on the site or not?

- **Cumulative pass bar** — the minimum combination of ratings needed to pass.
  *Example: "at least three `good` ratings and zero `bad` ratings."*
- **Go / no-go rules** — any single criterion that fails the whole row on its own, no matter how good everything else is.
  *Example: "if Grounding is not `good`, the description is rejected."*

Write both rules explicitly. You'll apply the exact same rules in Task 3 (by hand), Task 4 (after improving), and Task 6 (via the judge) — that's what makes the three comparable.

✅ **Done when:** you have a rubric document with 18 rating definitions (6 criteria × 3 bands) plus your pass bar and go/no-go rules, and you'd be comfortable handing it to a classmate and expecting them to score the same way you do.

---

## Task 2 — Generate a description for every product

**Goal:** produce the raw material you'll evaluate, and instrument the generation so you capture more than just the text.

### The prompt

Write a **system prompt** that instructs the model to produce a persuasive 50–90 word description from the product name and attributes. Apply the prompting guidelines from class — a clear role, explicit constraints, and a stated output format.

> **Your rubric is the spec for your prompt.** If your rubric says grounding means "no claims beyond the attributes," the prompt should say so. If it says 50–90 words, the prompt should say so. Writing the rubric first pays off immediately here.

### The model

Start with a deliberately **small** model from Hugging Face:

```
Qwen/Qwen2.5-0.5B-Instruct
```

Yes, it's tiny. That's the point — see the note at the end of this task.

### Structured output

For **each** call, collect these fields into a dictionary:

| Field | What it holds |
|---|---|
| `generated_description` | the model's output text |
| `latency_ms` | end-to-end generation time, in milliseconds |
| `input_tokens` | tokens sent to the model, **including the system prompt** |
| `output_tokens` | tokens returned by the model |

Latency and token counts are what let you talk about **cost and speed**, not just quality. A prompt change that lifts your pass rate by 5% while tripling your token count is a trade-off, not a win — and you can only see that if you measured it.

### Storage

Collect everything into a **DataFrame**, then add:

- one **blank column per rubric criterion** (Fluency, Grammar, Tone, Length, Grounding, Latency)
- a blank **`final_score`** column (pass / fail)

Save it as an Excel file named **`assignment_02.xlsx`**. Those blank columns are where your human scores go in Task 3 and where the judge's verdicts go in Task 6.

> **Expect bad output — that's the design.** A 0.5B model will ramble, miss the word count, and occasionally invent features. That's *useful*: it gives you real failures to score, real signal in your baseline, and real room to improve in Task 4. **You lose no points for weak descriptions** as long as the pipeline, the measurements, and the spreadsheet are correct.

✅ **Done when:** `assignment_02.xlsx` exists, has one row per product with the four generated fields populated, and has empty criterion + `final_score` columns ready to fill.

---

## Task 3 — Manual (human) evaluation

**Goal:** establish your baseline, and produce the ground truth you'll later measure the judge against.

Open `assignment_02.xlsx` and:

1. **Rate each criterion.** Pick **10–15 products** and score every criterion `good` / `ok` / `bad` using the Task 1 rubric. Apply the rubric literally, even when it hurts — if a description you like scores `bad` on Length, write `bad`. The moment you start fudging, your ground truth stops being ground truth.
2. **Set the final score.** Apply your cumulative pass bar and go/no-go rules to fill `final_score` (pass / fail) for each of those rows.
3. **Analyse the baseline.** Look across your scores and answer:
   - Which criteria did the model handle **best**?
   - Which did it handle **worst**?
   - Are the failures clustered — same failure mode over and over — or scattered?

That third step is the whole point. It's **error analysis**: it tells you what to fix in Task 4 instead of guessing. If 12 of 15 descriptions blow the word count but grounding is fine, your next move is a length constraint, not a bigger model.

> Doing this by hand is slow and slightly tedious — notice exactly *how* slow. That feeling is the argument for Task 5, and you should be able to quantify it later (roughly how many seconds per row × thousands of products per day).

✅ **Done when:** 10–15 fully scored rows with `final_score` set, plus a short written baseline analysis naming your best and worst criteria.

---

## Task 4 — Improvement cycle

**Goal:** run the EDD loop for real — change one thing, re-measure with the same rubric, and see whether the number moved.

You now have a baseline. Try to beat it.

### Levers to pull (you don't need to try all of them)

- **Prompt engineering** — rewrite the system prompt, add or change few-shot examples, tighten the constraints. *Cheapest lever, try it first.*
- **Model choice** — move to a stronger model such as **Claude Haiku**. We all know this will improve results, which is exactly why you should **exhaust prompt engineering first** — otherwise you learn nothing except "bigger model is better."
- **Decoding parameters** — adjust `temperature`, `top_p`, `top_k`, `max_new_tokens` to trade creativity against factuality. (Lowering temperature is a common fix for grounding failures.)
- **Post-processing** — run a grammar check or trim to length after generation. Deterministic code can fix deterministic problems more reliably than more prompting.

### Document every experiment

For each thing you try, record:

1. **What you changed** — one change, stated precisely.
2. **Why you expected it to help** — tie it to a failure mode you actually observed in Task 3.
3. **The new scores** — re-evaluate with the **same Task 1 rubric** on the **same products**, and report the delta from baseline.

> **Change one thing at a time.** If you swap the model *and* rewrite the prompt *and* drop the temperature, and the score improves, you've learned nothing about which change did it.
>
> **Beware of noise.** LLM output is stochastic. A jump from 8/15 to 9/15 passing is one row — it could easily be randomness rather than progress. Say so honestly in your write-up rather than claiming a win you can't defend. (This is the "don't ship noise as progress" point from class.)

✅ **Done when:** you have at least two documented experiments, each with the change, the hypothesis, and re-scored results against the baseline.

---

## Task 5 — Build a judge model

**Goal:** automate the scoring you did by hand in Task 3, by having an LLM apply your rubric.

Manual evaluation is thorough but slow and impossible to scale — you felt that in Task 3. **LLM-as-a-judge** trades a little accuracy for enormous scale. Now you build one.

### The model

Use a model you did **not** use in Task 2 — start with **Sonnet**. If it struggles to apply your rubric consistently, step up to **Opus** or **Fable**.

> Using a different model as the judge isn't arbitrary. It reduces **self-enhancement bias** — the tendency of an LLM judge to favour outputs from its own family. Judging your own homework is not evaluation.

### The judge prompt

Write a prompt that contains **your Task 1 rubric definitions verbatim**, so the judge applies exactly the standards you applied by hand. If the judge is working from a different (or vaguer) standard than you were, comparing its verdicts to yours in Task 6 is meaningless.

Two specifics:

- **Exclude Latency (and cost).** Those are measured programmatically. The judge scores the four text criteria: Fluency, Grammar, Tone, Length. *(Length is arguably countable in code too — if you'd rather compute it and hand the judge a word count, say so and justify it.)*
- **Think hard about what context the judge needs — especially for Grounding.** To judge fluency, the description alone is enough. To judge grounding, the judge must see **the source attributes**, or it has no way to know whether "water-resistant" was in the data or invented. Grounding is a comparison, so the judge needs both sides.

### The output schema

For **each** criterion the judge returns:

- `explanation` (string) — its reasoning for the verdict
- `verdict` (enum: `good` / `ok` / `bad`)

**`explanation` must come before `verdict` in the schema.** In your submission, explain why that ordering matters.

> *Hint:* think about how an autoregressive model generates its response — token by token, in order, each token conditioned on everything already emitted. What can the verdict token depend on in each ordering?

Enforce the structure with a **Pydantic** schema via the API's structured-output support, so a malformed response raises an error instead of silently flowing into your spreadsheet as garbage. (You met Pydantic in Assignment 1b — same idea, bigger payoff: here it's guarding hundreds of rows, not one.)

✅ **Done when:** you can call the judge on one description and get back a validated Pydantic object with an explanation and a verdict for each criterion.

---

## Task 6 — Run and analyse the judge

**Goal:** find out whether your judge can be trusted, and turn that into a recommendation.

### 1. Sanity check

Run the judge on **5 products** and read the output carefully by hand. Do the explanations actually make sense? Is it applying *your* rubric, or its own instincts? Common failures: rating everything `good`, ignoring the word-count band, or "grounding" against its own world knowledge instead of the supplied attributes.

Fix the prompt now, before you spend money on the full run.

### 2. Full run

Run the judge on **all** products. Compute `final_score` for each using **the same pass bar and go/no-go rules from Task 1**. Write everything — per-criterion verdicts, explanations, and final score — back into the spreadsheet, in columns clearly separated from your human scores.

### 3. Compare to human evaluation

For the 10–15 products you scored by hand in Task 3, compute an **agreement rate per criterion**:

```
agreement(criterion) = (# rows where judge verdict == your verdict) / (# rows scored by hand)
```

Then dig into it:

- **Where do you agree?** (Expect the countable criteria — Length, Grammar — to agree most.)
- **Where do you diverge?** (Expect the subjective ones — Tone, and grounding edge cases — to diverge most.)
- **Why?** Read the judge's explanations on the rows where you disagreed. Sometimes the judge is wrong. Sometimes the judge is right and *you* were inconsistent. And very often the real culprit is a **third thing: your rubric was ambiguous**, and you and the judge each resolved the ambiguity differently. That third case is the most valuable finding in this whole assignment — it means the fix is to tighten the rubric, not to blame either evaluator.

> Bonus, if you want it: don't just count agreements. Look at the *direction* of disagreement — is your judge systematically more generous than you are? A judge with 70% agreement that's unbiased is a very different tool from one with 70% agreement that always rates one band too high.

### 4. Analysis

Write up your conclusions:

- **a. Trade-offs.** What are the practical differences between human evaluation and LLM-as-a-judge, across **cost**, **scale**, **consistency**, and **accuracy**? Be concrete — use your own numbers: how long did 15 manual rows take you, what did the full judge run cost and how long did it take?
  *(Note that "consistency" and "accuracy" pull in opposite directions here. Humans are the gold standard for accuracy but disagree with themselves on different days. A judge at temperature 0 is highly consistent — including consistently wrong in the same way.)*
- **b. Recommendation.** For a production system generating **thousands of descriptions daily**, which approach would you recommend, and why? Consider whether it has to be a single choice — where would you place a human in the loop, and what would trigger escalation?

✅ **Done when:** all products have judge verdicts and a `final_score`, you have per-criterion agreement rates, and you've written the trade-off analysis and recommendation.

---

## What to submit

1. **Your code** — generation (Task 2), improvement experiments (Task 4), and the judge (Tasks 5–6). Committed and pushed to your course repo.
2. **`assignment_02.xlsx`** — one row per product, containing:
   - `generated_description`, `latency_ms`, `input_tokens`, `output_tokens`
   - your **human** verdicts + `final_score` for the 10–15 sampled rows
   - the **judge's** verdicts, explanations, and `final_score` for all rows
3. **A write-up** (markdown) containing:
   - **Task 1:** your full rubric — all 18 rating definitions, the cumulative pass bar, and the go/no-go rules
   - **Task 3:** your baseline analysis (best / worst criteria)
   - **Task 4:** each experiment — what you changed, why, and the resulting scores vs. baseline
   - **Task 5:** why `explanation` comes before `verdict` in the schema
   - **Task 6:** agreement rate per criterion, where and why you diverged, the trade-off analysis, and your production recommendation

---

## The takeaway

You'll finish this assignment with a working generate-and-evaluate pipeline. But the durable lesson is smaller and more useful than the pipeline:

**Your evaluation is only as good as your rubric.** Ambiguity in the rubric doesn't disappear when you automate — it just gets applied at scale, silently, by a model that will never tell you it was confused. The hardest and most valuable work in this assignment is Task 1, and you'll only find out how good your Task 1 was when Task 6 shows you where you and the judge disagreed.
