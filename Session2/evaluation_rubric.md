# Evaluation Rubric

Six dimensions, each scored **good / ok / bad**. Every response is rated on all six independently — don't let a failure on one dimension bleed into another (see the note on Grounding at the end).

---

## 1. Fluency
**Question:** Do the sentences read naturally, like a human wrote them?

Fluency is independent of correctness — a confidently written *wrong* answer can still score `good` here. Judge only whether the text reads smoothly, not whether it's true, helpful, or on-brand.

| Label | Test | Signals |
|---|---|---|
| **good** | You read it once, nothing snags. You'd believe a competent native speaker wrote it. | Natural word combinations, varied sentence length, connectors that fit the logic, consistent register |
| **ok** | Understandable on the first pass, but something reads "off" and you can point at it. | Translationese, stiff/robotic phrasing, wordiness, repetitive sentence structure, over-hedging |
| **bad** | You had to re-read a sentence to parse it, or it's visibly broken. | Broken syntax, sentences that don't connect, mid-text register switch, truncation, degenerate repetition |

**Aggregation:** zero "off" spots → good. One or two, none forcing a re-read → ok. Any re-read-forcing error, or three+ "off" spots → bad.

**Out of scope:** factual errors, missing information, length, formatting, grammar rule-violations (→ Grammar).

**Anchors:**
- good — "The migration finished in about four minutes. Most of that was re-indexing, so the actual data copy was fast."
- ok — "The migration process was completed in approximately four minutes of time. The majority of this duration was consumed by re-indexing operations, therefore the copying of the data itself was fast."
- bad — "The migration are finish in four minutes approximately, which the most of it re-index and because of this the data copy was fast actually."

**Tie-breaker:** if torn between two labels for more than a few seconds, pick the lower one.

---

## 2. Grammar
**Question:** Correct spelling, punctuation, agreement?

Grammar is rule-violations; Fluency is style. A sentence can be grammatically flawless and still sound stiff (Fluency issue, not Grammar). A sentence can sound a little clunky but be grammatically correct (still Grammar: good).

| Label | Test | Signals |
|---|---|---|
| **good** | Zero rule violations. | Correct subject-verb agreement, punctuation, spelling, tense throughout |
| **ok** | Minor, doesn't obscure meaning. | One typo, a missing/extra comma, one small capitalization slip |
| **bad** | Violation changes or obscures meaning, or multiple minor errors stack up. | Agreement errors ("the product have"), wrong tense, run-ons merging unrelated ideas, misspelled product/brand names, 2+ "ok"-level errors |

**Out of scope:** awkward-but-correct phrasing (→ Fluency), unnatural word choice that's still grammatically valid (→ Fluency).

**Anchors:**
- good — "This plan includes unlimited storage and 24/7 support."
- ok — "This plan includes unlimited storage, and 24/7 support."
- bad — "This plan include unlimited storage and 24/7 supports."

---

## 3. Tone
**Question:** Does it match a friendly, credible sales voice?

Define the target voice concretely before rating: warm but not gushing, confident but not hype-y, helpful but not pushy, plain language over jargon, no false urgency, no unverifiable superlatives.

| Label | Test | Signals |
|---|---|---|
| **good** | Sounds like a knowledgeable person on your side, not a script. | Warm, direct, confident, no filler, no pressure tactics |
| **ok** | Recognizably on-brand but leaning too far one direction. | Slightly too stiff/corporate, OR slightly too casual/gushing, OR mildly salesy — nothing a customer would flag |
| **bad** | Actively works against trust or brand fit. | Pushy/high-pressure, over-the-top hype with no basis, cold/robotic, unverifiable superlatives, sounds scripted |

**Out of scope:** factual accuracy of claims (→ Grounding), grammatical cleanliness (→ Grammar).

**Anchors:**
- good — "Our plan covers everything most teams need, and if you outgrow it, upgrading takes two minutes."
- ok — "Our plan is meticulously engineered to comprehensively address your organizational requirements."
- bad — "This is a LIMITED-TIME opportunity you can't afford to miss — the best deal you'll ever see!"

---

## 4. Length
**Question:** Is it within 50–90 words?

The only fully objective dimension — count, don't judge.

| Label | Rule |
|---|---|
| **good** | 50–90 words, inclusive |
| **ok** | Within 20% of either bound (40–49 or 91–108 words) |
| **bad** | Outside that range (< 40 or > 108 words) |

**Decide once, apply consistently:**
- If a slightly-off length is genuinely acceptable to the business, keep the "ok" band. If not, drop it and use a hard pass/fail: good = in range, bad = out of range.
- Decide whether you're counting words in the raw output or after stripping boilerplate (signatures, disclaimers) — lock this in so annotators count the same way.

---

## 5. Grounding
**Question:** Does it stick to the supplied attributes, inventing nothing?

This is a factuality checklist against the source attributes, not a vibe read.

| Label | Test | Signals |
|---|---|---|
| **good** | Every claim traces to a supplied attribute. Nothing added, nothing contradicted. | All stated facts (price, features, specs, availability) match the input exactly |
| **ok** | No fabricated facts, but there's unsupported *framing* — a reasonable inference not explicitly given. | E.g. "perfect for growing teams" inferred from "scales to 500 users"; rounding a number; light rephrasing that preserves meaning |
| **bad** | Any invented or contradicted fact. | A feature, number, price, or claim absent from the source and not a safe inference; a stat contradicting the supplied attributes; an uninstructed competitor comparison |

**Out of scope:** persuasiveness of phrasing (→ Tone), grammatical cleanliness (→ Grammar).

**Anchors** (given attributes: price $29/mo, 5 users, email support):
- good — "At $29/month, you get support for up to 5 users, with email support included."
- ok — "At $29/month, this plan is a great fit for small teams, with email support included."
- bad — "At $29/month, you get support for up to 5 users, with 24/7 phone support."

**Tie-breaker:** any single hallucinated fact caps the whole response at **bad**, regardless of how good the rest is. This is the highest-stakes dimension (compliance/legal risk) — don't let volume of good content offset one fabrication.

---

## 6. Latency
**Question:** Time per call (time to first byte / full response).

Mechanically different from the other five — measured off timestamps, not read by a human. Script this and reserve human eyes for spot-checks.

This is a short (50–90 word) generated sales description, not a chat conversation, so the bar is calibrated for a single, self-contained generation call rather than an interactive back-and-forth.

| Label | Time to first byte | Full response |
|---|---|---|
| **good** | ≤ 6
| **ok** | <7
| **bad** |>7

**How the two numbers combine into one label:** score TTFB and full-response time against their own rows independently, then the row's Latency label is the **worse of the two** — e.g. TTFB lands in `good` but full response lands in `ok` → the call is rated `ok`. This prevents a fast-starting-but-slow-finishing call from hiding behind a good TTFB.

**Per-call vs. distribution:** the label above is scored **per call** — it's what feeds the Grounding-style pass/fail row for that specific description, so one slow call can fail that one row. Separately, track **p95 TTFB and p95 full response across all calls** as a system-health metric; a healthy system should have its p95 sitting inside the `good` band even though individual calls will occasionally land in `ok`. Per-call scoring diagnoses one bad output; the p95 view tells you whether the system as a whole needs attention. Don't conflate the two — a single `bad` per-call score is a data point, not proof the system is unhealthy, until it shows up repeatedly in the p95.

**Assumption stated explicitly:** these thresholds (1.0 s / 3.0 s / 2.5 s / 6.0 s) are reasonable defaults for a short single-turn generation call and are what the anchors and pass/fail examples below assume. If your actual SLA differs, replace only the four numbers in the table above — nothing else in this document depends on their exact values.

---

## Pass / Fail Rules

Six per-criterion ratings don't ship a product — this section converts them into one bit: **publish or reject.** Apply the two rule sets below, in order, to every row.

### Step 1 — Go / No-Go rules (checked first)

Any single **bad** rating, on any criterion, rejects the row outright — no averaging, no exceptions. A description that's flawless on five dimensions and fabricates one fact is not "mostly good," it's unpublishable.

| Criterion rated `bad` | Result | Why this one criterion can't be outvoted |
|---|---|---|
| Grounding | **REJECT** | Fabricated or contradicted facts are a compliance/legal risk regardless of how well-written the rest is |
| Grammar | **REJECT** | Visible rule-violations undermine credibility on their own |
| Length | **REJECT** | Fails a hard structural requirement of the deliverable |
| Tone | **REJECT** | Off-brand or high-pressure copy can mislead or alienate a customer even if factually and grammatically clean |
| Fluency | **REJECT** | Text that forces a re-read isn't publishable copy no matter how accurate it is |
| Latency | **REJECT** | Fails the performance SLA — this is an engineering/ops rejection rather than a content one, but it still blocks go-live |

**Rule, stated plainly:** *If any criterion = bad, the row is rejected. Stop here — do not proceed to Step 2.*

### Step 2 — Cumulative pass bar (only for rows with zero `bad` ratings)

A row that clears Step 1 (no `bad` anywhere) still isn't automatically publishable — six mediocre "ok"s shouldn't pass just because none of them individually failed.

**Rule:** *A row PASSES if at least 4 of the 6 criteria are rated `good`, and the remaining criteria (at most 2) are rated no lower than `ok`.*
*A row with 3 or fewer `good` ratings — even with zero `bad`s — FAILS and is sent back for revision.*

| Good count (of 6) | Bad count | Result |
|---|---|---|
| 6 | 0 | PASS |
| 5 | 0 | PASS |
| 4 | 0 (remaining 2 are `ok`) | PASS |
| 3 | 0 (remaining 3 are `ok`) | FAIL — below the pass bar |
| any | ≥ 1 | FAIL — Step 1 already rejected it |

### Applying both steps together

1. Scan the row for any `bad`. Found one → **REJECT**, done.
2. None found → count the `good`s. ≥ 4 → **PASS**. ≤ 3 → **FAIL**.

Use this exact two-step sequence every time — by hand, after revision, and when a judge (human or model) applies it — so the three passes are directly comparable. Don't substitute judgment for the count; if a row is borderline, that's a signal the rubric's band definitions (not the pass bar) need tightening, not that this row deserves an exception.

---

## Structural note

**Grounding should gate the others.** If a response scores `bad` on Grounding (invented facts), its Fluency/Grammar/Tone scores are close to irrelevant — a fluent, well-punctuated, on-brand lie is worse than a clunky true statement. Consider flagging any Grounding failure for review regardless of how the other five score.

## Before scaling up

Pilot ~20 items with two annotators and measure agreement. Disagreement will concentrate on the good/ok or ok/bad borders — that's where to spend refinement effort. Turn every resolved disagreement into a new anchor example in this document.
