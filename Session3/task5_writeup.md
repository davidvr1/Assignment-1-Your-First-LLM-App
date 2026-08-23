# Assignment 3 write-up: RAG over a Hebrew health-insurance corpus

## Task 1: corpus and baseline

**Corpus.** 6 documents on Israeli supplementary health insurance (שב"ן), across 3
formats (PDF, Markdown, plain text — exceeds the ≥2 requirement). Full manifest and
rationale in `corpus/MANIFEST.md`; summary:

- `kelalit_moshlam_takanon.pdf` — a **real** 20-page PDF carried over from
  Session1/Session2. Its embedded font uses a non-standard Hebrew glyph mapping, so
  `pypdf` extraction is **genuinely garbled** (reversed/cp1255-mangled text) — this is
  the corpus's messy document, not a contrived one.
- `kelalit_moshlam_summary.md` — a clean, human-written Hebrew summary of that same
  PDF's plan, letting the write-up compare a clean parse against the mangled one.
- `maccabi_zahav_takanon.md`, `meuhedet_adif_takanon.md` — two synthetic competing
  plans (codes `MZ-2025`, `MA-ADIF-24`), deliberately overlapping/conflicting with
  Kelalit's numbers rather than being one-fact-per-paragraph, per the assignment's
  warning against a too-easy corpus.
- `comparison_faq.md` — synthetic cross-document FAQ, the deliberate source for the 2
  required multi-hop questions.
- `claims_process_guide.txt` — generic, insurer-agnostic claims process; explicitly
  does **not** contain caps/waiting-periods/coverage numbers, giving a genuine (not
  invented) source for unanswerable questions.

**Baseline (no retrieval at all).** `baseline.py` ran all 30 eval questions directly
against the generator model (Claude Haiku) with zero documents, classifying each answer
as refused / correct / hallucinated. Full results: `baseline_results.xlsx`.

| class | count (n=30) |
|---|---|
| refused | 30 (100%) |
| answered correctly | 0 |
| hallucinated | 0 |

avg latency 1129ms, avg input tokens 171, avg output tokens 42.

**The model refused every single question.** This is a direct, expected consequence of
the corpus design: the specific numbers, plan codes, and caps are all invented for this
assignment, so Haiku has no training-data shortcut to answer correctly, and — per its
system prompt's instruction to only answer if it knows — it refused rather than guess.
This is a clean baseline in one sense (any later RAG correctness is real, not the model
coasting on prior knowledge) but it also has a consequence documented later in Task 5
§3/§4: it means Task 5's required "RAG made it worse" and "right answer, broken
pipeline" cases can't be demonstrated on this corpus, because there's no correct
baseline answer for retrieval noise to either drag down or accidentally validate.

---

## Task 3: what reading the chunks and test retrievals showed

`build_index.py` parses → chunks (1000 chars / 150 overlap) → embeds (`BAAI/bge-small-en-v1.5`)
→ saves a FAISS index, then prints 10 random chunks and 3 canned test retrievals for manual
inspection before building generation on top of it. Re-run to capture this output fresh (no
API cost — embedding is local); full stdout is not committed (it's a one-off inspection
step, not a deliverable file), findings below.

**Parsing damage:** confirmed on the PDF (`kelalit_moshlam_takanon.pdf`) exactly as the
manifest predicted — text extracted by `pypdf` shows the known Hebrew glyph-mapping
issue for that document's embedded font. Comparing chunks from the PDF against the
equivalent section in `kelalit_moshlam_summary.md` (the clean hand-written parallel)
makes the difference directly visible: the Markdown chunk reads as normal Hebrew
prose, the PDF chunk from the same topic does not extract cleanly. This is the
corpus's intended "genuinely messy document" case.

**Chunk boundaries:** the Markdown/text documents' chunks generally hold together —
`RecursiveCharacterTextSplitter` prefers paragraph/sentence breaks, and at 1000
characters most chunks stay within one section (e.g. one coverage clause or one FAQ
entry). The PDF's chunks are the exception, both because of the glyph-mangling above
and because a 1000-character cut through a 20-page legal document sometimes lands
mid-clause.

**3 test retrievals (bare `similarity_search`, k=3):** for the three canned questions in
`build_index.py` (organ-transplant cap, waiting period for private surgery, orthodontic
coverage for children), the correct source document was retrieved for all 3 at the
document level. This looked encouraging at the time, but Task 5 later shows document-level
retrieval success is not the same as chunk-level success — see Task 5 §2 for the full
finding, which traces back to a decision made right here in Task 3: prepending the
document filename to every chunk's text (the "extra-credit enrichment" step) made every
chunk from a given document look more alike to the embedding model, including causing
generic intro/title chunks to out-compete the actual answer-bearing content chunks.
That trade-off wasn't visible from 3 hand-picked test queries — it only showed up once
Task 5 measured hit-rate against all 30 real questions with ground-truth evidence
labels, which is exactly why the assignment insists on the numeric evaluation rather
than trusting a few spot-checks.

**A second bug found by re-reading the chunks (fixed after Task 6):** the 10-random-chunk
sample also surfaced `corpus/MANIFEST.md` itself — the write-up file describing the
corpus — appearing as an indexed, retrievable chunk. `load_documents()` in
`build_index.py` was loading every `.md`/`.txt`/`.pdf` file under `corpus/`, including
its own documentation, not just the 6 actual insurance documents. This means every
result reported in Task 5, Experiment 1, and Experiment 2 above was computed against an
index containing 4 chunks of English corpus-documentation noise (96 chunks total, not
the intended 92).

**Impact, measured, not assumed:** fixed `load_documents()` to skip `MANIFEST.md`,
rebuilt the index (92 chunks), and re-ran both free fact-hit sweeps
(`sweep_hitrate.py`, `sweep_hybrid.py`) to compare against the contaminated-index
numbers before touching any paid judge results. **Every single fact-hit-rate number
was identical, bit-for-bit**, across all 8 chunk-size/prefix/K configs in the Task 6
sweep and both dense/hybrid configs in the Experiment 2 sweep. The manifest's English,
non-insurance content apparently never ranked into any top-K slot for any of the 30
Hebrew questions, dense or lexical. Given that, and per the assignment's own
"sweep against hit-rate before spending judge budget" advice, the paid Task 5/6
results were **not** re-run — the free check is strong enough evidence that this bug,
while real and worth fixing, was not load-bearing for any number in this write-up.
`build_index.py` now excludes it going forward.

---

## 1. Headline table (full table + easy/hard slices in `eval_summary.md`)

| metric | baseline | RAG | slice where it matters most |
|---|---|---|---|
| correctness (judge) | 0% (all refused) | 33% | — |
| context relevance (judge) | n/a | 23% | — |
| faithfulness (judge) | n/a | 97% | — |
| answer relevance (judge) | n/a | 97% | — |
| refusal correct (code) | n/a | 37% | easy 30%, hard 50% |
| hit-rate@5 (code, doc-level) | n/a | 96% | easy 100%, hard 86% |
| avg latency | 1.1s | 2.3s | — |
| avg input tokens | 171 | 4072 | — |

Full per-question data: `assignment_03.xlsx`. Raw run logs and the summary table generator: `evaluate.py`.

## 2. The central finding: hit-rate@K (doc-level) is lying to you here

**96% hit-rate@5, but only 37% refusal-correctness and 33% correctness.** That gap is the
single most important number in this assignment, and it's caused by exactly the trap
Task 3 flagged during chunk-reading: retrieval is measured **per-document**, but the
model needs the **specific chunk**, and those are not the same thing once "hit" means
"any chunk from the right file showed up."

Concretely: for question 1 ("what's the co-pay for a specialist consultation under
Maccabi Zahav?"), the top-5 retrieval returned:

```
[1] maccabi_zahav_takanon.md   -- section 3, membership eligibility rules
[2] kelalit_moshlam_summary.md -- chapter ה, intro to the services table
[3] meuhedet_adif_takanon.md   -- document intro
[4] comparison_faq.md          -- FAQ about seniority transfer between plans
[5] maccabi_zahav_takanon.md   -- document title block
```

Two of five chunks are from the right document (`maccabi_zahav_takanon.md`), so this
question counts as a **hit** at doc-level (`hit_at_k=1.0` in `assignment_03.xlsx`). But
neither retrieved chunk is Section 5's coverage table, which is where "45 ₪ לביקור"
actually lives. The system correctly refused rather than guess — which is the *right*
behavior given what it was shown — but it's counted as a false refusal because the
question genuinely was answerable, just not from what got retrieved.

**Root cause, following the trail from Task 3:** every chunk was prepended with
`[doc_name]` (build_index.py's extra-credit enrichment step), and every document's
title/intro chunk ("תקנון X - תמצית סינתטית...") is semantically close to *every*
question about that insurer, regardless of topic. So similarity search keeps pulling
back one "representative" chunk per document — proof of relevance at the file level,
useless at the fact level. This is a concrete, evidence-backed lead for Task 6: try
dropping the title-prefix (or shortening it to a doc code instead of the full title),
and/or shrinking chunk_size so the intro doesn't dominate as much of the embedding
space relative to content sections.

**Lesson for the rubric itself:** hit-rate@K needs a page/section check, not just a
doc-name check, or it will systematically overstate retrieval quality on documents
that have one dominant, generic-sounding intro chunk. The current `hit_at_k` in
`evaluate.py` only checks `doc_name` membership — tightening it to also require the
retrieved chunk's `page`/section to match `evidence_page` would have caught this before
the judges did.

## 3. Required finding (a): a question RAG made worse

**Does not occur in this dataset, and here's why that's itself informative.** Case (a)
requires a baseline answer that was *right* (from training knowledge) which RAG then
corrupted with irrelevant local context. But Task 1's baseline was **0% correct / 100%
refused across all 30 questions** — because the corpus is entirely synthetic
(fictional insurer codes like `MZ-2025`, invented numbers), Haiku has no training-data
shortcut to get *right* in the first place, so there's no good baseline answer for
retrieval noise to drag down. This is a direct, foreseeable consequence of building a
synthetic corpus for a domain (Israeli שב"ן plans) the model *does* have generic
priors about, but not these specific fictional numbers — worth stating explicitly
rather than papering over the missing case.

## 4. Required finding (b): "right answer, broken pipeline"

**Also does not occur here, for the same structural reason.** This case needs
`judge_correctness=True` AND `hit_at_k=False` — the model happening to know the fact
despite bad retrieval. Checking `assignment_03.xlsx`: every row where correctness
passed (ids 5, 11, 13, 15, 20, 24, 27) also has `hit_at_k=1.0`; the only row with
`hit_at_k=0` (id 23, orthodontics coverage) has `judge_correctness=False`. Same root
cause as §3 — a synthetic corpus with invented facts removes the model's ability to
"already know" anything, so this specific failure mode (which the assignment says is
usually the scary one, because it looks like success) can't be demonstrated on this
corpus. **If I had one more week**, this is the first thing I'd fix: mix in a handful
of questions about *real*, well-known facts (e.g. the actual national health basket
law, real waiting-period conventions) alongside the synthetic plans, specifically to
give cases (a) and (b) a chance to appear.

## 5. Required finding (c): retrieval vs. generation, 5 worst answers

Five worst = lowest combined judge score, all `answerable=True` with
`judge_correctness=False`: ids **1, 3, 4, 6, 7** (all easy, all false refusals).

| id | question topic | right doc retrieved? | right chunk retrieved? | verdict |
|---|---|---|---|---|
| 1 | Maccabi specialist consultation co-pay | yes (2/5 chunks, both title/§3) | no — §5 coverage table never retrieved | **retrieval** |
| 3 | Kelalit organ transplant cap (chapter א) | yes (2/5, both title/§ה intro) | no — chapter א's transplant figures never retrieved | **retrieval** |
| 4 | Meuhedet dental emergency count/co-pay (chapter ג) | yes (2/5, title + §3 enrollment) | no — chapter ג never retrieved | **retrieval** |
| 6 | Meuhedet children's dental age limit (chapter ג) | yes (1/5, title chunk) | no — chapter ג never retrieved | **retrieval** |
| 7 | Maccabi hearing-aid reimbursement cap (§5 table) | yes (1/5, title chunk) | no — §5 coverage table never retrieved | **retrieval** |

**Count: 5/5 retrieval failures, 0/5 generation failures.** Every one of the worst
answers is the model correctly declining to invent a number it wasn't shown, not the
model mishandling context it had. This cleanly answers where Task 6 effort should go:
**do not touch the generation prompt** — it's already doing the right thing (0
hallucinations, 97% faithfulness, 0 invalid citations across the whole run). Fix
retrieval: specifically, stop letting document-intro chunks compete equally with
content chunks (drop/shrink the title prefix), and consider raising K or shrinking
chunk_size so the §5-style tables that hold the actual numbers have a better chance of
surfacing.

## 6. One paragraph: what I'd fix first, and which number told me that

*(Written after Task 5, before running Task 6 — kept here unedited as the pre-registered
prediction; see the updated version below for what actually happened.)*

I'd fix the chunk-prefixing/chunk-size issue identified in §2 first, because the
96%-hit-rate-vs-33%-correctness gap is the single number that both (a) is cheapest to
act on — it's a Task 6 lever, free to sweep against hit-rate with zero judge calls per
the assignment's own advice — and (b) is clearly the bottleneck: faithfulness (97%) and
answer relevance (97%) show generation is already trustworthy, so every point of
correctness lost between the 96% doc-level hit-rate and the 33% correctness rate is
sitting in retrieval, not generation. That's the ceiling-on-everything-above-it point
from the assignment's own framing, borne out numerically rather than asserted.

### Updated after Task 6

The prediction was half right. I never actually shipped the chunk-prefix/chunk-size fix
— the free sweep (Task 6 Step 0) refuted it before spending any judge budget: dropping
the title prefix made fact-hit-rate *worse* (45%→32%), and shrinking chunk_size made it
much worse (down to 14–18%), the opposite of what §2's diagnosis predicted. That's the
value of the free-sweep-first discipline: a plausible, well-argued hypothesis from
reading real chunks turned out to be wrong, and it cost zero API calls to find out.

What actually worked, across two full experiments, was **top-K 5→10** (+7pp
correctness) and, more decisively, **hybrid dense+BM25 retrieval at k=10** (+17pp
correctness over dense-only k=10, zero regressions, and it fixed the one
exact-identifier question that raising K alone had broken). So if I had one more week
now, I'd spend it differently than I predicted: not on chunk-prefix/chunk-size (refuted),
but on (1) **re-running Task 5's full evaluation on the hybrid+k=10 config** to get a
clean apples-to-apples baseline-vs-RAG table on the *final* pipeline rather than the k=5
one this write-up's §1 table still reflects, and (2) **a page/section-level hit-rate
metric**, per §2's own lesson — doc-level hit-rate stayed flat at 96%→93% across every
experiment in this document while correctness moved 20+ points, so it never once flagged
which change actually helped; every real signal in Task 6 came from the fact-level proxy
I had to build myself, which means the assignment's own free "sweep against hit-rate"
advice was silently broken for this corpus from Task 3 onward and I only found out by
building a second metric to check the first one.

---

# Task 6: improvement cycles

## Step 0 — free hit-rate sweep (zero API calls)

Before spending judge-call budget, swept chunk_size, title-prefix, and top-K against a
**fact-level** hit metric (`sweep_hitrate.py`) — not Task 5's doc-level `hit_at_k`, which
had already saturated at 96% while correctness sat at 33% (§2 above). Full results in
`sweep_results.md`:

| config | fact-level hit-rate |
|---|---|
| baseline (1000/150, prefix on, k=5) | 45% |
| no title prefix, k=5 | 32% |
| no title prefix, k=8 | 41% |
| chunk_size=500/75, prefix on, k=5 | 18% |
| chunk_size=500/75, no prefix, k=5 | 18% |
| chunk_size=300/50, no prefix, k=5 | 14% |
| prefix on, k=8 | 45% |
| **prefix on, k=10** | **50%** |

Two candidates were refuted for free: dropping the title prefix made things *worse*
(45%→32%), and shrinking chunk_size made things much worse (down to 14–18%) — smaller
chunks apparently split the coverage tables (the ones holding the actual co-pay/cap
numbers) across chunk boundaries more often than they isolated them. Only raising K
helped, and only past k=8. That's the one candidate worth a full judge run.

## Experiment 1: top-K 5 → 10

**Hypothesis (written before running):** Hit-rate fails on my worst 5 questions because
the answer-bearing chunk (a §5 coverage table or similar) is present in the index but
ranked outside the top 5, crowded out by generic document-intro chunks (§2's root
cause). Raising K to 10 should pull those chunks into context without needing to touch
chunking or prompting.

**The one thing changed:** `k=5 → k=10` in `answer_with_rag`, same `faiss_index/` (same
chunk_size=1000/overlap=150, title-prefix on) as Task 5. Full re-run via
`experiment_topk.py` → `assignment_03_exp3_topk10.xlsx`, `exp3_topk10_summary.md`.

**Result — full table, k=5 (Task 5) vs. k=10:**

| metric | k=5 | k=10 | delta |
|---|---|---|---|
| correctness (judge) | 33% | 40% | **+7pp** |
| context relevance (judge) | 23% | 33% | +10pp |
| faithfulness (judge) | 97% | 97% | 0 |
| answer relevance (judge) | 97% | 93% | -4pp |
| refusal correct (code) | 37% | 43% | +6pp |
| hit-rate@K (doc-level, code) | 96% | 96% | 0 (already saturated) |
| avg latency (ms) | 2275 | 2635 | +360ms |
| avg input tokens | 4072 | 6710 | +2638 |

Easy/hard slices (from `exp3_topk10_summary.md`): correctness moved 25%→40% on easy,
stayed flat at 50%→40% on hard (one question's worth of noise at n=10 — see caveat
below). Context relevance improved on easy (25%→40%) but not hard (20%→20%).

**Per-question detail:** 4 questions flipped correctness verdict. Three flipped
wrong→correct (ids 16, 17, 18 — all easy, all questions whose evidence lives in a
coverage/procedure table that k=5 didn't reach). One flipped correct→wrong (id 24 —
the exact-identifier "plan code" question). That's a useful, foreseeable counter-example:
widening K adds more chunks to the context, which for an embedding-weak exact-match
query means more *near-miss* codes competing for the model's attention, not the right
one arriving. Confirms the assignment's own warning that identifier questions need a
different lever (hybrid/BM25 search), not top-K.

**What I learned:** the hypothesis was directionally right — correctness, context
relevance, and refusal-correctness all improved, for the reason predicted (previously
excluded coverage-table chunks now enter context) — but it's not free: input tokens
nearly doubled (4072→6710, ~65% more, at fixed per-token cost) and it made the one
exact-identifier question worse, not better. **Caveat on the numbers themselves:** at
n=30 (and n=10 for the hard slice), a handful of flips is a few percentage points each
— the hard-slice correctness "drop" (50%→40%) is a single question changing its
verdict, not a real regression, and should be read as noise rather than a finding.
doc-level hit-rate stayed flat at 96% in both runs, which is exactly why Step 0 built
the fact-level proxy instead — the doc-level metric would have shown *nothing* moved.

**Conclusion:** keep k=10 over k=5 — it's a net improvement on the metric that matters
(correctness) at the cost of latency/token spend, with the identifier-question
regression flagged as a separate, unsolved problem for a hybrid-search experiment.

## Experiment 2: hybrid (dense + BM25) retrieval, on top of k=10

**Hypothesis (written before running):** id-24 (the exact-identifier "plan code"
question) got *worse* going from k=5 to k=10 in Experiment 1 — more chunks in context
gave the model more near-miss codes to confuse, not the right one. Dense embeddings are
known to be weak at exact-match tokens (the assignment's own warning). Adding a lexical
retriever (BM25) alongside the dense one and fusing the two rankings (Reciprocal Rank
Fusion, k_rrf=60 — standard, weight-free) should surface the plan-code chunk on lexical
grounds even when its embedding is unremarkable, without touching chunk_size or K again.

**Free proxy first (`sweep_hybrid.py`, zero API calls):** fact-level hit-rate jumped
50% → 68% at k=10 (dense-only → hybrid). That's the largest single move of any lever
tried in Task 6, so it earned the full judge run. Note: even at this free-proxy stage,
id-24's *fact*-hit was already `True` for dense-only k=10 — the salient token (the plan
code string) was technically present somewhere in the top 10 chunks even before adding
BM25. That should have predicted id-24 might not flip on retrieval grounds alone; see
below for what actually happened.

**The one thing changed:** retriever type only — `HybridRetriever` (dense FAISS +
BM25Okapi, RRF-fused, `dense_k=bm25_k=20` before fusing down to the final `k=10`)
replacing plain `vectorstore.similarity_search`. Same k=10, same chunks
(chunk_size=1000/overlap=150, title-prefix on) as Experiment 1. Full re-run via
`experiment_hybrid.py` → `assignment_03_exp4_hybrid.xlsx`, `exp4_hybrid_summary.md`.

**Note on the run:** ids 29 and 30 initially failed on a `credit balance too low` error
from the Anthropic API partway through the first pass; after topping up the account,
both were re-run and merged back in. All numbers below are the full **n=30**.

**Result — dense k=10 (Exp.1) vs. hybrid k=10 (Exp.2), full n=30:**

| metric | dense k=10 | hybrid k=10 | delta |
|---|---|---|---|
| correctness (judge) | 40% | 57% | **+17pp** |
| context relevance (judge) | 33% | 63% | +30pp |
| faithfulness (judge) | 97% | 100% | +3pp |
| answer relevance (judge) | 93% | 100% | +7pp |
| refusal correct (code) | 43% | 57% | +14pp |
| hit-rate@10 (doc-level, code) | 96% | 93% | -3pp (noise: 1 question, n=27) |
| avg latency (ms) | 2635 | 3041 | +406ms |
| avg input tokens | 6710 | 6887 | +177 |

Easy/hard slices (`exp4_hybrid_summary.md`): correctness improved on both — easy
40%→50%, hard 40%→70% (id-29's unanswerable question correctly refused, plus 2 more
hard questions flipped to correct). Context relevance improved sharply on easy
(33%→65%) and moderately on hard (20%→60%).

**Per-question detail:** 5 questions flipped wrong→correct (ids 3, 8, 21, 22, **24**),
**zero regressions** anywhere in the full 30. Crucially, **id-24 flipped correct** —
hybrid retrieval fixed exactly the failure Experiment 1 introduced. Exp.1's answer was
a refusal ("לא מצאתי את התשובה..."); Exp.2's answer states the plan code `MA-ADIF-24`
directly with a citation. That's the hypothesis confirmed, not just directionally but
on the specific question it targeted — though per the free-proxy note above, the
*fact* was already retrievable by dense-only k=10, so the fix here looks like it came
from BM25 changing the chunk's **rank** (pushing it into the model's effective
attention) rather than making it retrievable for the first time. A useful refinement
for future work: log rank position, not just hit/miss, to distinguish "retrieved but
buried" from "not retrieved at all" — Task 5's binary hit-rate metric can't see this
distinction, and neither can the fact-hit proxy.

**What I learned:** hybrid retrieval was a clear, uniform win over dense-only k=10 on
this corpus — no tradeoff surfaced beyond a modest latency/token increase (~6% more
input tokens, ~400ms slower), and it fixed the one regression Experiment 1 created,
with no new ones introduced. The corpus here is Hebrew insurance-plan text with a
handful of literal alphanumeric codes (`MA-ADIF-24` etc.), which is close to the
textbook case BM25 is good at — this result may not generalize as cleanly to a corpus
without exact-identifier content.

**Conclusion:** adopt hybrid (dense+BM25, RRF-fused) retrieval at k=10 as the best
configuration found in Task 6 — the largest, cleanest, and only regression-free
improvement of the two experiments run.
