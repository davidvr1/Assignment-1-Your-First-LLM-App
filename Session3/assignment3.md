# Assignment 3 — RAG

**Lecture 3 · RAG & Grounding**

---

## What this assignment is about

In Assignment 1 you built `file_qa.py`: it read **one** file, pasted the **whole thing** into the prompt, and answered from it. That works beautifully — right up to the moment you have fifty documents instead of one, and they don't fit in the context window.

In Assignment 2 you built the instrument that tells you whether an answer is any good.

This assignment joins them. You'll replace "paste the whole document" with **retrieve the relevant few paragraphs**, and then you'll use your evaluation skills to prove — with numbers — whether that actually helped.

Here's the arc:

1. **Build a corpus and re-establish your baseline** — what does the model answer with *no* retrieval at all?
2. **Write your eval set first** — questions, reference answers, and which document holds the evidence.
3. **Index** — parse → chunk → enrich → embed → store.
4. **Retrieve and generate** — top-K chunks into a grounded prompt, with citations.
5. **Evaluate** — the RAG triad, retrieval hit-rate, and a head-to-head against your baseline.
6. **Improve** — three experiments, one variable at a time.

The through-line from class: **retrieval is the bottleneck, and a good final answer doesn't prove your pipeline worked.** Most of the grade is in whether you can tell those two things apart.

> **Keep working in your course repo.** Same repo as Assignments 1 and 2. You will need your Assignment 1 baseline answers and your Assignment 2 evaluation code — don't start a new project.

---

## Setup

Add to your existing virtual environment:

```bash
pip install langchain langchain-community langchain-huggingface
pip install sentence-transformers faiss-cpu          # or: chromadb
pip install pypdf openpyxl pandas
```

You'll keep using your **Claude** client from Assignment 1 (the OpenAI-compatible endpoint, `base_url` + `ANTHROPIC_API_KEY`).

Two roles, two models — same discipline as Assignment 2:

| Role | Model | Why |
|---|---|---|
| **Generator** | `claude-haiku-4-5` | cheap and fast; you'll call it hundreds of times |
| **Judge** | `claude-sonnet-5` | stronger, and *not* the model being judged |

> **A caveat worth stating in your write-up.** Haiku and Sonnet are the same model family, so using one to judge the other still leaves some **self-enhancement bias** (Assignment 2, Task 5). A cleaner setup uses a different provider or an open model as the judge. If you want the bonus rigour, spot-check 10 rows with a second judge from a different family and report whether the verdicts move.

**Embeddings run locally from HuggingFace** — no API key, no per-token cost:

```
BAAI/bge-small-en-v1.5
```

> ⚠️ **Read the model card before you use it.** `bge` models expect a **prefix on the query** (something like `Represent this sentence for searching relevant passages:`) but *not* on the indexed passages. Getting this wrong doesn't crash anything — it just quietly makes your retrieval worse. This is the "read the model card" point from class, and it costs people real marks.
>
> ⚠️ **Whatever you choose, use the same embedding model for indexing and for querying.** Changing it means re-indexing your whole corpus from scratch.

---

## The use case: your Q&A tool, grounded

Keep the **domain you picked in Assignment 1**. What changes is the scale of its knowledge.

**Assemble a corpus of 5–10 documents.** Requirements:

- **At least 2 different file formats** — e.g. PDF *and* Markdown/text. You need to feel the parsing problem, not read about it.
- (optional) **One document that is genuinely messy** — a real PDF with tables, columns, or a scan. If your corpus is all clean Markdown you've skipped the hardest step in RAG.
- Enough total text that it **does not fit comfortably in one prompt**. That's the whole reason retrieval exists — so favour a few **long** documents over many short ones.
- Content you can actually **verify answers against**, because you're about to write reference answers.

> **No suitable corpus?** Acceptable substitutes: your company's public docs, a set of papers, government or regulatory PDFs, product manuals, a wiki export. You may also have Claude generate a synthetic corpus — but if you do, generate **long, messy, overlapping** documents, not tidy one-fact-per-paragraph text, or Task 5 will be meaninglessly easy. Say what you did at the top of your submission.

---

## The evaluation criteria

Six things you'll measure. Note the column on the right — **who measures it** matters as much as what it measures.

| Criterion | What it's asking | Measured by |
|---|---|---|
| **Hit-rate @ K** | Did retrieval surface the document that actually holds the answer? | **code** (vs. your evidence labels) |
| **Context relevance** | Are the retrieved chunks about the question at all? | LLM judge |
| **Faithfulness** | Is every claim in the answer supported by the retrieved chunks? | LLM judge |
| **Answer relevance** | Does the answer address the question that was asked? | LLM judge |
| **Correctness** | Does the answer match your reference answer? | LLM judge |
| **Refusal correctness** | Does it say "I don't know" exactly when it should? | **code** + judge |
| **Latency & cost** | Time per query, tokens per query | **code** |

Three things to notice:

- **The first two evaluate retrieval; the next three evaluate generation.** That split is the point of the whole assignment. When your score drops, you need to know which half broke.
- **Faithfulness and Correctness are different, and you need both.** An answer can be *correct* because the model already knew the fact from training, while your retriever fetched nothing useful. That answer is right and your system is broken — and only faithfulness plus hit-rate will reveal it. You are explicitly asked to hunt for one of these in Task 5.
- **Hit-rate is free.** No judge, no reference answer, one boolean per query. It's the cheapest useful number in RAG, which is why it's the metric you'll sweep parameters against in Task 6.

---

## Task 1 — Corpus in, baseline out

**Goal:** know exactly how the model behaves with **no** retrieval, so every later number has something to beat.

1. **Assemble the corpus** per the rules above. Commit it (or a manifest + download script if it's large).
2. **Run the naive baseline.** For every question in your eval set (Task 2 — write them first, then come back), ask the **generator model directly**, with no context and no documents. Same system prompt rules as Assignment 1: answer only if you know, otherwise say you can't.
3. **Record what happens.** For each question capture the answer, latency, and token counts. Then classify each response as:
   - **refused** — said it didn't know
   - **answered correctly** — got it right from training knowledge alone
   - **hallucinated** — answered confidently and wrongly

> **This step is not busywork, and it is not optional.** Without it, "RAG works!" is an unfalsifiable claim. It also tells you something specific about *your* corpus: if the naive model already answers 80% of your questions correctly, your questions are too generic — they're testing world knowledge, not your documents. Go write harder, more corpus-specific questions.

✅ **Done when:** every eval question has a no-RAG answer, classified into one of the three buckets, with latency and tokens recorded.

---

## Task 2 — Write the eval set *first*

**Goal:** fix the target before you start tuning, so you can't move the goalposts.

Build **25–40 questions**. For each one record:

| Field | What it holds |
|---|---|
| `question` | the question, phrased the way a real user would |
| `reference_answer` | the correct answer, written by you |
| `evidence_doc` | which document holds the answer |
| `evidence_page` | page or section (needed for hit-rate) |
| `answerable` | `True` / `False` |
| `difficulty` | `easy` / `hard` |

### How to generate them

**Most** of them can be **synthetic**: hand a chunk to Claude and ask it to write a question that chunk answers. Keep the chunk's document and page as the evidence label — that's your retrieval ground truth, for free.

### The part that matters more

**At least 6 questions must be hand-written and hard.** Specifically, include:

- **2 multi-hop** — the answer needs *two different* documents or sections (e.g. a comparison across two reports).
- **2 unanswerable** — the answer is genuinely **not in your corpus**. Mark `answerable = False`. A system that invents an answer here has failed, no matter how good it looks.
- **1 with negation** — where "X is not covered" vs. "X is covered" is the difference between right and disastrous.
- **1 that needs an exact identifier** — a code, SKU, section number, or name. Embeddings are weak at these; this question is here to expose that.

> **This is the single-chunk trap from class.** LLM-generated questions are made *from* one chunk, so they're answerable *by* one chunk — and your top-K retrieval will look brilliant. Real users ask questions that span documents, or that your corpus can't answer at all. If you only run synthetic questions you will score ~90% and learn nothing.
>
> **Report easy and hard as separate slices.** An average across both hides exactly the failures you're looking for (per-slice metrics, Assignment 2).

✅ **Done when:** 25–40 questions saved as CSV or JSON, with reference answers and evidence labels, including at least 6 hard ones covering all four categories above.

---

## Task 3 — Index your corpus

**Goal:** build the offline half of the pipeline, and see with your own eyes what your chunks look like.

Build a script `build_index.py`:

1. **Parse.** Load every document. Use `PyPDFLoader` for PDFs, plain loaders for text/Markdown.
2. **Chunk.** Baseline settings — do **not** tune these yet, you'll do that in Task 6:
   ```python
   RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
   ```
3. **Enrich.** Attach metadata to **every** chunk: `doc_name`, `page` (or `section`), plus anything useful in your domain (date, product, language). For extra credit, prepend the document title and section heading to the chunk text itself.
4. **Embed and store.** Embed every chunk and load them into a **vector database** — **FAISS** or **Chroma** locally is right for this scale (a hosted store like Pinecone or pgvector solves a problem you don't have yet). **Save the index to disk** so you're not re-embedding on every run.

Use **LangChain** to wire these four steps together — `DocumentLoader → TextSplitter → Embeddings → VectorStore → Retriever`. The point isn't the lines of code it saves; it's that every component has the same interface, so swapping the embedding model or the vector store in Task 6 is a one-line change instead of a rewrite.

### Then actually look at it

Print **10 randomly chosen chunks** and read them. Then run 3 questions through a bare `similarity_search` and print the chunks it returns.

Answer these in your write-up:

- Did the parser mangle anything? Tables turned into word soup? Headers spliced mid-sentence? **Hebrew or other RTL text reversed or broken?**
- Are chunks cut mid-sentence or mid-table? Does any chunk make sense on its own?
- For your 3 questions — did the right chunk come back? If not, can you tell why?

> **Do not skip the reading.** This is step 1 of the debugging playbook from class, and about half of you will discover here that your parser quietly destroyed a document. Finding that now costs ten minutes; finding it in Task 5 costs your afternoon.
>
> ⚠️ **No metadata now = no citations and no hit-rate later**, and retrofitting it means rebuilding the whole index.

✅ **Done when:** a saved index exists, every chunk carries `doc_name` + `page`, and you've written up what you saw in the 10 chunks and the 3 retrievals.

---

## Task 4 — The RAG pipeline

**Goal:** one function that turns a question into a grounded, cited answer.

Build `answer_with_rag(query, k=5)` that:

1. **Retrieves** the top-K chunks (baseline `k=5`).
2. **Builds the prompt** — chunks numbered `[1]`, `[2]`, … with **clear separators** between them and each chunk's **metadata included**.
3. **Generates** with a grounding system prompt that instructs the model to:
   - answer **only** from the provided context
   - **cite** the chunk number for every claim
   - reply with your fixed "I can't find that in the documents" sentence when the context doesn't contain the answer
4. **Returns structured output**, enforced with **Pydantic**:

```python
class GroundedAnswer(BaseModel):
    answer: str
    sources: list[str]        # doc_name + page for each chunk used
    evidence: list[str]       # the exact quoted lines it relied on
    answered: bool            # False when it refused
```

> **Separators are not cosmetic.** Concatenate five chunks without them and the model reads one continuous document — then cheerfully merges a fact from chunk 1 with a fact from chunk 4 into a single confident sentence that no source supports.

### Add one deterministic guard

Before you return, **verify the citations in code**: if the model cites `[7]` but you only supplied 5 chunks, that's a hallucination you can catch with an `if` statement and zero LLM calls. Log every occurrence. Report how often it happened.

### Sanity check

Run your **two unanswerable questions** through it. Does it refuse? If it answers, your grounding prompt is too weak — fix it now, before you evaluate 40 questions.

✅ **Done when:** `answer_with_rag()` returns a validated `GroundedAnswer` for any question, refuses on the unanswerable ones, and the citation check is wired in.

---

## Task 5 — Evaluate: RAG vs. baseline

**Goal:** produce the head-to-head table, then explain what it hides.

### 1. Implement the metrics

- **Hit-rate @ K** — in code. Did any retrieved chunk come from `evidence_doc` (and `evidence_page`)? One boolean per question. Skip the unanswerable ones.
- **Context relevance · Faithfulness · Answer relevance · Correctness** — four LLM judges, using the **Sonnet** judge and the same pattern as Assignment 2: rubric in the prompt, **Pydantic** schema, and `explanation` **before** `verdict` (you explained why in Assignment 2 — it still applies).
- **Refusal correctness** — did it refuse exactly on `answerable = False` and answer on the rest? Count both error directions separately: **false refusals** (said "I don't know" when the answer was right there) and **false answers** (answered when it should have refused).
- **Latency & tokens** — from your timers.

Think about what each judge needs to see. Faithfulness needs the **answer + the retrieved chunks** and nothing else. Correctness needs the **answer + your reference answer**. Context relevance needs the **question + the chunks** — it must not see the answer at all, or it will reason backwards from it.

### 2. Run the full comparison

Produce one table: **no-RAG baseline vs. RAG**, per metric, split into **easy** and **hard** slices.

### 3. Now the interesting part

Three things you are required to find and write up:

- **a. A question RAG made *worse*.** There will be one. Usually a general-knowledge question where retrieval injected irrelevant local context and dragged the model off a perfectly good answer. Show the baseline answer, the RAG answer, and the chunks that caused it. **This is a result, not a failure** — RAG is not a free upgrade.
- **b. A "right answer, broken pipeline" case.** Find a question where **correctness passed but hit-rate failed** — the model answered correctly while retrieval fetched nothing useful, because it already knew the fact. Explain why this is dangerous and why correctness alone would never have caught it.
- **c. Retrieval or generation?** Take your **5 worst** answers and classify each: was the right chunk missing (retrieval), or present-but-misused (generation)? Give the count. This number tells you where Task 6 should spend its effort.

> **Look at the retrieved chunks before you theorise.** Every diagnosis in (c) should be backed by the actual chunks, pasted into your write-up. "I think the embedding model struggled" is not an analysis; "here are the 5 chunks it returned, none from the right document" is.

✅ **Done when:** you have the baseline-vs-RAG table with easy/hard slices, and written answers to (a), (b), and (c) with evidence.

---

## Task 6 — Three improvement cycles

**Goal:** run the EDD loop on a five-knob system without fooling yourself.

Run **two** experiments. Pick from:

| Lever | What it tends to fix |
|---|---|
| **Chunk size / overlap** | chunks too small to hold an answer, or too big to match precisely |
| **top-K** | the answer isn't in the context at all (raise it) or is drowning in noise (lower it) |
| **Embedding model** | poor semantic matching overall; multilingual corpora |
| **Hybrid search** (dense + BM25) | exact identifiers, SKUs, codes, rare names |
| **Re-ranking** (cross-encoder over 30–50 candidates) | right chunk retrieved but ranked too low |
| **Generation prompt** | the model ignores context, or refuses when it shouldn't |

For each experiment record:

1. **The hypothesis, written before you run it** — tied to a failure you actually saw in Task 5. *"Hit-rate fails on my two identifier questions because embeddings smear exact codes; adding BM25 hybrid search should fix those two without hurting the rest."*
2. **The one thing you changed.**
3. **The result** — every metric, re-measured on the **same** eval set, as a delta from your Task 5 numbers.
4. **What you learned** — including when you were wrong. A refuted hypothesis that you explain well is worth more than a lucky improvement you can't account for.

> **Change one thing at a time.** Swap the embedding model *and* the chunk size together and you've learned nothing about either.
>
> **Watch out for noise.** On 30 questions, one question is 3.3 percentage points. A "+3%" improvement is one row changing its mind — it is not progress, and claiming it is will cost you more marks than reporting a flat result honestly. If you built bootstrap confidence intervals in Assignment 2, this is where they earn their keep.
>
> **Sweep against hit-rate.** It's free, needs no judge, and isolates retrieval — so you can try six chunk sizes for the cost of zero API calls before you spend money on the full judge run.

✅ **Done when:** three documented experiments, each with a pre-registered hypothesis, one changed variable, re-measured metrics, and an honest conclusion.

---

## Task 7 — Bonus (optional)

Pick one if you want to go further:

- **Multi-scale chunking.** Build two or three indices at different chunk sizes, retrieve from all of them, and merge. Is one chunk size actually right for *all* queries, or do short factual lookups and broad explanatory questions prefer different granularities? Show the per-question evidence.
- **Query rewriting.** Have an LLM rewrite each question three ways, retrieve for all three, and union the results. Measure the hit-rate change — especially on your hard questions.
- **Cross-check with Ragas.** Run `ragas` for faithfulness and context relevance and compare it to your hand-built judges. Where do they disagree, and which do you trust?
- **A second judge from a different family.** Re-judge 10 rows with an open model via Hugging Face or another provider. Does the verdict move? What does that say about your main judge?

---

## What to submit

1. **Your code**, committed and pushed to your course repo:
   - `build_index.py` (Task 3)
   - your RAG pipeline with `answer_with_rag()` (Task 4)
   - your evaluation code (Task 5) and experiment runner (Task 6)
2. **Your corpus** (or a manifest + fetch script) and your **eval set** as CSV/JSON.
3. **`assignment_03.xlsx`** — one row per question, containing:
   - `question`, `reference_answer`, `evidence_doc`, `evidence_page`, `answerable`, `difficulty`
   - the **no-RAG baseline** answer + its classification
   - the **RAG** answer, `sources`, `evidence`, `answered`
   - `hit@k`, plus the four judge verdicts and explanations
   - `latency_ms`, `input_tokens`, `output_tokens`
4. **A write-up** (markdown) containing:
   - **Task 1:** your corpus description, and the baseline breakdown (refused / correct / hallucinated)
   - **Task 3:** what you found reading your chunks — parsing damage, bad boundaries, the 3 test retrievals
   - **Task 5:** the baseline-vs-RAG table with easy/hard slices, plus (a) the question RAG made worse, (b) the right-answer-broken-pipeline case, and (c) your retrieval-vs-generation count for the 5 worst answers
   - **Task 6:** three experiments — hypothesis, change, deltas, conclusion
   - **One paragraph:** if you had one more week on this system, what would you fix first, and which number told you that?

---

## The takeaway

You'll finish with a Q&A tool that answers from fifty documents instead of one, and cites its sources. That's the visible result.

The durable one is the diagnostic habit. Anyone can wire up a LangChain tutorial in an afternoon — the engineer is the one who looks at a bad answer, prints the retrieved chunks *before* theorising, and says "the right paragraph was never retrieved, so no amount of prompt engineering will save this."

**Retrieval is the ceiling on everything above it.** And a correct answer is not proof that your pipeline works — it might just be a model that already knew, sitting on top of a retriever that fetched nothing. The only way to tell the difference is to measure the components separately, which is exactly what Task 5 made you do.
