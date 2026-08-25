"""
RAG playground web server.

A small FastAPI app that wraps the existing Session3 RAG pipeline
(build_index.py / rag_pipeline.py / hybrid_retriever.py) behind an HTTP API,
and serves a single-file React (hooks) UI from index.html.

Lets you, from the browser:
  - pick/activate an embedding model (rebuilds the FAISS index for that model,
    cached in memory so switching back is instant)
  - pick a retriever mode (dense-only, or hybrid dense+BM25 with an adjustable
    weight) and a top-k
  - pick which Claude model answers
  - ask a question and see the grounded answer, citations, evidence, latency
  - browse the vector table (every chunk's metadata + text + a small preview
    of its embedding vector) for whichever embedding model is active

Run:
    cd Session3/webapp
    ../.venv/bin/uvicorn server:app --reload --port 8008
Then open http://localhost:8008/
"""

import os
import re
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

SESSION_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SESSION_DIR))

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI

from build_index import (
    CORPUS_DIR,
    QUERY_INSTRUCTION,
    BgeQueryPrefixEmbeddings,
    chunk_documents,
    load_documents,
)
from hybrid_retriever import HybridRetriever
from rag_pipeline import (
    ANSWER_TOOL,
    SYSTEM_PROMPT,
    GroundedAnswer,
    build_prompt,
    check_citations,
    load_api_key,
)

app = FastAPI(title="RAG Playground")

# ---------------------------------------------------------------------------
# Available models the UI can choose between.
# ---------------------------------------------------------------------------
EMBEDDING_MODELS = [
    "BAAI/bge-small-en-v1.5",   # this session's assignment default (asymmetric query prefix)
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-base-en-v1.5",
]
DEFAULT_EMBEDDING_MODEL = EMBEDDING_MODELS[0]

LLM_MODELS = [
    "claude-haiku-4-5",
    "claude-sonnet-5",
    "claude-opus-5",
]
DEFAULT_LLM_MODEL = LLM_MODELS[0]

# ---------------------------------------------------------------------------
# State: parsed/chunked corpus is embedding-model independent, so it's parsed
# once. Per embedding model we lazily build+cache an in-memory FAISS index
# (and its HybridRetriever) so switching between already-activated models is
# instant and re-embedding the whole corpus only happens once per model.
# ---------------------------------------------------------------------------
_RAW_DOCS = None
_CHUNKS = None
_INDEX_CACHE: dict[str, dict] = {}  # embedding_model -> {"vectorstore", "hybrid"}
_ACTIVE_EMBEDDING_MODEL = None


def _get_chunks():
    global _RAW_DOCS, _CHUNKS
    if _CHUNKS is None:
        cwd = os.getcwd()
        os.chdir(SESSION_DIR)  # build_index.py's CORPUS_DIR is relative to Session3/
        try:
            _RAW_DOCS = load_documents()
        finally:
            os.chdir(cwd)
        _CHUNKS = chunk_documents(_RAW_DOCS)
    return _CHUNKS


def _build_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    if model_name == "BAAI/bge-small-en-v1.5":
        # Reuse the exact asymmetric-prefix behavior the assignment relies on.
        return BgeQueryPrefixEmbeddings(model_name=model_name)
    return HuggingFaceEmbeddings(model_name=model_name)


def activate_embedding_model(model_name: str) -> dict:
    """Build (or fetch from cache) the FAISS + hybrid retriever for a model."""
    global _ACTIVE_EMBEDDING_MODEL
    if model_name not in EMBEDDING_MODELS:
        raise HTTPException(400, f"Unknown embedding model: {model_name}")

    if model_name not in _INDEX_CACHE:
        chunks = _get_chunks()
        embeddings = _build_embeddings(model_name)
        t0 = time.perf_counter()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        build_ms = (time.perf_counter() - t0) * 1000
        hybrid = HybridRetriever(chunks, vectorstore)
        _INDEX_CACHE[model_name] = {
            "vectorstore": vectorstore,
            "hybrid": hybrid,
            "build_ms": build_ms,
            "num_chunks": len(chunks),
        }

    _ACTIVE_EMBEDDING_MODEL = model_name
    entry = _INDEX_CACHE[model_name]
    return {
        "embedding_model": model_name,
        "num_chunks": entry["num_chunks"],
        "build_ms": entry["build_ms"],
        "cached": True,
    }


# ---------------------------------------------------------------------------
# Weighted hybrid retrieval: HybridRetriever.similarity_search always does
# plain (unweighted) RRF. The UI wants an adjustable dense<->lexical weight,
# so we redo the RRF fusion here with a weight instead of importing a second
# copy of the fusion logic into hybrid_retriever.py.
# ---------------------------------------------------------------------------
def weighted_hybrid_search(hybrid: HybridRetriever, query: str, k: int, dense_weight: float) -> list:
    dense_hits = hybrid.vectorstore.similarity_search(query, k=hybrid.dense_k)
    bm25_scores = hybrid.bm25.get_scores(query.split())
    bm25_ranked_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[: hybrid.bm25_k]
    bm25_hits = [hybrid.chunks[i] for i in bm25_ranked_idx]

    def key(doc):
        return (doc.metadata.get("doc_name"), doc.metadata.get("chunk_id"))

    fused_scores: dict = {}
    chunk_by_key: dict = {}
    for rank, doc in enumerate(dense_hits):
        k_ = key(doc)
        chunk_by_key[k_] = doc
        fused_scores[k_] = fused_scores.get(k_, 0.0) + dense_weight * (1.0 / (hybrid.rrf_k + rank))
    for rank, doc in enumerate(bm25_hits):
        k_ = key(doc)
        chunk_by_key[k_] = doc
        fused_scores[k_] = fused_scores.get(k_, 0.0) + (1.0 - dense_weight) * (1.0 / (hybrid.rrf_k + rank))

    ranked_keys = sorted(fused_scores, key=lambda k_: fused_scores[k_], reverse=True)
    return [chunk_by_key[k_] for k_ in ranked_keys[:k]]


# ---------------------------------------------------------------------------
# API schemas
# ---------------------------------------------------------------------------
class ActivateRequest(BaseModel):
    embedding_model: str


class AskRequest(BaseModel):
    query: str
    k: int = 5
    retriever_mode: str = "dense"   # "dense" | "hybrid"
    dense_weight: float = 0.5       # only used when retriever_mode == "hybrid"
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    llm_model: str = DEFAULT_LLM_MODEL


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/config")
def get_config():
    return {
        "embedding_models": EMBEDDING_MODELS,
        "llm_models": LLM_MODELS,
        "active_embedding_model": _ACTIVE_EMBEDDING_MODEL,
        "activated_models": list(_INDEX_CACHE.keys()),
        "default_embedding_model": DEFAULT_EMBEDDING_MODEL,
        "default_llm_model": DEFAULT_LLM_MODEL,
    }


@app.post("/api/activate-embedding")
def activate_embedding(req: ActivateRequest):
    return activate_embedding_model(req.embedding_model)


@app.get("/api/vectors")
def get_vectors(embedding_model: str | None = None, limit: int = 200, offset: int = 0, preview_dims: int = 8):
    model_name = embedding_model or _ACTIVE_EMBEDDING_MODEL
    if model_name is None or model_name not in _INDEX_CACHE:
        raise HTTPException(400, "That embedding model hasn't been activated yet. POST /api/activate-embedding first.")

    entry = _INDEX_CACHE[model_name]
    vectorstore = entry["vectorstore"]
    index = vectorstore.index
    docstore = vectorstore.docstore
    id_map = vectorstore.index_to_docstore_id

    total = index.ntotal
    rows = []
    end = min(offset + limit, total)
    for i in range(offset, end):
        doc_id = id_map[i]
        doc = docstore.search(doc_id)
        vec = index.reconstruct(i).tolist()
        rows.append({
            "row": i,
            "doc_name": doc.metadata.get("doc_name"),
            "page": doc.metadata.get("page"),
            "chunk_id": doc.metadata.get("chunk_id"),
            "text": doc.page_content,
            "vector_preview": vec[:preview_dims],
            "vector_dims": len(vec),
        })

    return {
        "embedding_model": model_name,
        "total": total,
        "offset": offset,
        "limit": limit,
        "rows": rows,
    }


@app.post("/api/ask")
def ask(req: AskRequest):
    if req.embedding_model not in _INDEX_CACHE:
        activate_embedding_model(req.embedding_model)
    if req.llm_model not in LLM_MODELS:
        raise HTTPException(400, f"Unknown LLM model: {req.llm_model}")
    if req.retriever_mode not in ("dense", "hybrid"):
        raise HTTPException(400, "retriever_mode must be 'dense' or 'hybrid'")
    if not req.query.strip():
        raise HTTPException(400, "query must not be empty")

    entry = _INDEX_CACHE[req.embedding_model]

    t_retrieve = time.perf_counter()
    if req.retriever_mode == "dense":
        chunks = entry["vectorstore"].similarity_search(req.query, k=req.k)
    else:
        chunks = weighted_hybrid_search(entry["hybrid"], req.query, req.k, req.dense_weight)
    retrieval_ms = (time.perf_counter() - t_retrieve) * 1000

    prompt = build_prompt(req.query, chunks)
    client = OpenAI(base_url="https://api.anthropic.com/v1/", api_key=load_api_key())

    t_gen = time.perf_counter()
    last_error = None
    grounded_answer = None
    resp = None
    for attempt in range(3):
        resp = client.chat.completions.create(
            model=req.llm_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            tools=[ANSWER_TOOL],
            tool_choice={"type": "function", "function": {"name": "submit_answer"}},
            temperature=0 if attempt == 0 else 0.3,
        )
        tool_calls = resp.choices[0].message.tool_calls
        try:
            if not tool_calls:
                raise RuntimeError(f"no tool call, finish_reason={resp.choices[0].finish_reason}")
            import json
            parsed = json.loads(tool_calls[0].function.arguments)
            grounded_answer = GroundedAnswer.model_validate(parsed)
            break
        except Exception as e:
            last_error = e
    if grounded_answer is None:
        raise HTTPException(500, f"Generation failed after 3 attempts: {last_error}")
    generation_ms = (time.perf_counter() - t_gen) * 1000

    invalid_citations = check_citations(grounded_answer.answer, len(chunks))

    return {
        "answer": grounded_answer.answer,
        "answered": grounded_answer.answered,
        "sources": grounded_answer.sources,
        "evidence": grounded_answer.evidence,
        "invalid_citations": invalid_citations,
        "retrieval_ms": retrieval_ms,
        "generation_ms": generation_ms,
        "input_tokens": resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
        "retrieved_chunks": [
            {
                "doc_name": c.metadata.get("doc_name"),
                "page": c.metadata.get("page"),
                "chunk_id": c.metadata.get("chunk_id"),
                "text": c.page_content,
            }
            for c in chunks
        ],
        "params": req.model_dump(),
    }


# Serve the single-file React UI.
STATIC_DIR = Path(__file__).parent


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
