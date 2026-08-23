"""
Task 3 - Build the offline half of the RAG pipeline: parse -> chunk -> enrich
-> embed -> store.

Reads:  corpus/*.pdf, corpus/*.md, corpus/*.txt
Writes: faiss_index/  (FAISS vector store, saved to disk so later scripts
        don't have to re-embed the corpus every run)

This script also does the "look at it" step the assignment insists on: it
prints 10 random chunks and runs 3 canned questions through a bare
similarity_search, so you can eyeball whether parsing/chunking/retrieval look
sane BEFORE building the full generation pipeline on top of them (Task 4).
"""

import random
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

CORPUS_DIR = Path("corpus")
INDEX_DIR = "faiss_index"

# BAAI/bge-small-en-v1.5 was trained with an ASYMMETRIC convention: queries at
# search time need a fixed instruction prefix, but the passages you index do
# NOT get that prefix. This is exactly the "read the model card" trap the
# assignment calls out -- get it backwards (or symmetric) and nothing crashes,
# retrieval just gets quietly worse. HuggingFaceEmbeddings only ever applies
# query_instruction to embed_query(), never to embed_documents(), so as long
# as we use those two methods correctly (embed_query for the user's question,
# embed_documents for the corpus) the asymmetry is handled automatically.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class BgeQueryPrefixEmbeddings(HuggingFaceEmbeddings):
    """HuggingFaceEmbeddings, but prepends QUERY_INSTRUCTION only for queries.

    The installed langchain-huggingface version dropped the old
    `query_instruction` constructor kwarg, so we get the same asymmetric
    behavior by overriding embed_query() directly: embed_documents() (used at
    indexing time) is untouched, embed_query() (used at search time) gets the
    prefix. This is the same requirement bge-small-en-v1.5's model card
    describes -- just implemented at this call-site instead of via a kwarg.
    """

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(QUERY_INSTRUCTION + text)

# Baseline chunking settings from the assignment. Do NOT tune these here --
# Task 6 is where chunk_size/overlap become an experiment variable. Changing
# them now would make the Task 3 "look at your chunks" exercise describe a
# different pipeline than the one Task 5's numbers are measured against.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

TEST_QUERIES = [
    "מה תקרת הכיסוי להשתלת איבר בחו\"ל בתכנית כללית מושלם?",
    "מהי תקופת האכשרה לניתוח פרטי בישראל במאוחדת עדיף?",
    "האם אורתודנטיה לילדים מכוסה במכבי זהב?",
]


def load_documents():
    """Parse every file in corpus/ into LangChain Documents.

    PyPDFLoader gives one Document per PDF page (so 'page' metadata comes
    for free). For .md/.txt we don't have pages, so we use the filename
    itself as the 'section' identifier -- chunk-level section metadata gets
    added later, once we know which chunk came from which part of the file.
    """
    docs = []
    for path in sorted(CORPUS_DIR.iterdir()):
        if path.suffix == ".pdf":
            # Each returned Document already has metadata={"source": ..., "page": N}
            loaded = PyPDFLoader(str(path)).load()
        elif path.suffix in (".md", ".txt"):
            loaded = TextLoader(str(path), encoding="utf-8").load()
        else:
            continue

        for doc in loaded:
            # Normalize metadata across loaders so every downstream chunk has
            # the same two keys, regardless of which loader produced it.
            doc.metadata["doc_name"] = path.name
            doc.metadata["page"] = doc.metadata.get("page", "n/a")
        docs.extend(loaded)

    return docs


def chunk_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_title_prefix=True):
    """Split each Document into chunks, size/overlap/prefix all overridable.

    RecursiveCharacterTextSplitter tries to break on paragraph/sentence
    boundaries before falling back to a hard character cut, which is why it's
    the default choice over a naive fixed-width splitter -- it still WILL cut
    mid-sentence sometimes on long unstructured text, which is exactly what
    the Task 3 "read your chunks" step below is meant to catch.

    Task 6, experiment 1's hypothesis is that `add_title_prefix=True` (the
    Task 3 default) is itself the retrieval bug: every chunk's title/intro
    line looks similar across documents, so it's the one parameter this
    function was built to switch off without touching Task 3's frozen
    defaults (build_index.py's own CHUNK_SIZE/CHUNK_OVERLAP module constants
    are untouched by callers who don't pass them).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)

    # Enrich: every chunk must carry doc_name + page so that later (a) we can
    # print citations, and (b) hit-rate@K can check "did the retrieved chunk
    # come from evidence_doc/evidence_page" in code, no LLM needed.
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        if add_title_prefix:
            # Extra-credit enrichment: prepend doc name to the chunk TEXT
            # itself (not just metadata), so a bare similarity_search over
            # the chunk content alone still carries a hint of which document
            # it's from. Task 5 found this backfires (see Task 6 exp. 1).
            chunk.page_content = f"[{chunk.metadata['doc_name']}] {chunk.page_content}"

    return chunks


def build_and_save_index(chunks, embeddings=None):
    """Embed every chunk and persist a FAISS index to disk.

    Saving to disk matters because embedding is the slow, repeatable part of
    indexing -- Task 4/5/6 scripts should FAISS.load_local() this instead of
    re-embedding the whole corpus on every run. `embeddings` is overridable so
    a sweep script (Task 6) can reuse one already-loaded model across many
    in-memory candidate indices instead of reloading it per config.
    """
    if embeddings is None:
        embeddings = BgeQueryPrefixEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_DIR)
    return vectorstore


def main() -> None:
    print("Parsing corpus...")
    docs = load_documents()
    print(f"  loaded {len(docs)} raw documents/pages")

    print("Chunking...")
    chunks = chunk_documents(docs)
    print(f"  produced {len(chunks)} chunks "
          f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    print("Embedding + building FAISS index (this downloads the model on first run)...")
    vectorstore = build_and_save_index(chunks)
    print(f"  saved index to {INDEX_DIR}/")

    # --- "Then actually look at it" step, required by the assignment ---
    print("\n" + "=" * 70)
    print("10 RANDOM CHUNKS (read these for parsing/chunking damage)")
    print("=" * 70)
    for chunk in random.sample(chunks, min(10, len(chunks))):
        print(f"\n--- {chunk.metadata['doc_name']} "
              f"(page={chunk.metadata['page']}, chunk_id={chunk.metadata['chunk_id']}) ---")
        print(chunk.page_content[:400])

    print("\n" + "=" * 70)
    print("3 TEST RETRIEVALS (bare similarity_search, k=3)")
    print("=" * 70)
    for query in TEST_QUERIES:
        print(f"\nQuery: {query}")
        results = vectorstore.similarity_search(query, k=3)
        for rank, chunk in enumerate(results, start=1):
            print(f"  [{rank}] {chunk.metadata['doc_name']} "
                  f"(page={chunk.metadata['page']}): {chunk.page_content[:200]}")


if __name__ == "__main__":
    main()
