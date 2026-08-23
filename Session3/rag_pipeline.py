"""
Task 4 - The RAG pipeline: one function that turns a question into a
grounded, cited answer.

Reads:  faiss_index/  (built by build_index.py)
Exposes: answer_with_rag(query, k=5) -> GroundedAnswer

Run directly to sanity-check on the two unanswerable questions from
eval_set.json, per the assignment's instruction to check this BEFORE
evaluating the full 30-question set in Task 5.
"""

import json
import os
import re
from pathlib import Path

from langchain_community.vectorstores import FAISS
from openai import OpenAI
from pydantic import BaseModel

from build_index import BgeQueryPrefixEmbeddings, EMBEDDING_MODEL, INDEX_DIR

GENERATOR_MODEL = "claude-haiku-4-5"
KEY_FILE = Path(__file__).parent / "key.txt"

# The exact refusal sentence the model must use verbatim when the retrieved
# context doesn't answer the question. Keeping this as one fixed string (not
# "say you don't know somehow") is what makes refusal-correctness checkable
# by exact string match in Task 5, instead of needing a judge for it.
REFUSAL_SENTENCE = "לא מצאתי את התשובה במסמכים שסופקו."


class GroundedAnswer(BaseModel):
    """Structured output contract for answer_with_rag().

    `answer` contains inline [n] citations pointing at the numbered chunks
    in the prompt (see build_prompt). `sources` and `evidence` let the caller
    show/verify provenance without re-parsing the answer text.
    """
    answer: str
    sources: list[str]   # doc_name + page, one per chunk actually relied on
    evidence: list[str]  # exact quoted lines from the chunks that support the answer
    answered: bool        # False when it refused


SYSTEM_PROMPT = f"""אתה עוזר שעונה על שאלות אודות תכניות ביטוח בריאות משלים, אך ורק
על סמך קטעי המידע ("Context") שסופקו לך למטה בהודעת המשתמש.

כללים מחייבים:
1. ענה אך ורק מתוך ה-Context שסופק. אל תשתמש בידע חיצוני ואל תנחש.
2. כל טענה עובדתית בתשובה שלך חייבת להיות מלווה בציטוט של מספר הקטע הרלוונטי
   בסוגריים מרובעים, למשל [1] או [2][3]. אל תצטט מספר קטע שלא סופק לך.
3. אם התשובה אינה מופיעה ב-Context, או שה-Context אינו מכיל מספיק מידע כדי
   לענות בביטחון, החזר answered=false ותן answer בדיוק המשפט הבא ואך ורק אותו:
   "{REFUSAL_SENTENCE}"
4. קרא לפונקציה submit_answer עם התשובה שלך. sources: רשימת doc_name + page
   עבור כל קטע שבו השתמשת בפועל. evidence: רשימת הציטוטים המדויקים (מילה
   במילה) מתוך הקטעים שעליהם התבססת. כאשר answered=false, sources ו-evidence
   יהיו רשימות ריקות.
"""

# Structured output via tool-calling (OpenAI-compatible function calling),
# NOT hand-written JSON in the model's text output. Task 4 asks for output
# "enforced with Pydantic" -- the first version of this pipeline had the
# model write raw ```json text and parsed it with json.loads(), which broke
# whenever the Hebrew answer contained an internal quote mark (e.g. the
# abbreviation שר"פ) that the model forgot to escape. Tool-calling sidesteps
# that: the API constructs valid JSON arguments for us, so this class of bug
# can't happen regardless of what characters appear in the answer text.
ANSWER_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_answer",
        "description": "Submit the grounded answer to the user's question.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "sources": {"type": "array", "items": {"type": "string"}},
                "evidence": {"type": "array", "items": {"type": "string"}},
                "answered": {"type": "boolean"},
            },
            "required": ["answer", "sources", "evidence", "answered"],
        },
    },
}

SEPARATOR = "\n" + ("-" * 40) + "\n"


def load_api_key() -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        return api_key
    if KEY_FILE.exists():
        match = re.search(r'ANTHROPIC_API_KEY="([^"]+)"', KEY_FILE.read_text())
        if match:
            return match.group(1)
    raise RuntimeError(
        f"ANTHROPIC_API_KEY not set and not found in {KEY_FILE}. "
        "Set the env var, or create key.txt with: ANTHROPIC_API_KEY=\"...\""
    )


def load_vectorstore() -> FAISS:
    embeddings = BgeQueryPrefixEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(
        INDEX_DIR, embeddings, allow_dangerous_deserialization=True
    )


def build_prompt(query: str, chunks: list) -> str:
    """Number the retrieved chunks [1], [2], ... with explicit separators and
    metadata, so the model can't accidentally merge facts from two different
    chunks into one uncited sentence (the assignment's "separators are not
    cosmetic" warning).
    """
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        doc_name = chunk.metadata.get("doc_name", "unknown")
        page = chunk.metadata.get("page", "n/a")
        blocks.append(
            f"[{i}] (מקור: {doc_name}, עמוד/סעיף: {page})\n{chunk.page_content}"
        )
    context = SEPARATOR.join(blocks)
    return f"Context:\n{context}\n\nשאלה: {query}"


def check_citations(answer_text: str, num_chunks: int) -> list[int]:
    """Deterministic guard (no LLM call): find every [n] citation in the
    answer text and flag any n that's outside the range of chunks we actually
    supplied. This catches a specific, cheap-to-detect hallucination -- the
    model inventing a source that was never in its context window.
    """
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer_text)}
    return sorted(n for n in cited if n < 1 or n > num_chunks)


def _run_rag(query: str, k: int, vectorstore: FAISS) -> dict:
    """Internal workhorse: does the actual retrieve+generate+validate, and
    returns everything (not just the GroundedAnswer) -- latency, token
    counts, and the retrieved chunks themselves. Task 5 needs all of this
    (hit-rate@k needs the chunks' doc_name/page; the cost table needs
    latency/tokens), but Task 4's public contract only promises a
    GroundedAnswer, so answer_with_rag() below trims this down.
    """
    import time

    start = time.perf_counter()

    chunks = vectorstore.similarity_search(query, k=k)
    prompt = build_prompt(query, chunks)

    client = OpenAI(
        base_url="https://api.anthropic.com/v1/",
        api_key=load_api_key(),
    )

    # Tool-calling forces the API itself to build valid JSON arguments, so a
    # retry loop here only needs to guard against transient/empty responses,
    # not against the model's own JSON-escaping mistakes.
    last_error = None
    for attempt in range(3):
        resp = client.chat.completions.create(
            model=GENERATOR_MODEL,
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
            parsed = json.loads(tool_calls[0].function.arguments)
            grounded_answer = GroundedAnswer.model_validate(parsed)
            break
        except Exception as e:
            last_error = e
    else:
        raise RuntimeError(f"RAG generation failed after 3 attempts: {last_error}")

    latency_ms = (time.perf_counter() - start) * 1000
    invalid_citations = check_citations(grounded_answer.answer, len(chunks))

    return {
        "grounded_answer": grounded_answer,
        "invalid_citations": invalid_citations,
        "chunks": chunks,
        "latency_ms": latency_ms,
        "input_tokens": resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
    }


def answer_with_rag(query: str, k: int = 5, vectorstore: FAISS | None = None) -> tuple[GroundedAnswer, list[int]]:
    """Retrieve top-k chunks, generate a grounded+cited answer, validate it.

    Returns (GroundedAnswer, invalid_citation_numbers). The second element is
    the output of the citation guard -- empty list means every [n] the model
    cited was actually among the k chunks supplied.
    """
    if vectorstore is None:
        vectorstore = load_vectorstore()
    result = _run_rag(query, k, vectorstore)
    return result["grounded_answer"], result["invalid_citations"]


def _sanity_check() -> None:
    """Run the two unanswerable questions from eval_set.json through the
    pipeline and confirm it refuses instead of inventing an answer, per the
    assignment's explicit instruction to do this BEFORE running Task 5.
    """
    with open("eval_set.json", "r", encoding="utf-8") as f:
        eval_set = json.load(f)
    unanswerable = [q for q in eval_set if not q["answerable"]]

    vectorstore = load_vectorstore()
    citation_violations_total = 0

    for item in unanswerable:
        answer, invalid_citations = answer_with_rag(
            item["question"], k=5, vectorstore=vectorstore
        )
        citation_violations_total += len(invalid_citations)
        status = "REFUSED (correct)" if not answer.answered else "ANSWERED (grounding too weak!)"
        print(f"\n[{item['id']}] {status}")
        print(f"  Q: {item['question']}")
        print(f"  A: {answer.answer}")
        if invalid_citations:
            print(f"  !! invalid citations (out of range): {invalid_citations}")

    print(f"\nTotal invalid-citation occurrences across sanity check: {citation_violations_total}")


if __name__ == "__main__":
    _sanity_check()
