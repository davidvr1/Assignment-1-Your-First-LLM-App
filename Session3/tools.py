"""
Task 2 - Agent tools.

Three tools, each returning an explanatory string on failure (never raises,
never returns None/"", per the assignment's failure contract):

1. retrieve_policy_docs  - Assignment 3's retriever, wrapped as a tool.
2. calculate              - arithmetic the retriever structurally cannot do
                             (e.g. "4 specialist visits x 45 NIS copay").
3. date_duration          - date/duration utility: waiting periods, notice
                             periods, days-until-renewal. Chosen over web
                             search because the corpus's numeric facts are
                             about time windows (waiting periods, appeal
                             deadlines, cancellation notice), so this is a
                             domain-relevant, network-free, fully
                             deterministic second non-retrieval tool -
                             no flaky external API in the eval loop.

Toggle BROKEN_TOOL (see below) to force one tool to fail on purpose, for the
two `tool_fails` tasks (Task 3.5).
"""

import ast
import operator
import os
from datetime import date, datetime

from langchain_core.tools import tool

from build_index import BgeQueryPrefixEmbeddings, EMBEDDING_MODEL, INDEX_DIR
from langchain_community.vectorstores import FAISS

# Set via env var by the eval/experiment runners for the tool_fails tasks.
# e.g. BROKEN_TOOL=calculate. Read dynamically (not cached at import time)
# because the runner sets this env var *after* tools.py is already imported.
def _broken_tool() -> str:
    return os.environ.get("BROKEN_TOOL", "")

_VECTORSTORE: FAISS | None = None


def _get_vectorstore() -> FAISS:
    global _VECTORSTORE
    if _VECTORSTORE is None:
        embeddings = BgeQueryPrefixEmbeddings(model_name=EMBEDDING_MODEL)
        _VECTORSTORE = FAISS.load_local(
            INDEX_DIR, embeddings, allow_dangerous_deserialization=True
        )
    return _VECTORSTORE


@tool
def retrieve_policy_docs(query: str, k: int = 5) -> str:
    """Search the Israeli supplementary health insurance (שב"ן) corpus: 6
    documents covering three insurers' takanonim (Kelalit Moshlam, Maccabi
    Zahav, Meuhedet Adif), a cross-insurer comparison FAQ, and a generic
    claims-filing guide. Covers: co-pays, coverage caps, waiting periods,
    exclusions, claims process, plan codes.

    Returns up to k passages, each tagged with its source document name and
    page/section number, so an answer can cite them.

    Does NOT cover: prices in ILS beyond what the takanonim state, live
    claim status, anything not written in these 6 documents, and arithmetic
    over numbers it returns (use `calculate` for that).

    Args:
        query: the Hebrew or English search query.
        k: number of passages to return (default 5, max 10).
    """
    if _broken_tool() == "retrieve_policy_docs":
        return "ERROR: the document index is unavailable (connection to the vector store timed out). This tool is unavailable right now; do not retry more than once."
    try:
        k = max(1, min(int(k), 10))
        vectorstore = _get_vectorstore()
        chunks = vectorstore.similarity_search(query, k=k)
        if not chunks:
            return f"NO_RESULTS: no passage matched '{query}'. Try a broader or differently-phrased query."
        blocks = []
        for i, chunk in enumerate(chunks, start=1):
            doc_name = chunk.metadata.get("doc_name", "unknown")
            page = chunk.metadata.get("page", "n/a")
            blocks.append(f"[{i}] (מקור: {doc_name}, עמוד/סעיף: {page})\n{chunk.page_content}")
        return "\n----\n".join(blocks)
    except Exception as e:
        return f"ERROR: retrieval failed ({e}). This tool is unavailable right now; do not retry more than once."


# A small, safe arithmetic-only evaluator: no name lookups, no calls, no
# attribute access - just numbers and +-*/%** and parentheses. Avoids the
# security risk of a bare eval() on model-generated text.
_ALLOWED_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("expression contains a disallowed element")


@tool
def calculate(expression: str) -> str:
    """Evaluate a pure arithmetic expression: +, -, *, /, %, **, parentheses,
    and numeric literals only. Use this for totals, percentages (e.g. VAT),
    multi-visit copay sums, or any math over numbers already retrieved from
    the documents (e.g. "45 * 4" for 4 specialist visits at a 45 NIS copay).

    Returns the numeric result as a string, or an ERROR string if the
    expression is invalid or unsafe.

    Does NOT cover: currency conversion, unit conversion, dates (use
    `date_duration`), or looking up any number itself - retrieve the number
    with `retrieve_policy_docs` first.

    Args:
        expression: a plain arithmetic expression, e.g. "45 * 4" or "4390 * 1.18".
    """
    if _broken_tool() == "calculate":
        return ("ERROR: the calculation service is unavailable (internal evaluator crashed). "
                "This tool is unavailable right now; do not retry more than once, and do NOT "
                "compute the result yourself instead, even if the arithmetic looks simple - "
                "report to the user that the calculation could not be completed.")
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return str(result)
    except Exception as e:
        return f"ERROR: could not evaluate '{expression}' ({e}). Provide a plain arithmetic expression using only numbers, +, -, *, /, %, **, and parentheses."


@tool
def date_duration(start_date: str, end_date: str | None = None, add_days: int | None = None) -> str:
    """Date/duration utility for waiting periods, notice periods and
    renewal windows. All dates are ISO format YYYY-MM-DD.

    Two modes (pass exactly one):
    - start_date + end_date -> returns the number of whole days between them.
    - start_date + add_days -> returns the resulting date after adding
      add_days days to start_date (add_days may be negative).

    "Today" (if a task needs it) is {today}.

    Does NOT cover: business-day/holiday-aware counting, timezones, or
    looking up how many days a waiting period *is* - retrieve that number
    with `retrieve_policy_docs` first, then pass it here as add_days.

    Args:
        start_date: ISO date YYYY-MM-DD.
        end_date: ISO date YYYY-MM-DD, for the "days between" mode.
        add_days: integer number of days to add to start_date, for the "resulting date" mode.
    """
    if _broken_tool() == "date_duration":
        return "ERROR: the date utility is unavailable (internal clock service crashed). This tool is unavailable right now; do not retry more than once."
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
    except Exception:
        return f"ERROR: start_date '{start_date}' is not a valid YYYY-MM-DD date."

    if end_date is not None and add_days is not None:
        return "ERROR: pass exactly one of end_date or add_days, not both."
    if end_date is not None:
        try:
            end = datetime.strptime(end_date, "%Y-%m-%d").date()
        except Exception:
            return f"ERROR: end_date '{end_date}' is not a valid YYYY-MM-DD date."
        return str((end - start).days)
    if add_days is not None:
        try:
            result = start + __import__("datetime").timedelta(days=int(add_days))
        except Exception as e:
            return f"ERROR: could not add {add_days} days to {start_date} ({e})."
        return result.isoformat()
    return "ERROR: pass either end_date or add_days."


# Fill in "today" dynamically in the docstring the model sees, without a
# runtime call - LangChain reads __doc__ at import time for the tool schema.
date_duration.description = date_duration.description.replace(
    "{today}", date.today().isoformat()
)

ALL_TOOLS = [retrieve_policy_docs, calculate, date_duration]
