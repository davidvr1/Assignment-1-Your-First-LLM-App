"""
Task 5 - Agent vs. static RAG evaluation matrix.

For every task in agent_tasks.json, runs BOTH configs:
  - "rag":   the frozen Assignment 3 pipeline (rag_pipeline.answer_with_rag)
  - "agent": the LangGraph agent (agent.run_agent)
5 times each, records one row per (task, config, run) with all the columns
the assignment's submission format requires, and writes assignment_04.xlsx.

Usage:
    .venv/bin/python eval_runner.py --tasks 3 --runs 1          # cheap dry run
    .venv/bin/python eval_runner.py                              # full matrix
    .venv/bin/python eval_runner.py --config agent --runs 5
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd

from agent import run_agent
from rag_pipeline import _run_rag, load_vectorstore, REFUSAL_SENTENCE

TASKS_PATH = Path("agent_tasks.json")
TRACES_DIR = Path("traces")
OUT_XLSX = Path("assignment_04.xlsx")

RAG_INAPPLICABLE_TYPES = {"no_tool", "tool_fails"}


# --- code-checkable success scoring, keyed by task id (see agent_tasks.json
# success_criteria for the human-readable version of each of these) --------
def _num_in(answer: str, *needles: str) -> bool:
    normalized = answer.replace(",", "")
    return any(n.replace(",", "") in normalized for n in needles)


CHECKERS = {
    "s01": lambda a: _num_in(a, "45") and "₪" in a,
    "s02": lambda a: _num_in(a, "9") and "חודש" in a,
    "s03": lambda a: "MZ-2025" in a,
    "s04": lambda a: _num_in(a, "5,000", "5000"),
    "s05": lambda a: _num_in(a, "45") and "יום" in a,
    "s06": lambda a: _num_in(a, "68.90", "68.9"),
    "s07": lambda a: ("אין תקופת המתנה" in a) or ("אין" in a and "המתנה" in a),
    "s08": lambda a: _num_in(a, "500") and "₪" in a,
    "s09": lambda a: _num_in(a, "60") and "יום" in a,
    "s10": lambda a: _num_in(a, "35,000", "35000"),
    "s11": lambda a: _num_in(a, "45"),
    "m01": lambda a: _num_in(a, "180"),
    "m02": lambda a: _num_in(a, "1,000", "1000"),
    "m03": lambda a: ("מאוחדת" in a) and _num_in(a, "9"),
    "m04": lambda a: bool(re.search(r"2026-0[34]-\d{2}|2026\D0?[34]\D\d{1,2}|1\.4\.2026|31\.3\.2026|1 באפריל|31 במרץ", a)),
    "m05": lambda a: ("מכבי" in a) and _num_in(a, "300,000", "300000") and ("נלוו" in a or "משלים" in a or "ליווי" in a),
    "m06": lambda a: _num_in(a, "1,500", "1500"),
    "n01": None,  # scored on tool_calls == 0
    "n02": None,
    "n03": None,
    "u01": None,  # scored on refusal
    "u02": None,
    "u03": None,
    "tf01": None,  # scored on terminal_state / no fabrication
    "tf02": None,
}


def score_task(task: dict, answer: str, tool_calls: int, terminal_state: str) -> tuple[bool, bool]:
    """Returns (success, refused)."""
    ttype = task["type"]
    refused = terminal_state == "refused" or REFUSAL_SENTENCE in (answer or "") or bool(
        re.search(r"לא (מצאתי|ניתן|יכול)|מצטער", answer or "")
    )

    if ttype == "no_tool":
        return tool_calls == 0, refused
    if ttype == "unanswerable":
        return refused, refused
    if ttype == "tool_fails":
        # success = it did NOT fabricate an answer and DID report a failure/refusal,
        # i.e. terminal_state in {refused, error, cap_breached}
        success = terminal_state in ("refused", "error", "cap_breached")
        return success, refused

    checker = CHECKERS.get(task["id"])
    if checker is None:
        return False, refused
    return bool(checker(answer or "")), refused


def run_rag_config(task: dict, run_idx: int, vectorstore) -> dict:
    if task["type"] in RAG_INAPPLICABLE_TYPES:
        return {
            "answer": "n/a", "success": "n/a", "refused": "n/a", "terminal_state": "n/a",
            "tool_calls": "n/a", "tools_used": "n/a", "steps": "n/a",
            "latency_ms": None, "input_tokens": None, "output_tokens": None,
        }
    try:
        result = _run_rag(task["task"], k=5, vectorstore=vectorstore)
        answer = result["grounded_answer"].answer
        refused = not result["grounded_answer"].answered
        latency_ms = result["latency_ms"]
        input_tokens = result["input_tokens"]
        output_tokens = result["output_tokens"]
    except Exception as e:
        answer = f"ERROR: {e}"
        refused = False
        latency_ms = None
        input_tokens = output_tokens = None
    success, refused = score_task(task, answer, tool_calls=1, terminal_state="refused" if refused else "answered")
    return {
        "answer": answer, "success": success, "refused": refused,
        "terminal_state": "refused" if refused else "answered",
        "tool_calls": 1, "tools_used": "[retrieve_policy_docs]", "steps": 1,
        "latency_ms": latency_ms, "input_tokens": input_tokens, "output_tokens": output_tokens,
    }


def run_agent_config(task: dict, run_idx: int, use_guardrails: bool) -> dict:
    os.environ["BROKEN_TOOL"] = task.get("broken_tool", "") if task["type"] == "tool_fails" else ""

    trace_path = TRACES_DIR / f"{task['id']}_agent_run{run_idx}.jsonl"
    result = run_agent(task["task"], task_id=task["id"], run_idx=run_idx,
                        trace_path=trace_path, use_guardrails=use_guardrails)
    success, refused = score_task(task, result["answer"], result["tool_calls"], result["terminal_state"])
    return {
        "answer": result["answer"], "success": success, "refused": refused,
        "terminal_state": result["terminal_state"],
        "tool_calls": result["tool_calls"], "tools_used": json.dumps(result["tools_used"], ensure_ascii=False),
        "steps": result["steps"], "latency_ms": result["wall_ms"],
        "input_tokens": None, "output_tokens": result["total_tokens"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=None, help="limit to first N tasks (dry run)")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--config", choices=["rag", "agent", "both"], default="both")
    parser.add_argument("--guardrails", action="store_true")
    parser.add_argument("--out", default=str(OUT_XLSX))
    args = parser.parse_args()

    TRACES_DIR.mkdir(exist_ok=True)
    tasks = json.loads(TASKS_PATH.read_text(encoding="utf-8"))
    if args.tasks:
        tasks = tasks[: args.tasks]

    vectorstore = load_vectorstore() if args.config in ("rag", "both") else None

    rows = []
    for task in tasks:
        for run_idx in range(1, args.runs + 1):
            if args.config in ("rag", "both"):
                r = run_rag_config(task, run_idx, vectorstore)
                rows.append({"task_id": task["id"], "task": task["task"], "type": task["type"],
                             "answerable": task["answerable"], "success_criteria": task["success_criteria"],
                             "config": "rag", "run": run_idx, **r})
                print(f"[rag   {task['id']} run{run_idx}] success={r['success']} state={r['terminal_state']}")
            if args.config in ("agent", "both"):
                r = run_agent_config(task, run_idx, args.guardrails)
                rows.append({"task_id": task["id"], "task": task["task"], "type": task["type"],
                             "answerable": task["answerable"], "success_criteria": task["success_criteria"],
                             "config": "agent", "run": run_idx, **r})
                print(f"[agent {task['id']} run{run_idx}] success={r['success']} state={r['terminal_state']} "
                      f"tools={r['tool_calls']} tokens={r['output_tokens']}")

    df = pd.DataFrame(rows)
    df.to_excel(args.out, index=False)
    print(f"\nWrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
