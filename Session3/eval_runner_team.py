"""
Assignment 5, Task 6 - single agent (frozen Assignment 4, commit 515034d)
vs. the multi-agent team, on assignment5_tasks.json, N runs each.

Usage:
    .venv/bin/python eval_runner_team.py --tasks 3 --runs 1          # dry run
    .venv/bin/python eval_runner_team.py                              # full matrix (reduced: 3 runs)
    .venv/bin/python eval_runner_team.py --config team --runs 3
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from agent import run_agent, SYSTEM_PROMPT_V2
from rag_pipeline import REFUSAL_SENTENCE
from team import run_team
from team_judges import judge_team_run

TASKS_PATH = Path("assignment5_tasks.json")
TRACES_SINGLE = Path("traces5_single")
TRACES_TEAM = Path("traces5_team")
OUT_XLSX = Path("assignment_05.xlsx")


def _num_in(answer: str, *needles: str) -> bool:
    normalized = (answer or "").replace(",", "")
    return any(n.replace(",", "") in normalized for n in needles)


def _is_hebrew(text: str) -> bool:
    return bool(re.search(r"[֐-׿]", text or ""))


def _word_count(text: str) -> int:
    return len((text or "").split())


CODE_CHECKERS = {
    "c01": lambda a: _num_in(a, "1,500", "1500"),
    "c02": lambda a: bool(re.search(r"2026-03-11|11\.3\.2026|11 ב?מרץ", a)),
    "c03": lambda a: _num_in(a, "180"),
    "c04": lambda a: _num_in(a, "1,000", "1000"),
    "c05": lambda a: bool(re.search(r"2026-03-18|18\.3\.2026|18 ב?מרץ", a)),
    "c06": lambda a: ("מאוחדת" in a) and _num_in(a, "1,000", "1000"),
    "c07": lambda a: bool(re.search(r"2026-09-28|28\.9\.2026|28 בספטמבר", a)),
    "c08": lambda a: _num_in(a, "826.8", "826.80"),
    "b02": lambda a: _num_in(a, "45") and "יום" in a,
    "b04": lambda a: _num_in(a, "68.90", "68.9"),
    "s01": lambda a: _num_in(a, "45") and "₪" in a,
}

# task types scored structurally rather than by answer content
STRUCTURAL_TYPES = {"no_tool", "unanswerable"}

HANDOFF_CHECKS = {
    "h01": lambda a: _is_hebrew(a) and _word_count(a) <= 50,
    "h02": lambda a: _num_in(a, "500") and not _is_hebrew(a) and a.strip().count(".") <= 2,
}


def _refused(answer: str, terminal_state: str) -> bool:
    return (
        terminal_state in ("refused",)
        or REFUSAL_SENTENCE in (answer or "")
        or bool(re.search(r"לא (מצאתי|ניתן|יכול)|מצטער", answer or ""))
    )


def score_code(task: dict, answer: str, worker_turns: int, terminal_state: str) -> tuple:
    """Returns (success_or_None, refused). success is None when a judge is needed."""
    ttype = task["type"]
    refused = _refused(answer, terminal_state)

    if ttype == "no_tool":
        return worker_turns == 0, refused
    if ttype == "unanswerable":
        return refused, refused

    checker = CODE_CHECKERS.get(task["id"])
    if checker is not None:
        return bool(checker(answer or "")), refused

    return None, refused  # fall back to judge (task_success from judge_team_run / a similar single-config judge)


def run_single_config(task: dict, run_idx: int) -> dict:
    trace_path = TRACES_SINGLE / f"{task['id']}_single_run{run_idx}.jsonl"
    result = run_agent(task["task"], task_id=task["id"], run_idx=run_idx, trace_path=trace_path,
                        use_guardrails=True, system_prompt=SYSTEM_PROMPT_V2)
    success, refused = score_code(task, result["answer"], 0 if result["tool_calls"] == 0 else 1, result["terminal_state"])
    routing_correct = "n/a"  # no routing concept for a single agent
    handoff_correct = "n/a"
    if task["type"] == "handoff_stress":
        check = HANDOFF_CHECKS.get(task["id"])
        handoff_correct = bool(check(result["answer"])) if check else "n/a"
        if success is None:
            success = handoff_correct

    if success is None:
        jv = judge_team_run(task["task"], task["success_criteria"], ["agent"],
                             {"agent_output": result["answer"]}, result["answer"], result["terminal_state"])
        success = jv.task_success
        faithfulness, faith_expl = jv.faithfulness, jv.faithfulness_explanation
    else:
        jv = judge_team_run(task["task"], task["success_criteria"], ["agent"],
                             {"agent_output": result["answer"]}, result["answer"], result["terminal_state"])
        faithfulness, faith_expl = jv.faithfulness, jv.faithfulness_explanation

    return {
        "config": "single", "run": run_idx, "answer": result["answer"], "success": success,
        "refused": refused, "terminal_state": result["terminal_state"],
        "route": json.dumps(["agent"]), "agent_turns": 1 if result["tool_calls"] else 0,
        "tool_calls": result["tool_calls"], "routing_correct": routing_correct,
        "handoff_correct": handoff_correct, "per_agent_success": "n/a",
        "faithfulness": faithfulness, "faithfulness_explanation": faith_expl,
        "latency_ms": result["wall_ms"], "input_tokens": result.get("total_input_tokens"),
        "output_tokens": result.get("total_output_tokens"), "total_tokens": result.get("total_tokens"),
        "breach_reason": (
            result["terminal_state"] if result["terminal_state"] == "cap_breached" else None
        ),
    }


def run_team_config(task: dict, run_idx: int) -> dict:
    trace_path = TRACES_TEAM / f"{task['id']}_team_run{run_idx}.jsonl"
    result = run_team(task["task"], task_id=task["id"], run_idx=run_idx, trace_path=trace_path)
    worker_turns = result["agent_turns"]
    success, refused = score_code(task, result["answer"], worker_turns, result["terminal_state"])

    routing_correct = "n/a"
    if task["type"] != "no_tool":
        routing_correct = bool(result["route"]) and result["route"][0] in task["capable_agents"]
    else:
        routing_correct = worker_turns == 0

    handoff_correct = "n/a"
    if task["type"] == "handoff_stress":
        check = HANDOFF_CHECKS.get(task["id"])
        handoff_correct = bool(check(result["answer"])) if check else "n/a"

    jv = judge_team_run(task["task"], task["success_criteria"], result["route"], result["facts"],
                         result["answer"], result["terminal_state"])
    if success is None:
        success = jv.task_success
    if task["type"] == "handoff_stress" and handoff_correct != "n/a":
        success = success and handoff_correct

    per_agent_success = {a: v.success for a, v in jv.per_agent.items()}

    return {
        "config": "team", "run": run_idx, "answer": result["answer"], "success": success,
        "refused": refused, "terminal_state": result["terminal_state"],
        "route": json.dumps(result["route"], ensure_ascii=False), "agent_turns": worker_turns,
        "tool_calls": result["tool_calls"], "routing_correct": routing_correct,
        "handoff_correct": handoff_correct, "per_agent_success": json.dumps(per_agent_success, ensure_ascii=False),
        "faithfulness": jv.faithfulness, "faithfulness_explanation": jv.faithfulness_explanation,
        "latency_ms": result["wall_ms"], "input_tokens": None,
        "output_tokens": None, "total_tokens": result["total_tokens"], "breach_reason": result["breach_reason"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=None, help="limit to first N tasks (dry run)")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--config", choices=["single", "team", "both"], default="both")
    parser.add_argument("--out", default=str(OUT_XLSX))
    args = parser.parse_args()

    TRACES_SINGLE.mkdir(exist_ok=True)
    TRACES_TEAM.mkdir(exist_ok=True)
    tasks = json.loads(TASKS_PATH.read_text(encoding="utf-8"))
    if args.tasks:
        tasks = tasks[: args.tasks]

    rows = []
    for task in tasks:
        for run_idx in range(1, args.runs + 1):
            base = {
                "task_id": task["id"], "task": task["task"], "type": task["type"],
                "answerable": task["answerable"], "success_criteria": task["success_criteria"],
                "capable_agents": json.dumps(task["capable_agents"], ensure_ascii=False),
            }
            if args.config in ("single", "both"):
                r = run_single_config(task, run_idx)
                rows.append({**base, **r})
                print(f"[single {task['id']} run{run_idx}] success={r['success']} state={r['terminal_state']}")
            if args.config in ("team", "both"):
                r = run_team_config(task, run_idx)
                rows.append({**base, **r})
                print(f"[team   {task['id']} run{run_idx}] success={r['success']} state={r['terminal_state']} "
                      f"route={r['route']} turns={r['agent_turns']}")

    df = pd.DataFrame(rows)
    df.to_excel(args.out, index=False)
    print(f"\nWrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
