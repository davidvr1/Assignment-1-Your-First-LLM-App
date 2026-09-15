"""
Task 6 - two improvement cycles, run cheaply against only the task-type
slices each hypothesis targets (not the full 25x5 matrix - see the
assignment's "sweep cheaply" note).

Experiment 1 (structural, not a model swap): SYSTEM_PROMPT_V2 in agent.py -
a hard 4-call cap on retrieve_policy_docs plus a mandatory verbatim refusal
sentence - re-run against unanswerable + multi_hop (the two slices where
Task 5 showed the failures this targets: m06 looping to a cap breach, u02
refusing in substance but not in a wording the refusal-check recognized).

Experiment 2 (model upgrade, run only after the structural fix, per the
assignment's ordering rule): claude-sonnet-5 instead of claude-haiku-4-5,
same v2 prompt, on the multi_hop slice only (the slice with sub-100%
success and the reasoning-shaped failures: m01, m05).

Usage:
    .venv/bin/python task6_experiments.py --exp 1
    .venv/bin/python task6_experiments.py --exp 2
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from agent import run_agent, SYSTEM_PROMPT, SYSTEM_PROMPT_V2, AGENT_MODEL, AGENT_MODEL_UPGRADE
from eval_runner import score_task, TASKS_PATH

TRACES_DIR = Path("traces_task6")
TRACES_DIR.mkdir(exist_ok=True)

EXP1_TYPES = {"unanswerable", "multi_hop"}
EXP2_TYPES = {"multi_hop"}


def run_condition(tasks, label, model, system_prompt, runs=5):
    rows = []
    for task in tasks:
        for run_idx in range(1, runs + 1):
            trace_path = TRACES_DIR / f"{task['id']}_{label}_run{run_idx}.jsonl"
            t0 = time.perf_counter()
            result = run_agent(
                task["task"], task_id=task["id"], run_idx=run_idx, trace_path=trace_path,
                use_guardrails=False, model=model, system_prompt=system_prompt,
            )
            success, refused = score_task(task, result["answer"], result["tool_calls"], result["terminal_state"])
            rows.append({
                "label": label, "task_id": task["id"], "type": task["type"], "run": run_idx,
                "success": success, "terminal_state": result["terminal_state"],
                "tool_calls": result["tool_calls"], "steps": result["steps"],
                "tokens": result["total_tokens"], "latency_ms": result["wall_ms"],
            })
            print(f"[{label} {task['id']} run{run_idx}] success={success} state={result['terminal_state']} "
                  f"tools={result['tool_calls']} tokens={result['total_tokens']}")
    return rows


def summarize(rows, label):
    df = pd.DataFrame(rows)
    df = df[df.label == label]
    if df.empty:
        return
    print(f"\n=== {label} ===")
    for ttype, g in df.groupby("type"):
        succ = g.success.sum()
        n = len(g)
        breaches = (g.terminal_state == "cap_breached").sum()
        print(f"  {ttype:12s} success={succ}/{n}  avg_tool_calls={g.tool_calls.mean():.2f}  "
              f"avg_tokens={g.tokens.mean():.0f}  p50={g.latency_ms.quantile(.5):.0f}ms  "
              f"p95={g.latency_ms.quantile(.95):.0f}ms  cap_breaches={breaches}")
    succ = df.success.sum(); n = len(df)
    print(f"  {'TOTAL':12s} success={succ}/{n}  avg_tool_calls={df.tool_calls.mean():.2f}  "
          f"avg_tokens={df.tokens.mean():.0f}  p50={df.latency_ms.quantile(.5):.0f}ms  "
          f"p95={df.latency_ms.quantile(.95):.0f}ms  cap_breaches={(df.terminal_state=='cap_breached').sum()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, choices=[1, 2], required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    all_tasks = json.loads(TASKS_PATH.read_text(encoding="utf-8"))

    if args.exp == 1:
        tasks = [t for t in all_tasks if t["type"] in EXP1_TYPES]
        rows = []
        rows += run_condition(tasks, "baseline_v1_haiku", AGENT_MODEL, SYSTEM_PROMPT, args.runs)
        rows += run_condition(tasks, "exp1_v2_haiku", AGENT_MODEL, SYSTEM_PROMPT_V2, args.runs)
        summarize(rows, "baseline_v1_haiku")
        summarize(rows, "exp1_v2_haiku")
        out = args.out or "task6_exp1_results.json"
    else:
        tasks = [t for t in all_tasks if t["type"] in EXP2_TYPES]
        rows = []
        prior = Path("task6_exp1_results.json")
        if prior.exists():
            prior_rows = json.loads(prior.read_text(encoding="utf-8"))
            rows += [r for r in prior_rows if r["label"] == "exp1_v2_haiku" and r["type"] in EXP2_TYPES]
            print(f"Reusing {len(rows)} exp1_v2_haiku rows from {prior} as the 'before' arm (no re-run).")
        else:
            rows += run_condition(tasks, "exp1_v2_haiku", AGENT_MODEL, SYSTEM_PROMPT_V2, args.runs)
        rows += run_condition(tasks, "exp2_v2_sonnet", AGENT_MODEL_UPGRADE, SYSTEM_PROMPT_V2, args.runs)
        summarize(rows, "exp1_v2_haiku")
        summarize(rows, "exp2_v2_sonnet")
        out = args.out or "task6_exp2_results.json"

    Path(out).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
