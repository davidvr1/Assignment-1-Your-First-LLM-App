"""
Assignment 5, Task 5 - procedural memory with/without measurement.
Runs the 20-task set once each way (team config only - memory is a team
concept here) and reports the delta on success and tokens.
"""

import json

import pandas as pd

from eval_runner_team import score_code
from team import run_team

TASKS_PATH = "assignment5_tasks.json"
tasks = json.loads(open(TASKS_PATH, encoding="utf-8").read())

rows = []
for use_memory in (True, False):
    for task in tasks:
        trace_path = f"traces5_team/{task['id']}_team_mem{int(use_memory)}_ablation.jsonl"
        r = run_team(task["task"], task_id=task["id"], run_idx=1, trace_path=trace_path, use_memory=use_memory)
        success, refused = score_code(task, r["answer"], r["agent_turns"], r["terminal_state"])
        if success is None:
            success = False  # code-only pass for this ablation, no judge call - keeps cost down
        rows.append({
            "task_id": task["id"], "type": task["type"], "use_memory": use_memory,
            "success": success, "terminal_state": r["terminal_state"],
            "total_tokens": r["total_tokens"], "agent_turns": r["agent_turns"],
        })
        print(f"[mem={use_memory} {task['id']}] success={success} tokens={r['total_tokens']} state={r['terminal_state']}")

df = pd.DataFrame(rows)
df.to_csv("task5_memory_ablation.csv", index=False)

summary = df.groupby("use_memory").agg(success_rate=("success", "mean"), avg_tokens=("total_tokens", "mean"))
print("\n", summary)
summary.to_csv("task5_memory_ablation_summary.csv")
