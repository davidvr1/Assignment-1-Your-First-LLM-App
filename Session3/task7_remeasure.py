"""Assignment 5, Task 7 - re-measure b01 (hallucinated-handoff mitigation,
rule 2 tightened) and c06 (duplicated-work mitigation, rule 7 added) after
the team.py prompt fixes, 3 runs each, team config only."""

import json

import pandas as pd

from eval_runner_team import run_team_config

tasks = {t["id"]: t for t in json.loads(open("assignment5_tasks.json", encoding="utf-8").read())}

rows = []
for task_id in ("b01", "c06"):
    task = tasks[task_id]
    for run_idx in range(1, 4):
        r = run_team_config(task, run_idx)
        rows.append({"task_id": task_id, "run": run_idx, **r})
        print(f"[after-fix {task_id} run{run_idx}] success={r['success']} state={r['terminal_state']} "
              f"route={r['route']} tokens={r['total_tokens'] if 'total_tokens' in r else r.get('output_tokens')}")

pd.DataFrame(rows).to_csv("task7_remeasure.csv", index=False)
print("done")
