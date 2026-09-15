"""One-off: re-run only the rows that hit the orchestrator parse-miss
(u01/u02, team config, all 3 runs) after the retry fix in team.py, and
merge the results into assignment_05.xlsx in place."""

import json

import pandas as pd

from eval_runner_team import run_team_config

TASKS_PATH = "assignment5_tasks.json"
XLSX_PATH = "assignment_05.xlsx"

tasks = {t["id"]: t for t in json.loads(open(TASKS_PATH, encoding="utf-8").read())}
df = pd.read_excel(XLSX_PATH)

is_broken = (df.task_id.isin(["u01", "u02"])) & (df.config == "team") & (df.terminal_state == "error")
broken = df[is_broken]
print(f"patching {len(broken)} rows")

new_rows = []
for _, row in broken.iterrows():
    task = tasks[row.task_id]
    r = run_team_config(task, int(row.run))
    base = {
        "task_id": task["id"], "task": task["task"], "type": task["type"],
        "answerable": task["answerable"], "success_criteria": task["success_criteria"],
        "capable_agents": json.dumps(task["capable_agents"], ensure_ascii=False),
    }
    new_rows.append({**base, **r})
    print(f"[team {task['id']} run{row.run}] success={r['success']} state={r['terminal_state']}")

df = pd.concat([df[~is_broken], pd.DataFrame(new_rows)], ignore_index=True)
df.to_excel(XLSX_PATH, index=False)
print("done")
