"""Re-score the 6 single-config handoff_stress rows (h01/h02) using the
already-collected answers (no need to re-run the agent) with the fixed,
judge+format-symmetric success formula in eval_runner_team.py."""

import json

import pandas as pd

from eval_runner_team import HANDOFF_CHECKS
from team_judges import judge_team_run

df = pd.read_excel("assignment_05.xlsx")
for col in ("success", "handoff_correct", "faithfulness", "faithfulness_explanation"):
    df[col] = df[col].astype(object)
target = (df.task_id.isin(["h01", "h02"])) & (df.config == "single")

for idx in df[target].index:
    row = df.loc[idx]
    check = HANDOFF_CHECKS.get(row.task_id)
    handoff_correct = bool(check(row.answer)) if check else True
    jv = judge_team_run(row.task, row.success_criteria, ["agent"], {"agent_output": row.answer},
                         row.answer, row.terminal_state)
    new_success = bool(jv.task_success) and handoff_correct
    df.loc[idx, "success"] = new_success
    df.loc[idx, "handoff_correct"] = handoff_correct
    df.loc[idx, "faithfulness"] = jv.faithfulness
    df.loc[idx, "faithfulness_explanation"] = jv.faithfulness_explanation
    print(f"[single {row.task_id} run{row.run}] old_success={row.success} -> new_success={new_success} "
          f"(judge_task_success={jv.task_success}, format_ok={handoff_correct})")

df.to_excel("assignment_05.xlsx", index=False)
print("done")
