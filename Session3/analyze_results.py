"""Assignment 5, Task 6 - builds Table 1 (head-to-head, sliced by type) and
Table 2 (per-agent attribution) from assignment_05.xlsx and traces5_*/."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

df = pd.read_excel("assignment_05.xlsx")


def p50(s):
    return np.percentile(s.dropna(), 50) if len(s.dropna()) else float("nan")


def p95(s):
    return np.percentile(s.dropna(), 95) if len(s.dropna()) else float("nan")


# ---- Table 1: head-to-head, sliced by type ----
rows = []
for ttype, g in df.groupby("type"):
    for config, gc in g.groupby("config"):
        n_tasks = gc.task_id.nunique()
        n_runs = len(gc)
        success_rate = gc.success.mean()
        rows.append({
            "type": ttype, "config": config, "tasks": n_tasks, "runs": n_runs,
            "success": f"{gc.success.sum():.0f}/{n_runs} ({success_rate:.0%})",
            "agent_turns_avg": round(gc.agent_turns.mean(), 2),
            "tool_calls_avg": round(gc.tool_calls.mean(), 2),
            "p50_latency_ms": round(p50(gc.latency_ms), 0),
            "p95_latency_ms": round(p95(gc.latency_ms), 0),
            "cap_breaches": (gc.terminal_state == "cap_breached").sum(),
            "loops": (gc.terminal_state == "loop_detected").sum(),
        })
table1 = pd.DataFrame(rows).sort_values(["type", "config"])
table1.to_csv("task6_table1_head_to_head.csv", index=False)
print("=== TABLE 1 ===")
print(table1.to_string(index=False))

# ---- Table 2: per-agent attribution (team config only) ----
team = df[df.config == "team"].copy()
agent_stats = {}
for _, row in team.iterrows():
    route = json.loads(row.route) if isinstance(row.route, str) else row.route
    per_agent = json.loads(row.per_agent_success) if isinstance(row.per_agent_success, str) and row.per_agent_success not in ("n/a",) else {}
    for agent in route:
        s = agent_stats.setdefault(agent, {"turns": 0, "success_votes": [], "failures": []})
        s["turns"] += 1
        if agent in per_agent:
            s["success_votes"].append(bool(per_agent[agent]))
            if not per_agent[agent]:
                s["failures"].append(row.task_id)

rows2 = []
for agent, s in agent_stats.items():
    rate = (sum(s["success_votes"]) / len(s["success_votes"])) if s["success_votes"] else float("nan")
    rows2.append({
        "agent": agent, "turns_taken": s["turns"],
        "per_agent_success_rate": f"{rate:.0%}" if rate == rate else "n/a",
        "failures_attributed": ", ".join(sorted(set(s["failures"]))) or "(none)",
    })
table2 = pd.DataFrame(rows2)
table2.to_csv("task6_table2_per_agent.csv", index=False)
print("\n=== TABLE 2 ===")
print(table2.to_string(index=False))

# ---- (f) cost of coordination ----
single = df[df.config == "single"]
team_df = df[df.config == "team"]
print("\n=== (f) cost of coordination: team / single ===")
print("tokens:", round(team_df.total_tokens.mean() / single.total_tokens.mean(), 2))
print("agent_turns:", round(team_df.agent_turns.mean() / single.agent_turns.mean(), 2))
print("p95 latency:", round(p95(team_df.latency_ms) / p95(single.latency_ms), 2))
