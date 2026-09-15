"""
Assignment 5, Task 4 - the multi-agent team: orchestrator + 3 workers.

Topology: orchestrator-worker (Task 2.2). The orchestrator classifies and
dispatches on every turn; workers never call each other and always return
to the orchestrator (or, for the writer, terminate the run directly - see
below). That single-hub structure is what keeps one trace readable instead
of three, per Task 4.2's invariant.

Agents (Task 2.1 scope contracts - each passes the "no *and*" test):

  researcher  - searches the שב"ן corpus and returns cited passages.
                tools: retrieve_policy_docs
  analyst     - computes a number or date from facts it's given.
                tools: calculate, date_duration
  writer      - synthesizes the final answer under the handoff's constraints.
                tools: none

Why this split (the wall, named in the write-up, is *context bloat*):
Assignment 4's m06 blew 155k tokens and a cap breach re-querying the same
retriever inside one ever-growing context. Splitting retrieval, computation
and synthesis into three small, single-purpose contexts means no single
agent ever re-sends a growing multi-tool history - each worker gets exactly
the payload the orchestrator decided it needs (contracts.HandoffPayload),
nothing more.

Four safety nets (Task 4.3), checked every orchestrator turn:
  1. max agent turns (MAX_AGENT_TURNS)
  2. token budget (TOKEN_BUDGET)
  3. wall-clock timeout (WALLCLOCK_TIMEOUT_S)
  4. loop detection: the same (from, to) handoff pair twice in a row

Public entry point: run_team(task, task_id, run_idx, trace_path) -> dict
(same summary shape as agent.run_agent, plus `route` and per-agent stats,
so eval_runner_team.py can score both configs uniformly).
"""

import json
import re
import time
from pathlib import Path
from typing import Literal, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from contracts import Handoff, HandoffPayload, TeamState
from rag_pipeline import load_api_key, REFUSAL_SENTENCE
from tools import calculate, date_duration, retrieve_policy_docs

ORCHESTRATOR_MODEL = "claude-haiku-4-5"
WORKER_MODEL = "claude-haiku-4-5"
WORKER_MODEL_UPGRADE = "claude-sonnet-5"  # Task 7 upgrade lever, held in reserve

MAX_AGENT_TURNS = 8
TOKEN_BUDGET = 150_000
WALLCLOCK_TIMEOUT_S = 120
WORKER_RECURSION_LIMIT = 10  # per-worker sub-cap: a worker gets a handful of tool calls, not an unbounded loop

AGENTS_MD = Path(__file__).with_name("AGENTS.md").read_text(encoding="utf-8")

WORKER_TOOLS = {
    "researcher": [retrieve_policy_docs],
    "analyst": [calculate, date_duration],
    "writer": [],
}

SCOPE_CONTRACTS = {
    "researcher": (
        "את/ה ה-researcher בצוות שב\"ן. תפקידך היחיד: לחפש במסמכי הפוליסות "
        "ולהחזיר עובדות מצוטטות עם מקור. אל תחשב/י ארוחמטיקה ואל תנסח/י "
        "תשובה סופית מעוצבת - זה לא תפקידך. אם החיפוש לא מעלה תוצאה אחרי "
        "2-4 ניסיונות, דווח/י שלא נמצא ועצור/י."
    ),
    "analyst": (
        "את/ה ה-analyst בצוות שב\"ן. תפקידך היחיד: לחשב מספר או תאריך מתוך "
        "עובדות שכבר סופקו לך ב-facts. אל תחפש/י במסמכים ואל תנחש/י עובדה "
        "חסרה - אם עובדה הדרושה לחישוב לא סופקה, דווח/י זאת ועצור/י."
    ),
    "writer": (
        "את/ה ה-writer בצוות שב\"ן. תפקידך היחיד: לנסח תשובה סופית מהעובדות "
        "שסופקו לך, תוך עמידה מדויקת באילוצים (שפה, אורך). אל תוסיף/י "
        "עובדה או מספר שלא הופיע ב-facts שסופקו לך."
    ),
}

ORCHESTRATOR_SYSTEM = f"""את/ה האורכסטרטור של צוות סוכנים לשב\"ן (ביטוח בריאות משלים בישראל).
תפקידך היחיד: לסווג את הבקשה ולשגר אותה לעובד המתאים, או לענות ישירות אם
אין צורך בשום עובד.

צוות העובדים הזמינים לך (תיאור קצר בלבד - אל תוסיף/י ידע עליהם):
- researcher: מחפש במסמכי הפוליסות (שלושה מבטחים) ומחזיר עובדות מצוטטות.
- analyst: מחשב מספר או תאריך מתוך עובדות שכבר נאספו.
- writer: מנסח תשובה סופית העומדת באילוצי שפה/אורך, מתוך עובדות שכבר נאספו.

כללים:
1. אין לוגיקה עסקית כאן - רק סיווג ושיגור. אל תחשב/י בעצמך ואל תחפש/י בעצמך.
2. אם הבקשה לא דורשת שום עובד (שיחת חולין, שאלה על היכולות שלך, תודה),
   ענה/י ישירות (destination="direct_answer") - זה המסלול המהיר.
3. אם יש עובדות שכבר נאספו (facts) שמספיקות לענות, ואין אילוצי פורמט
   (שפה/אורך) שדורשים ניסוח מיוחד - ענה/י ישירות מהעובדות הללו.
4. אם יש אילוצי פורמט (שפה מסוימת, מגבלת מילים/משפטים) - יש לשגר ל-writer
   כדי שהתשובה הסופית תעמוד בהם, גם אם העובדות כבר ידועות.
5. שגר/י ל-analyst רק כאשר יש חישוב אריתמטי/תאריך לבצע על עובדות שכבר יש.
6. אם אחרי חיפוש סביר (2-4 קריאות researcher) אין עובדות - ענה/י ישירות
   בסירוב, במדויק במילים הבאות: "{REFUSAL_SENTENCE}"
7. אם researcher כבר דיווח כישלון וברצונך לשגר אליו שוב - מותר פעם אחת
   בלבד, ורק אם ה-open_question בפעולה החדשה מפרש/ת בשמות מפורשים מונחי
   חיפוש **שונים לגמרי** ממה שכבר נוסה (למשל: שם מבטח ספציפי לבד, מספר
   סעיף, מילת מפתח חלופית) - אל תשגר/י שוב עם אותה בקשה כללית. אם אין לך
   רעיון למונח חיפוש חדש וממשי, עברי/עבור ישירות לסירוב (כלל 6) במקום
   לשגר שוב.

{AGENTS_MD}
"""


def _worker_system_prompt(agent: str) -> str:
    return f"{SCOPE_CONTRACTS[agent]}\n\n{AGENTS_MD}"


class _HandoffFlat(BaseModel):
    """Tool schema actually shown to the orchestrator LLM. Anthropic's tool
    calling occasionally double-JSON-encodes a *nested* object field as a
    string, and when that string itself echoes source text containing a
    literal embedded quote (e.g. this corpus's own \"חו\"ל\"), the nested
    JSON's escaping breaks and the payload string isn't even valid JSON -
    seen live on the u01/u02 unanswerable tasks, which happen to contain
    exactly that character. A flat schema (no nested object for the model to
    self-serialize) sidesteps the bug entirely; Handoff/HandoffPayload
    (contracts.py) stay nested for everything else in the codebase - they're
    just reassembled from this flat shape in `_to_handoff` below."""

    destination: Literal["researcher", "analyst", "writer", "direct_answer"]
    reason: str
    direct_answer: Optional[str] = None
    summary: Optional[str] = None
    constraints: list[str] = Field(default_factory=list)
    facts: dict[str, str] = Field(default_factory=dict)
    open_question: Optional[str] = None


def _to_handoff(flat: _HandoffFlat) -> Handoff:
    payload = None
    if flat.destination != "direct_answer":
        payload = HandoffPayload(
            summary=flat.summary or "", constraints=flat.constraints,
            facts=flat.facts, open_question=flat.open_question or "",
        )
    return Handoff(destination=flat.destination, payload=payload, reason=flat.reason,
                    direct_answer=flat.direct_answer)


def _build_orchestrator():
    llm = ChatAnthropic(model=ORCHESTRATOR_MODEL, api_key=load_api_key(), max_tokens=512, temperature=0)
    return llm.with_structured_output(_HandoffFlat, include_raw=True)


def _parse_handoff(result: dict) -> Handoff | None:
    if result["parsed"] is not None:
        return _to_handoff(result["parsed"])
    result["_recovery_error"] = str(result.get("parsing_error"))
    return None


def _build_worker(agent: str, model: str = WORKER_MODEL):
    kwargs = {"model": model, "api_key": load_api_key(), "max_tokens": 1024}
    if model != WORKER_MODEL_UPGRADE:
        kwargs["temperature"] = 0
    llm = ChatAnthropic(**kwargs)
    return create_react_agent(llm, WORKER_TOOLS[agent], prompt=_worker_system_prompt(agent))


def _payload_message(payload: HandoffPayload) -> str:
    lines = [f"בקשה: {payload.open_question}", f"תקציר: {payload.summary}"]
    if payload.facts:
        lines.append("עובדות ידועות: " + json.dumps(payload.facts, ensure_ascii=False))
    if payload.constraints:
        lines.append("אילוצים מחייבים: " + "; ".join(payload.constraints))
    return "\n".join(lines)


def _run_worker(agent: str, payload: HandoffPayload, task_id, run_idx, seq_ref, trace_lines, model=WORKER_MODEL):
    """Runs one worker to completion. Returns (output_text, tool_calls, tokens_used)."""
    worker = _build_worker(agent, model=model)
    config = {"recursion_limit": WORKER_RECURSION_LIMIT}
    output_text = ""
    tool_calls = 0
    tokens_used = 0
    last_tool_name = None
    start = time.perf_counter()

    for chunk in worker.stream(
        {"messages": [HumanMessage(content=_payload_message(payload))]},
        config=config,
        stream_mode="values",
    ):
        last = chunk["messages"][-1]
        usage = getattr(last, "usage_metadata", None) or {}
        tokens_used += (usage.get("input_tokens", 0) or 0) + (usage.get("output_tokens", 0) or 0)
        role = getattr(last, "type", "")
        if role == "ai":
            tc_list = getattr(last, "tool_calls", []) or []
            if tc_list:
                last_tool_name = tc_list[0]["name"]
                seq_ref[0] += 1
                trace_lines.append({
                    "task_id": task_id, "run": run_idx, "seq": seq_ref[0], "agent": agent,
                    "event": "tool_call_start", "tool": last_tool_name, "input": tc_list[0]["args"],
                    "owner": agent,
                })
            else:
                output_text = last.content if isinstance(last.content, str) else str(last.content)
        elif role == "tool":
            tool_calls += 1
            duration_ms = round((time.perf_counter() - start) * 1000, 1)
            seq_ref[0] += 1
            trace_lines.append({
                "task_id": task_id, "run": run_idx, "seq": seq_ref[0], "agent": agent,
                "event": "tool_call", "tool": getattr(last, "name", last_tool_name),
                "output": last.content if isinstance(last.content, str) else str(last.content),
                "owner": agent, "duration_ms": duration_ms,
            })
    return output_text, tool_calls, tokens_used


_NUM_RE = re.compile(r"\d[\d,]*\.?\d*")


def _extract_facts(agent: str, output_text: str) -> dict:
    """Cheap deterministic fact-extraction: pull numbers/dates the worker
    stated, keyed by agent+index, so the orchestrator's next turn can see
    them without re-reading the worker's full reasoning trace."""
    facts = {}
    if agent == "researcher":
        for i, m in enumerate(_NUM_RE.findall(output_text)[:6]):
            facts[f"researcher_fact_{i+1}"] = m
    elif agent == "analyst":
        for i, m in enumerate(_NUM_RE.findall(output_text)[:3]):
            facts[f"analyst_result_{i+1}"] = m
    facts[f"{agent}_output"] = output_text.strip()[:800]
    return facts


def run_team(
    task: str,
    task_id: str = "adhoc",
    run_idx: int = 1,
    trace_path: str | Path | None = None,
    worker_model: str = WORKER_MODEL,
    upgrade_agent: str | None = None,
) -> dict:
    """Runs one task through the team once. Returns a summary dict and (if
    trace_path given) appends JSONL trace lines: tool_call events, handoff
    events, and one summary line (terminal_state one of: answered / refused
    / cap_breached / loop_detected / error)."""
    state = TeamState(task_id=task_id, run_idx=run_idx, task=task)
    trace_lines = []
    seq = [0]
    start = time.perf_counter()
    orchestrator = _build_orchestrator()

    try:
        while True:
            # --- safety nets, checked before every orchestrator turn ---
            if state.agent_turns >= MAX_AGENT_TURNS:
                state.terminal_state = "cap_breached"
                state.breach_reason = "max_agent_turns"
                state.final_answer = "לא ניתן היה להשלים את המשימה במגבלת תורות הצוות (agent turn cap)."
                break
            if state.total_tokens > TOKEN_BUDGET:
                state.terminal_state = "cap_breached"
                state.breach_reason = "token_budget"
                state.final_answer = "לא ניתן היה להשלים את המשימה במסגרת תקציב הטוקנים."
                break
            if time.perf_counter() - start > WALLCLOCK_TIMEOUT_S:
                state.terminal_state = "cap_breached"
                state.breach_reason = "wallclock_timeout"
                state.final_answer = "לא ניתן היה להשלים את המשימה בזמן הקצוב (timeout)."
                break

            # --- orchestrator turn: classify + dispatch ---
            state.last_active = "orchestrator"
            prompt = (
                f"משימה מקורית: {task}\n\n"
                f"מסלול עד כה: {state.route or '(עדיין לא שוגר אף עובד)'}\n"
                f"עובדות שנאספו עד כה: {json.dumps(state.facts, ensure_ascii=False) or '(אין)'}\n"
                f"אילוצים שזוהו: {state.constraints or '(אין)'}"
            )
            handoff = None
            last_parsing_error = None
            for _orch_attempt in range(2):  # one retry: rare structured-output misses under load (empirically ~5%)
                result = orchestrator.invoke([
                    {"role": "system", "content": ORCHESTRATOR_SYSTEM},
                    {"role": "user", "content": prompt},
                ])
                raw_msg = result["raw"]
                usage = getattr(raw_msg, "usage_metadata", None) or {}
                orch_tokens = (usage.get("input_tokens", 0) or 0) + (usage.get("output_tokens", 0) or 0)
                state.total_tokens += orch_tokens
                state.per_agent_tokens["orchestrator"] = state.per_agent_tokens.get("orchestrator", 0) + orch_tokens
                handoff = _parse_handoff(result)
                last_parsing_error = result.get("_recovery_error") or result.get("parsing_error")
                if handoff is not None:
                    break

            if handoff is None:
                state.terminal_state = "error"
                state.final_answer = f"ERROR: orchestrator failed to produce a structured handoff after 2 attempts: {last_parsing_error}"
                break

            from_agent = state.route[-1] if state.route else "orchestrator"
            seq[0] += 1
            trace_lines.append({
                "task_id": task_id, "run": run_idx, "seq": seq[0], "agent": "orchestrator",
                "event": "handoff", "from": from_agent, "to": handoff.destination,
                "reason": handoff.reason,
                "payload_keys": list(handoff.payload.model_dump().keys()) if handoff.payload else None,
                "owner": handoff.destination,
            })

            # --- loop detection: same (from, to) pair twice in a row ---
            state.handoff_log.append({"from": from_agent, "to": handoff.destination})
            if len(state.handoff_log) >= 2:
                a, b = state.handoff_log[-2], state.handoff_log[-1]
                if a == b and a["to"] not in ("direct_answer",):
                    state.terminal_state = "loop_detected"
                    state.breach_reason = f"repeated handoff {a['from']}->{a['to']}"
                    state.final_answer = "התגלה לולאת שיגור חוזרת בין אותם סוכנים; העיבוד נעצר."
                    break

            if handoff.destination == "direct_answer":
                state.terminal_state = "answered"
                state.final_answer = handoff.direct_answer or "לא הוגדרה תשובה."
                state.last_active = "orchestrator"
                break

            if handoff.payload and handoff.payload.constraints:
                for c in handoff.payload.constraints:
                    if c not in state.constraints:
                        state.constraints.append(c)

            # --- dispatch the worker ---
            agent = handoff.destination
            state.last_active = agent
            state.route.append(agent)
            state.agent_turns += 1
            state.per_agent_turns[agent] = state.per_agent_turns.get(agent, 0) + 1

            payload = handoff.payload or HandoffPayload(summary=task, open_question=task)
            model_for_agent = WORKER_MODEL_UPGRADE if agent == upgrade_agent else worker_model
            output_text, tool_calls, tokens_used = _run_worker(
                agent, payload, task_id, run_idx, seq, trace_lines, model=model_for_agent
            )
            state.tool_calls += tool_calls
            state.total_tokens += tokens_used
            state.per_agent_tokens[agent] = state.per_agent_tokens.get(agent, 0) + tokens_used

            new_facts = _extract_facts(agent, output_text)
            state.facts.update(new_facts)

            if agent == "writer":
                # the writer's job is exactly final synthesis - it always
                # terminates the run rather than handing back, so ownership
                # can never bounce writer -> orchestrator -> writer forever.
                state.terminal_state = "answered"
                state.final_answer = output_text.strip() or "לא הופקה תשובה."
                break

        if REFUSAL_SENTENCE in (state.final_answer or "") and state.terminal_state == "answered":
            state.terminal_state = "refused"

    except Exception as e:
        state.terminal_state = "error"
        state.final_answer = f"ERROR: team run failed: {e}"

    wall_ms = (time.perf_counter() - start) * 1000
    summary = {
        "task_id": task_id, "run": run_idx, "agent_turns": state.agent_turns,
        "tool_calls": state.tool_calls, "total_tokens": state.total_tokens,
        "wall_ms": wall_ms, "terminal_state": state.terminal_state, "answer": state.final_answer,
        "breach_reason": state.breach_reason, "route": state.route,
        "per_agent_turns": state.per_agent_turns, "per_agent_tokens": state.per_agent_tokens,
        "facts": state.facts,
    }
    _write_trace(trace_path, trace_lines, summary)
    return summary


def _write_trace(trace_path, trace_lines, summary):
    if trace_path is None:
        return
    with open(trace_path, "a", encoding="utf-8") as f:
        for line in trace_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        f.write(json.dumps({"summary": True, **summary}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    task = (
        "כמה בסך הכל ישלם מבוטח מאוחדת עדיף עבור 3 ניתוחים פרטיים בישראל השנה, "
        "לפי דמי ההשתתפות העצמית שלו?"
    )
    result = run_team(task, task_id="demo", run_idx=1, trace_path="traces_team_demo.jsonl")
    print(json.dumps(result, ensure_ascii=False, indent=2))
