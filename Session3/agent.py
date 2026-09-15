"""
Task 3.2-3.4 - The LangGraph ReAct agent, with the three safety nets and
full JSONL tracing.

Also implements Task 4's chosen pattern: **guardrails** (see
`apply_guardrails` below) - input scope check + output citation/refusal
check - toggled with `use_guardrails=True/False` so Task 4's before/after
table is a single flag flip on this same file.

Public entry point: run_agent(task, task_id, run_idx, trace_path, ...) -> dict
"""

import json
import time
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

from rag_pipeline import load_api_key, REFUSAL_SENTENCE
from tools import ALL_TOOLS

AGENT_MODEL = "claude-haiku-4-5"
AGENT_MODEL_UPGRADE = "claude-sonnet-5"  # Task 6, experiment 2
MAX_ITERATIONS = 12          # net 1: max steps (each step = one model call)
TOKEN_BUDGET = 120_000       # net 2: cumulative input+output tokens per task
WALLCLOCK_TIMEOUT_S = 90     # net 3: wall-clock per task

SYSTEM_PROMPT = """את/ה עוזר סוכן (agent) לתכניות ביטוח בריאות משלים (שב"ן) בישראל.
יש לך שלושה כלים: retrieve_policy_docs (חיפוש במסמכי הפוליסות),
calculate (חישוב אריתמטי), date_duration (חישובי תאריכים/תקופות).

כללים מחייבים:
1. ענה/י אך ורק על סמך מה שהכלים החזירו בפועל בשיחה הזו. אל תמציא/י מספרים,
   תאריכים או עובדות שלא הופיעו בתוצאות כלי.
2. אם כלי מחזיר הודעת שגיאה (ERROR:) פעמיים ברצף עבור אותה מטרה, הפסק/י
   לנסות ואמור/י בפירוש שלא ניתן להשלים את המשימה בגלל תקלה בכלי - אל תמציא/י
   תשובה במקום זאת. חוק זה חל גם כאשר הכלי שנכשל הוא calculate וניתן
   בעיון היה לחשב את התוצאה בעצמך בקלות - גם אז אסור לחשב בעצמך; יש
   לדווח על התקלה, כי המדיניות היא לא לעקוף כלי שנכשל בשום תנאי.
3. אל תקרא/י לכלי אם כבר יש לך את כל המידע הדרוש לתשובה.
4. אם השאלה לא דורשת שום כלי (למשל שאלה כללית על היכולות שלך, או בקשה
   לסכם משהו שכבר נאמר בשיחה), אל תקרא/י לאף כלי - ענה/י ישירות.
5. סרב/י בנימוס ובבירור אם התשובה אינה זמינה במסמכים או בכלים, אחרי חיפוש
   סביר (2-4 קריאות כלי לכל היותר) - אל תמשיך/י לנסות שוב ושוב וגם אל
   תנחש/י.
"""

# Task 6, experiment 1: same rules, but rule 5 is replaced with a *hard*
# numeric cap on retrieval retries (instead of the vague "2-4 calls, don't
# keep trying") and a mandatory verbatim refusal sentence, shared with the
# RAG pipeline's REFUSAL_SENTENCE so refusal is exact-string-checkable for
# both configs. See Task 6 write-up for the failure this targets: m06 made
# 7 retrieve_policy_docs calls with re-worded queries before hitting the
# token-budget net, and u02 refused in substance ("that's not in my
# documents") but in wording the eval's refusal regex didn't recognize.
SYSTEM_PROMPT_V2 = SYSTEM_PROMPT.replace(
    """5. סרב/י בנימוס ובבירור אם התשובה אינה זמינה במסמכים או בכלים, אחרי חיפוש
   סביר (2-4 קריאות כלי לכל היותר) - אל תמשיך/י לנסות שוב ושוב וגם אל
   תנחש/י.
""",
    f"""5. חובה: אל תקרא/י ל-retrieve_policy_docs יותר מ-4 פעמים באותה משימה,
   גם אם ניסוח השאילתה משתנה בכל פעם. אם אחרי 4 קריאות (או פחות, אם ברור
   שהמידע לא קיים) עדיין אין תשובה, הפסק/י מיד לחפש.
6. כשאין תשובה זמינה במסמכים או בכלים, יש להשיב **במדויק** במשפט הבא, מילה
   במילה, ולא בניסוח אחר: "{REFUSAL_SENTENCE}"
""",
)

GUARDRAIL_SCOPE_KEYWORDS = None  # set lazily below, kept simple/deterministic


def _in_scope(task: str) -> bool:
    """Guardrail (input): reject only obviously out-of-domain requests before
    spending any tokens. Deliberately permissive - a false reject is worse
    than an unnecessary tool call, so this only catches clear off-topic asks
    (code/poetry/unrelated-domain requests), not borderline insurance ones.
    """
    off_topic_markers = ["כתוב לי שיר", "כתוב קוד", "write a poem", "write code", "מתכון ל"]
    return not any(m in task for m in off_topic_markers)


_NUMBER_RE = __import__("re").compile(r"\d[\d,]*\.?\d*")


def _ungrounded_numbers(answer: str, tool_outputs: list[str]) -> list[str]:
    """Output guardrail: every number >=100 the model states in its final
    answer must appear (comma-insensitive) somewhere in this run's actual
    tool outputs. Numbers below 100 are skipped - they're usually quantities
    copied straight from the question (e.g. "3 visits"), not facts a tool
    supplied, so checking them would just produce false positives.

    This is a deterministic, code-only check (no LLM call) that directly
    targets the fabrication failure mode seen in the tool_fails tasks: a
    tool returns ERROR, and the model computes/states the number anyway.
    """
    haystack = " ".join(tool_outputs).replace(",", "")
    unverified = []
    for match in _NUMBER_RE.findall(answer):
        normalized = match.replace(",", "")
        try:
            value = float(normalized)
        except ValueError:
            continue
        if value < 100:
            continue
        if normalized not in haystack:
            unverified.append(match)
    return unverified


def _build_agent(use_guardrails: bool, model: str = AGENT_MODEL, system_prompt: str = SYSTEM_PROMPT):
    # claude-sonnet-5 rejects an explicit temperature (deprecated for that
    # model); only pass it for models that still accept it.
    kwargs = {"model": model, "api_key": load_api_key(), "max_tokens": 1024}
    if model != AGENT_MODEL_UPGRADE:
        kwargs["temperature"] = 0
    llm = ChatAnthropic(**kwargs)
    return create_react_agent(llm, ALL_TOOLS, prompt=system_prompt)


def run_agent(
    task: str,
    task_id: str = "adhoc",
    run_idx: int = 1,
    trace_path: str | Path | None = None,
    use_guardrails: bool = False,
    model: str = AGENT_MODEL,
    system_prompt: str = SYSTEM_PROMPT,
) -> dict:
    """Run one task through the agent once. Returns a summary dict and (if
    trace_path given) appends one JSONL line per step plus a summary line.

    terminal_state is one of: answered / refused / cap_breached / error
    """
    trace_lines = []
    start = time.perf_counter()

    if use_guardrails and not _in_scope(task):
        summary = _summary(task_id, run_idx, steps=0, total_tokens=0,
                            wall_ms=(time.perf_counter() - start) * 1000,
                            terminal_state="refused", answer="מצטער/ת, אני יכול/ה לעזור רק בשאלות על תכניות ביטוח בריאות משלים.",
                            tool_calls=0, tools_used=[])
        _write_trace(trace_path, trace_lines, summary)
        return summary

    agent = _build_agent(use_guardrails, model=model, system_prompt=system_prompt)

    total_tokens = 0
    total_input_tokens = 0
    total_output_tokens = 0
    tool_calls = 0
    tools_used = []
    tool_outputs_text = []
    terminal_state = "answered"
    final_answer = ""
    step = 0

    try:
        config = {"recursion_limit": MAX_ITERATIONS * 2 + 2}  # LangGraph counts each node visit
        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": task}]},
            config=config,
            stream_mode="values",
        ):
            if time.perf_counter() - start > WALLCLOCK_TIMEOUT_S:
                terminal_state = "cap_breached"
                final_answer = "לא ניתן היה להשלים את המשימה בזמן הקצוב (timeout)."
                break
            if total_tokens > TOKEN_BUDGET:
                terminal_state = "cap_breached"
                final_answer = "לא ניתן היה להשלים את המשימה במסגרת תקציב הטוקנים."
                break

            messages = chunk["messages"]
            last = messages[-1]
            usage = getattr(last, "usage_metadata", None) or {}
            in_tok = usage.get("input_tokens", 0) or 0
            out_tok = usage.get("output_tokens", 0) or 0
            total_tokens += in_tok + out_tok
            total_input_tokens += in_tok
            total_output_tokens += out_tok

            role = getattr(last, "type", "")
            if role == "ai":
                step += 1
                tc_list = getattr(last, "tool_calls", []) or []
                trace_lines.append({
                    "task_id": task_id, "run": run_idx, "step": step,
                    "thought": last.content if isinstance(last.content, str) else str(last.content),
                    "tool": tc_list[0]["name"] if tc_list else None,
                    "input": tc_list[0]["args"] if tc_list else None,
                    "output": None, "duration_ms": None,
                    "input_tokens": in_tok, "output_tokens": out_tok,
                })
                if not tc_list:
                    final_answer = last.content if isinstance(last.content, str) else str(last.content)
            elif role == "tool":
                tool_calls += 1
                tools_used.append(last.name)
                if trace_lines and trace_lines[-1]["tool"] == last.name and trace_lines[-1]["output"] is None:
                    trace_lines[-1]["output"] = last.content
                if isinstance(last.content, str):
                    tool_outputs_text.append(last.content)
        else:
            pass

        if step >= MAX_ITERATIONS and terminal_state == "answered" and not final_answer:
            terminal_state = "cap_breached"
            final_answer = "לא ניתן היה להשלים את המשימה במגבלת הצעדים (step limit)."

    except Exception as e:
        terminal_state = "error"
        final_answer = f"ERROR: agent run failed: {e}"

    if use_guardrails and terminal_state == "answered" and tool_calls > 0:
        unverified = _ungrounded_numbers(final_answer, tool_outputs_text)
        if unverified:
            terminal_state = "refused"
            final_answer = (
                "לא ניתן לאשר את התשובה: הסכום/המספר "
                f"{', '.join(unverified)} אינו מופיע בתוצאות הכלים בפועל בשיחה הזו, "
                "וייתכן שהוא שגוי או הומצא. אני נמנע/ת ממתן תשובה שלא ניתן לאמת."
            )

    if terminal_state == "answered":
        refusal_markers = ["לא מצאתי", "אינני יכול", "לא ניתן", "מצטער"]
        if any(m in final_answer for m in refusal_markers) and tool_calls > 0:
            terminal_state = "refused"
        elif not final_answer:
            terminal_state = "refused"
            final_answer = "לא הצלחתי למצוא תשובה למשימה זו."

    wall_ms = (time.perf_counter() - start) * 1000
    summary = _summary(task_id, run_idx, steps=step, total_tokens=total_tokens,
                        wall_ms=wall_ms, terminal_state=terminal_state,
                        answer=final_answer, tool_calls=tool_calls, tools_used=tools_used,
                        total_input_tokens=total_input_tokens, total_output_tokens=total_output_tokens,
                        model=model)
    _write_trace(trace_path, trace_lines, summary)
    return summary


def _summary(task_id, run_idx, steps, total_tokens, wall_ms, terminal_state, answer, tool_calls, tools_used,
             total_input_tokens=0, total_output_tokens=0, model=AGENT_MODEL) -> dict:
    return {
        "task_id": task_id, "run": run_idx, "steps": steps,
        "total_tokens": total_tokens,
        "total_input_tokens": total_input_tokens, "total_output_tokens": total_output_tokens,
        "wall_ms": wall_ms, "model": model,
        "terminal_state": terminal_state, "answer": answer,
        "tool_calls": tool_calls, "tools_used": tools_used,
    }


def _write_trace(trace_path, trace_lines, summary):
    if trace_path is None:
        return
    with open(trace_path, "a", encoding="utf-8") as f:
        for line in trace_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        f.write(json.dumps({"summary": True, **summary}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    task = (
        "כמה יעלה בסך הכל למבוטח מכבי זהב לבצע 3 ביקורים אצל רופא מומחה "
        "השנה, לפי דמי ההשתתפות העצמית שלו?"
    )
    result = run_agent(task, task_id="demo", run_idx=1, trace_path="traces_demo.jsonl")
    print(json.dumps(result, ensure_ascii=False, indent=2))
