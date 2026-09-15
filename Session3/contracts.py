"""
Assignment 5, Task 3 - the contract layer.

Decision (Task 3.1): **hybrid**. A small shared TeamState travels through
the LangGraph graph (task_id, run_idx, last_active, turn/token/time
counters, the accumulated `facts` dict, `constraints` list, terminal
state) - but the *content* handed from one agent to the next is never the
raw conversation history. It is always an explicit `Handoff` built by the
orchestrator from the previous worker's structured output. A worker never
sees another worker's tool-call trace, only the payload built for it.

Why hybrid and not pure message-passing: `facts` and `constraints` need to
survive across an arbitrary number of hops (e.g. cross_domain tasks where
the analyst needs a number the researcher found two hops ago), and
re-deriving that from a payload-only chain would mean every worker forwards
everything it ever received "just in case" - which is exactly the context
bloat this split exists to avoid. Why not pure shared state: if workers
wrote directly into one growing state object, nothing would force a
worker to summarize before handing off, and the state would re-accumulate
into the single bloated context this project's Assignment 4 wall
(context bloat on multi-hop tasks, see write-up) already showed is the
failure mode to avoid.

Ownership (Task 3.3): exactly one agent owns the turn at any moment,
stored in `TeamState.last_active` and logged on every transition. There is
no code path that leaves `last_active` unset after the graph starts -
"nobody owns it" cannot happen; a run that would leave ownership stuck
instead hits the loop-detection or turn-cap safety net and terminates with
a recorded reason code (Task 4.3).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

WorkerName = Literal["researcher", "analyst", "writer"]
AgentName = Literal["orchestrator", "researcher", "analyst", "writer"]


class HandoffPayload(BaseModel):
    """What actually travels between agents. NOT the conversation history."""

    summary: str = Field(description="one or two sentences of what's needed, not the whole task text")
    constraints: list[str] = Field(default_factory=list, description='e.g. "language=he", "max_words=50"')
    facts: dict[str, str] = Field(default_factory=dict, description="facts already established, key -> value")
    open_question: str = Field(
        default="", description="the specific thing the receiver is being asked to do"
    )


class Handoff(BaseModel):
    destination: WorkerName | Literal["direct_answer"] = Field(
        description='who takes over next, or "direct_answer" if the orchestrator can answer without dispatching'
    )
    payload: Optional[HandoffPayload] = None
    reason: str = Field(description="why this destination - for traces, never shown to the user")
    direct_answer: Optional[str] = Field(
        default=None, description="filled only when destination == direct_answer"
    )


class WorkerResult(BaseModel):
    """What a worker returns to the orchestrator after doing its job."""

    agent: WorkerName
    output: str = Field(description="the worker's answer/finding, in its own words")
    new_facts: dict[str, str] = Field(default_factory=dict, description="facts this worker established, to merge into state")
    could_not_complete: bool = False
    reason: Optional[str] = None


class TeamState(BaseModel):
    """The small shared state object every node reads/writes (Task 3.1)."""

    task_id: str
    run_idx: int
    task: str

    last_active: Optional[AgentName] = None  # ownership - never left unset once the run starts
    route: list[str] = Field(default_factory=list)  # ordered list of agents dispatched to
    handoff_log: list[dict] = Field(default_factory=list)  # for the loop-detection net

    facts: dict[str, str] = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)

    agent_turns: int = 0
    tool_calls: int = 0
    total_tokens: int = 0
    per_agent_turns: dict[str, int] = Field(default_factory=dict)
    per_agent_tokens: dict[str, int] = Field(default_factory=dict)

    final_answer: Optional[str] = None
    terminal_state: Optional[str] = None  # answered / refused / cap_breached / loop_detected / error
    breach_reason: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True
