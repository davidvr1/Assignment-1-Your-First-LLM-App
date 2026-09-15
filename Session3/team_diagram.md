# Team diagram — Assignment 5

```mermaid
flowchart TB
    U[User request] --> O

    subgraph Team["orchestrator-worker topology"]
        O["Orchestrator (claude-haiku-4-5)<br/>classify + dispatch only<br/>owns: TeamState (shared)"]
        R["Researcher (claude-haiku-4-5)<br/>tools: retrieve_policy_docs<br/>owns: nothing persists past its turn"]
        A["Analyst (claude-haiku-4-5)<br/>tools: calculate, date_duration"]
        W["Writer (claude-haiku-4-5 / sonnet-5 upgrade)<br/>tools: none — synthesis only"]
    end

    O -- "Handoff(destination=researcher, payload)" --> R
    R -- "WorkerResult -> facts merged into TeamState" --> O
    O -- "Handoff(destination=analyst, payload)" --> A
    A -- "WorkerResult -> facts merged into TeamState" --> O
    O -- "Handoff(destination=writer, payload)" --> W
    W -- "final_answer (terminates run)" --> U
    O -- "direct_answer (no_tool fast path)" --> U

    R -.-> DOCS[(שב"ן corpus<br/>FAISS index)]
    A -.-> CALC[[AST-only arithmetic]]
    A -.-> DATE[[date/duration utility]]
```

**Where state lives:** `TeamState` (contracts.py) is the small shared object
every node reads/writes — `last_active` (ownership, single-valued, logged
every transition), `route`, `handoff_log` (for loop detection), `facts`,
`constraints`, and the turn/token/time counters for the four safety nets.
It is *not* the conversation history. What crosses each arrow above is an
explicit `HandoffPayload` (summary, constraints, facts, open_question) —
never the full message list — so no worker's context re-accumulates the
whole run the way Assignment 4's single agent did.

**Ownership invariant:** workers never call each other and always return
to the orchestrator, except the writer, whose output always terminates the
run (it is the synthesis step, so there is nothing left to hand back) —
this is what makes "nobody owns it" structurally impossible rather than a
silent failure mode.
