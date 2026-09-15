# Three annotated traces (Assignment 5, submission item 5)

## 1. `best_cross_domain_win_c07.jsonl` — best cross-domain win

Task c07: waiting-period end-date computation. **Decisive step**: the
`researcher -> analyst` handoff at the second `HANDOFF` line — the payload's
`facts` carries `{"תקופת אכשרה": "9 חודשים", "תאריך התחלה": "2026-01-01"}`
straight from the researcher's retrieval into the analyst's
`date_duration` call, which returns `2026-09-28`. This is the handoff the
whole assignment is about: the second agent computed over a fact it never
looked up itself. Single agent: 0/3 on this exact task (see write-up
finding (a)).

## 2. `worst_coordination_failure_c06.jsonl` — worst coordination failure

Task c06 (Task-5 memory-ablation run, memory **on**): `researcher ->
researcher` fires twice in a row with near-duplicate query terms after an
already-failed search, total cost climbs to 134,608 tokens, and the loop
safety net fires (`terminal_state: loop_detected`). This is the trace
behind Task 7's "duplicated work" failure mode — see the write-up for the
mitigation (orchestrator rule 7: a retry must name a genuinely new search
term or go straight to refusal) and the re-measurement (0/6 duplicate
dispatches post-fix, tokens down to ~51k).

## 3. `handoff_stress_h01.jsonl` — a handoff_stress run

Task h01 (*"summarize the cancellation policy — in Hebrew, under 50
words"*). **What went wrong** is upstream of the handoff: `researcher`
made 3 genuinely-varied retrieval attempts and never surfaced any
cancellation-policy content (it isn't well-covered by this corpus's
chunking for either config — single agent hits the identical gap), so the
orchestrator refused via rule 6 before the writer — the agent actually
responsible for the language/length constraint — was ever dispatched. The
constraint itself was never lost; it was never exercised. See write-up
finding (d) for a case (`h02`) where the constraint *was* exercised and
correctly honored, exposing a bug in my own success-checker instead of a
real lossy handoff.
