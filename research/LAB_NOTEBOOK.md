# Lab notebook

One entry per experiment, newest at the bottom. Write the entry **before** you
run: mechanism and prediction first, measurement after. An entry with no
prediction is not an experiment, it is a change.

Rules: one hypothesis per entry; state the mechanism in causal terms; predict a
range, not a point; record the verdict from results.json verbatim, including the
ones that embarrass the prediction. If a sceptic (a helper agent, or you) raised an objection,
answer it here before you run.

**This file ships empty.** The entry below is a filled-in form showing what a
complete entry looks like. Every number in it is invented: it was not measured
on this box, and no hypothesis in it has been settled. Replace it with your
first real entry.

---

## E0 — EXAMPLE ENTRY, INVENTED NUMBERS, NOT A RESULT

**id**: `async-sched-on`
**tag**: RECIPE
**hypothesis**: turning on vLLM's async scheduling raises tokens per second by
4 to 9%.
**mechanism**: the scheduler currently runs between forward passes on the same
thread, so every decode step pays CPU scheduling time the GPU spends idle.
Async scheduling overlaps the next step's scheduling with the current forward
pass. The profile shows 3.1 ms of a 19 ms decode step in scheduler Python, so
the ceiling is about 16% on decode.
**prediction**: tokens per second +4% to +9%, no quality change (the sampling
path is untouched).
**sceptic said**: "the engine may already enable this for this model." Checked:
`async_scheduling` defaults to None and the baseline does not set it.
**cost**: 6 min.

**result**: REJECTED_SCREEN. Screen window 2.4% below the incumbent's stored
value, so no full comparison. Server log shows async scheduling active
("async_scheduling=True" at startup), so the flag took effect.
**what I learned**: the scheduler is not on the critical path at this
concurrency: the CPU work overlaps a forward pass that is already long enough
to hide it. This weakens every "reduce CPU overhead" hypothesis in the
backlog; I have retagged three of them MEASURE FIRST.
**next**: attack the prefill path instead (see HYPOTHESES.md H4).
