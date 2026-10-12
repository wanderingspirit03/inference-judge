# Plan

What you intend to do with the clock, and what changed. Keep it current: the
driver gives you elapsed hours and tallies on every iteration, and this file is
where you reconcile them with reality. Two lines of update beat a rewrite.

## Shape of the day (an example for a 24-hour budget; scale it to yours)

- hours 0-1: profile the baseline, fan out to helper agents if you have them, fill the backlog
- hours 1-4: settings and scheduling levers (cheap, likely wins)
- hours 4-22: kernels, engine patches, engine swaps, interleaved with config follow-ups
- hours 22-23: re-run the best promoted config to confirm it
- the last 30 minutes: SUMMARY.md (the driver tells you when the wrap-up starts; if you
  write none, it leaves a stub of its own)

## Current intent

_EXAMPLE TABLE — invented, replace with your own. The hour numbers and the
hypothesis ids below refer to the example entries in HYPOTHESES.md and
LAB_NOTEBOOK.md, which are also examples. Nothing here has been measured._

| hour | intent | why |
|---|---|---|
| 0.0-0.7 | profile the baseline, nsys 30 s slice | nothing else is informed until the decode step is broken down |
| 0.7-1.0 | fan out to the hypothesis agents, then the sceptic | fill the backlog before spending GPU time |
| 1.0-2.5 | H3 (batch size), H1 (async sched) | cheapest two, 16 min of GPU total |
| 2.5-5.0 | H2 (prefix pinning) | the only hypothesis whose predicted effect grows with concurrency |

## Changes to the plan

_EXAMPLE ENTRIES — invented, replace._

- **02:14Z** H1 rejected at screen (E0). The "reduce CPU overhead" family is
  weaker than I assumed; moved H5 and H7 to MEASURE FIRST and pulled H2 forward.
- **03:40Z** H2 variant `prefill-chunk-1024` promoted at -8.2%. Kernel track
  starts now rather than at hour 6: the config levers are giving less than the
  profile says the MoE dispatch is worth.
