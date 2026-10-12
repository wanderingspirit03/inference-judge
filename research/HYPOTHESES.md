# Hypotheses

The backlog. Append, never delete: a tried hypothesis gets a status and the
notebook entry that settled it. The literature pass (loop/literature.sh) and any
helper agents you use append here too.

Status: `open` | `running` | `settled (E<n>)` | `dropped (reason)`.

Tag every entry RECIPE (a documented setting), ADAPT (a known technique applied
where the docs do not cover it) or NEW (you searched and found nothing; say what
you searched).

**This file ships empty.** The single entry below is a filled-in form showing
the shape of an entry, not a claim about this box. Delete it or leave it; either
way, nothing in it has been measured here.

---

### H0 (EXAMPLE ENTRY -- NOT A RESULT, NOT MEASURED ON THIS BOX)
- **title**: one line naming the change, not the outcome
- **tag**: RECIPE | ADAPT | NEW
- **mechanism**: why this would change the target's metric, in causal terms,
  naming the stage of the request it touches (prefill, scheduling, decode
  steps, detokenisation, SSE write). "It is faster" is not a mechanism.
- **predicted**: a range for the headline number, with a sign. If the target
  measures several input sizes, say which should move more, because that is
  the tell that the mechanism is the real one.
- **cost**: minutes of GPU time, including the build.
- **risk**: what it could break, especially anything the canary or the quality
  gate would catch.
- **source**: where the idea came from; for NEW, the searches that came up empty.
- **already shipped?**: whether the engine already does this by default.
- **status**: open

---

<!-- Append real hypotheses below this line. -->
