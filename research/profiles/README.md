# profiles/

Empty until you make one. CLAUDE.md says to read "the latest profile if one
exists"; this is where it lives.

There is no profile after bootstrap. The baselines serve, get measured and are
torn down, and wrapping the baseline itself in a profiler would change the
number every promotion is measured against.

To get the first one:

```bash
echo '{"id": "profile-baseline", "kind": "instrument"}' > queue/profile-baseline.json
```

`experiments/profile-baseline/launch.sh` is `baseline-vllm` under
`nsys`. Its verdict is INSTRUMENT: an instrument run is never scored, and the
instrumentation costs a few percent. What you want is what run.sh collects out of `/out`, which lands in the
run's `artifacts/` directory:

```
/opt/bench/results/profile-baseline/artifacts/profile.nsys-rep      the trace
/opt/bench/results/profile-baseline/artifacts/profile-summary.txt   kernel + API tables
/opt/bench/results/profile-baseline/artifacts/gpu-dmon.txt          only if nsys was missing
```

`--kill none` is why the server survives its own profile: nsys terminates the
application it launched when the capture window closes unless it is told not to.
Keep it in any profile you write.

Copy the summary here under a name that says what it profiled and when, e.g.
`profiles/2026-01-15T01-10Z-baseline-vllm.txt`, and say in the
notebook entry which profile a hypothesis came from. The `.nsys-rep` itself is
gitignored; the text summary is what belongs in the repo.

A profile of a *candidate* is made the same way: copy your experiment's
launch.sh, wrap the `exec` line in the same `nsys profile` call, and give it a
new id ending in `-profile`. Never profile the run you are measuring.
