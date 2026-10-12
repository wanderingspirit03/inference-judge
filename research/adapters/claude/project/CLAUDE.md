@AGENTS.md

## Helper agents (Claude Code)

Three subagents are in `.claude/agents/`:

- **scout** has WebSearch and WebFetch. Use it for the web search before each hypothesis, and for
  reading logs, profiles, source and docs so your own context stays for decisions.
- **hypothesizer** returns ranked hypotheses with mechanisms and numeric predictions. If the
  operator added narrower generators, use those whose description matches the target's
  accelerator; otherwise call the one generator once per angle.
- **sceptic** tries to kill an idea before it costs a ruler slot.

When the backlog is thin or you are stuck, fan out: give the hypothesizer and the sceptic the
latest profile and verdicts, then merge what they return.
