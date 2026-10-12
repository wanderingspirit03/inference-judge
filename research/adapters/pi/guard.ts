// Pi extension: every tool call goes through the inference-judge guard before
// it runs.  The guard is the set of scripts in research/guard, the same ones
// Claude Code and Codex call as hooks.  session.sh loads this file with
// `pi -e` and names the guard's folder in IJ_GUARD_DIR.
//
// Plain JavaScript on purpose (no types, nothing imported from pi), so it
// loads under whatever version of pi is installed.
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { join } from "node:path";

const GUARD_DIR = process.env.IJ_GUARD_DIR || "";
// Pi's own tools that only read.  Every other tool is checked.
const READERS = new Set(["read", "grep", "find", "ls"]);

// Run one guard script on one tool call.  The script reads
// {tool_name, tool_input, cwd} on stdin and exits 0 to allow the call.
function guard(script, event, ctx) {
  const path = join(GUARD_DIR, script);
  if (!GUARD_DIR || !existsSync(path)) {
    return { ok: false, why: `the guard script ${path || script} is missing (IJ_GUARD_DIR)` };
  }
  const call = { tool_name: event.toolName, tool_input: event.input || {}, cwd: (ctx && ctx.cwd) || process.cwd() };
  const r = spawnSync(path, [], { input: JSON.stringify(call), encoding: "utf8", timeout: 15000 });
  if (r.status === 0) return { ok: true, why: "" };
  return { ok: false, why: (r.stderr || "").trim() || (r.error ? String(r.error) : `${script} exited ${r.status}`) };
}

export default function (pi) {
  // A call the guard does not allow is blocked, and so is a call the guard
  // could not check: a missing script counts as a refusal.
  pi.on("tool_call", async (event, ctx) => {
    if (READERS.has(event.toolName)) return undefined;
    // bash is checked as a command; write, edit and any tool an extension
    // added are checked for the path they name.
    const script = event.toolName === "bash" ? "pretooluse-bash.sh" : "pretooluse-edit.sh";
    const r = guard(script, event, ctx);
    return r.ok ? undefined : { block: true, reason: r.why };
  });

  // One line per finished tool call in the audit log.  Never blocks.
  pi.on("tool_result", async (event, ctx) => {
    guard("posttooluse-audit.sh", event, ctx);
    return undefined;
  });
}
