#!/bin/sh
# Claude Code plugin PreToolUse hook (fix round 26, live run 11 #2): refuse a
# scheduled wakeup while a multi-llm phase this session started is running.
#
# hooks.json runs this for the `ScheduleWakeup` and `CronCreate` tools only.
# A headless host waiting on a phase's background subagents scheduled a
# wakeup whose prompt re-ran the phase: the `claude -p` process stayed alive
# after its final result, and the wakeup would have run the phase again.
# `multi_llm.py util wakeup-check` reads the session's transcript and, while a
# phase it started has printed no next step yet, prints one
# `{"hookSpecificOutput": {... "permissionDecision": "deny" ...}}` line whose
# reason tells the model to keep waiting for its subagents instead.
#
# It never affects the user's own wakeups, and it fails open: this script
# exits 0 on every path and prints nothing unless the check refuses.
#
# - A tool other than the two exits at once (the matcher already says so).
# - Claude Code writes the transcript in batches, so when this hook starts the
#   call it asks about, and a phase call made in the same message, may not be
#   on file yet (fix round 27). The script first waits for the call's own
#   `tool_use_id` to be on file, at most 2 s (20 steps of 0.1 s; at once
#   where `sleep` takes no fraction), and the util waits the same way for the
#   results of the phase calls that message made before it.
# - A transcript with no `multi_llm.py` in it then exits at once too, before
#   uv is looked up: a session that never ran the dispatcher pays that wait
#   (0.1-0.2 s measured on 2.1.293) and one grep.
# - The util's stdout also carries the dispatcher's `MULTI_LLM_RESULT` line,
#   and Claude Code parses hook stdout as JSON only when the whole of it is
#   JSON, so only the decision line (`{"hookSpecificOutput"...`) is passed on.
# - Nothing here touches the network.
#
# uv is looked for the way `utils/unattended.find_uv()` and every rendered shim
# look for it: $MULTI_LLM_UV, then PATH, then the documented install locations.
# Silent when none resolves.

input="$(cat 2>/dev/null)"

printf '%s\n' "$input" \
  | grep -q -E '"tool_name"[[:space:]]*:[[:space:]]*"(ScheduleWakeup|CronCreate)"' \
  || exit 0

# The transcript path, as the hook input spells it (see report-check.sh).
transcript="$(printf '%s\n' "$input" \
  | sed -n 's/.*"transcript_path"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
  | sed -e 's/\\\\/\\/g' -e 's#\\/#/#g' | head -n 1)"
[ -n "$transcript" ] || exit 0
# The call's own id, quoted as the transcript writes it; with none, the wait
# is for the file alone.
tool_use_id="$(printf '%s\n' "$input" \
  | sed -n 's/.*"tool_use_id"[[:space:]]*:[[:space:]]*"\([A-Za-z0-9_-]*\)".*/\1/p' | head -n 1)"
i=0
while [ "$i" -lt 20 ]; do
  if [ -f "$transcript" ]; then
    [ -z "$tool_use_id" ] && break
    grep -q -F "\"$tool_use_id\"" "$transcript" 2>/dev/null && break
  fi
  sleep 0.1 2>/dev/null || break
  i=$((i + 1))
done
[ -f "$transcript" ] || exit 0
grep -q -F 'multi_llm.py' "$transcript" 2>/dev/null || exit 0

root="${CLAUDE_PLUGIN_ROOT:-$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)}"
skill="$root/skills/multi-llm"
[ -f "$skill/multi_llm.py" ] || exit 0

uv=""
if [ -n "$MULTI_LLM_UV" ] && [ -f "$MULTI_LLM_UV" ]; then
  uv="$MULTI_LLM_UV"
elif command -v uv >/dev/null 2>&1; then
  uv="$(command -v uv)"
else
  # One line, in probe order: tests/test_uv_candidates.py holds it against
  # `unattended.UV_CANDIDATES` element for element.
  for candidate in "$HOME/.local/bin/uv" "$HOME/.local/bin/uv.exe" "$HOME/.cargo/bin/uv" "$HOME/.cargo/bin/uv.exe" /opt/homebrew/bin/uv /usr/local/bin/uv /opt/local/bin/uv; do
    if [ -f "$candidate" ]; then
      uv="$candidate"
      break
    fi
  done
fi
[ -n "$uv" ] || exit 0

# `--frozen --no-dev` is `utils/unattended.uv_run_args()`'s default list, spelled
# out here for the same reason doctor.sh spells it out: nothing renders this file.
out="$(printf '%s\n' "$input" \
  | "$uv" run --frozen --no-dev --project "$skill" -- python "$skill/multi_llm.py" util wakeup-check --hook-input - 2>/dev/null)"
printf '%s\n' "$out" | grep '^{"hookSpecificOutput"' | head -n 1
exit 0
