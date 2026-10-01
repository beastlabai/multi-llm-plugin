#!/bin/sh
# Claude Code plugin Stop hook (plans/next-phase-command.md T11): the report check.
#
# A phase result prints the next step as `[NEXT_COMMAND] <command>` lines (or
# `[WORKFLOW_COMPLETE]`), and the report has to end with each command in a
# fenced block, copied exactly. `multi_llm.py util report-check` compares the
# turn's final message with the last such block the turn printed and, on a
# miss, prints one `{"decision": "block", "reason": ...}` line, which gives the
# model one more turn to add the command.
#
# Advisory, never breaking: this script exits 0 on every path and prints
# nothing unless the check blocks, as doctor.sh does.
#
# - `stop_hook_active` true means this turn was already blocked once, so the
#   script exits at once: it blocks at most once per turn and cannot loop.
# - A transcript with no `[NEXT_COMMAND` and no `[WORKFLOW_COMPLETE]` text
#   exits at once too, before uv is looked up: a session that never ran a
#   phase pays one grep.
# - The util's stdout also carries the dispatcher's `MULTI_LLM_RESULT` line,
#   and Claude Code parses hook stdout as JSON only when the whole of it is
#   JSON. So only the decision line (`{"decision"...`) is passed through.
#
# uv is looked for the way `utils/unattended.find_uv()` and every rendered shim
# look for it: $MULTI_LLM_UV, then PATH, then the documented install locations.
# Silent when none resolves.

input="$(cat 2>/dev/null)"

# Loop guard first: a turn this hook already blocked is let through.
if printf '%s\n' "$input" | grep -q '"stop_hook_active"[[:space:]]*:[[:space:]]*true'; then
  exit 0
fi

# The transcript path, as the hook input spells it. `\\` and `\/` are the only
# JSON escapes a path is likely to carry; a path this cannot recover is simply
# not found, which is a silent pass. The util re-reads it with a JSON parser.
transcript="$(printf '%s\n' "$input" \
  | sed -n 's/.*"transcript_path"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
  | sed -e 's/\\\\/\\/g' -e 's#\\/#/#g' | head -n 1)"
[ -n "$transcript" ] && [ -f "$transcript" ] || exit 0
grep -q -F -e '[NEXT_COMMAND' -e '[WORKFLOW_COMPLETE]' "$transcript" 2>/dev/null || exit 0

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
  | "$uv" run --frozen --no-dev --project "$skill" -- python "$skill/multi_llm.py" util report-check --hook-input - 2>/dev/null)"
printf '%s\n' "$out" | grep '^{"decision"' | head -n 1
exit 0
