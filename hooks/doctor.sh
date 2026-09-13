#!/bin/sh
# Claude Code plugin SessionStart hook (D6 step 5): `multi_llm.py doctor --warn-only`.
#
# Advisory, never blocking. `--warn-only` exits 0 whatever it finds, and this
# script exits 0 on every path of its own — a SessionStart hook that exits 2
# would block the session, and a missing uv is exactly the machine where the
# doctor's own message would be the least useful thing to block on. The check
# is re-run per session rather than stored per repo because a verdict taken at
# install time goes stale with the next CLI or kernel update.
#
# uv is looked for the way `utils/unattended.find_uv()` and every rendered shim
# look for it: $MULTI_LLM_UV, then PATH, then the documented install locations
# (`hosts/uv-missing.txt`). Silent when none resolves: the first real command
# the user runs prints that file with exit 3, which is the right place for it.

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
# out here for the same reason SKILL.md spells it out: nothing renders this file.
"$uv" run --frozen --no-dev --project "$skill" -- python "$skill/multi_llm.py" doctor --warn-only 2>/dev/null
exit 0
