#!/bin/bash
# The whole capture session (first run 2026-11-10, ~20 min wall, $0): every row
# for every CLI this kit covers, or the CLIs named as arguments. The five CLIs
# of one row run in parallel containers, each with its own loopback.
#
# Exit status: 1 when any run_row.sh exited non-zero (each one is listed on
# stderr with its status and its output's last lines), else 0.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
read -r -a CLIS <<< "${*:-codex kilocode kimi opencode qwen-code}"
ROWS=(
  "success stub" "empty-answer stub" "multi-assistant script"
  "truncated fail trunc" "error-envelope fail errbody" "server-error fail 500"
  "auth-error fail 401" "rate-limit fail 429" "invalid-model fail 404"
  "context-overflow fail ctx" "zero-body fail empty"
)
LOGS=$(mktemp -d) || exit 1
trap 'rm -rf "$LOGS"' EXIT
# Stage the shared files once, before any run starts (run_row.sh then skips it).
"$HERE/stage.sh" "${CAPTURE_OUT:-$HOME/.cache/multi-llm-capture/rows}" || exit 1
export ROW_CAPTURE_STAGED=1
failed=()
for row in "${ROWS[@]}"; do
  read -r -a args <<< "$row"
  pids=(); runs=()
  for cli in "${CLIS[@]}"; do
    "$HERE/run_row.sh" "$cli" "${args[@]}" > "$LOGS/$cli.${args[0]}.log" 2>&1 &
    pids+=("$!"); runs+=("$cli ${args[0]}")
  done
  for i in "${!pids[@]}"; do
    status=0
    wait "${pids[$i]}" || status=$?
    log="$LOGS/${runs[$i]/ /.}.log"
    echo "=== ${runs[$i]} (exit $status): $(tail -2 "$log" | head -1 | cut -c1-200)"
    if [ "$status" -ne 0 ]; then
      failed+=("${runs[$i]}")
      { echo "--- ${runs[$i]} failed (exit $status):"; tail -5 "$log"; } >&2
    fi
  done
done
if [ "${#failed[@]}" -gt 0 ]; then
  echo "batch.sh: ${#failed[@]} run(s) failed: $(IFS=,; echo "${failed[*]}")" >&2
  exit 1
fi
