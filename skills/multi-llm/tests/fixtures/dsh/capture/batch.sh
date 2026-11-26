#!/bin/bash
# Every dsh row of tests/fixtures/dsh/manifest.yaml, four containers at a time
# (README.md here). Exits 1 and lists the rows whose run_row.sh failed.
# ROWS="row row ..." limits the session to those rows.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${CAPTURE_OUT:-$HOME/.cache/multi-llm-capture/dsh}
SKILL=$(cd "$HERE/../../../.." && pwd)
KIT=$(cd "$HERE/../../row_capture" && pwd)
mkdir -p "$OUT/rows/dsh" "$OUT/scenarios"
cp "$HERE/incontainer.py" "$KIT/scriptserver.py" "$KIT/prompt.txt" "$OUT/"
cp -r "$KIT/scenarios/." "$OUT/scenarios/"
(cd "$SKILL" && uv run --project . -- python tests/fixtures/dsh/capture/render.py "$OUT" >/dev/null)
chmod -R a+rwX "$OUT"
export DSH_CAPTURE_STAGED=1

PLAN='plans/auth-feature.md'
rows=(
  "success|stub|"
  "empty-answer|stub|"
  "truncated|fail|trunc"
  "error-envelope|fail|errbody"
  "server-error|fail|500"
  "auth-error|fail|401"
  "rate-limit|fail|429"
  "invalid-model|fail|404"
  "context-overflow|fail|ctx"
  "zero-body|fail|empty"
  "ephemeral-only|fail|stall"
  "multi-assistant|script|read|{\"file_path\": \"/work/sample-project/$PLAN\"}"
  "write-denied|script|write|{\"file_path\": \"/work/sample-project/CANARY.txt\", \"content\": \"x\\n\"}"
  "write-outside-denied|script|write|{\"file_path\": \"/work/outside/CANARY.txt\", \"content\": \"x\\n\"}"
  "bash-denied|script|bash|{\"description\": \"canary\", \"command\": \"echo x > CANARY_SHELL.txt; echo y > /work/outside/CANARY_SHELL.txt; id; echo END\"}"
  "unsupported-effort|stub|"
)

run() {
  IFS='|' read -r row server mode args <<<"$1"
  local env=()
  case $row in
    empty-answer) env+=(SCENARIO=empty-answer) ;;
    unsupported-effort) env+=(SCENARIO=success REASONING_OVERRIDE=xhigh) ;;
    ephemeral-only) env+=(CAPTURE_DEADLINE_S=6) ;;
  esac
  if [ "$server" = script ]; then
    env+=(TOOL="$mode" TOOL_ARGS="$args")
    mode=""
  fi
  if env "${env[@]}" "$HERE/run_row.sh" "$row" "$server" $mode >"$OUT/rows/dsh/$row.run.log" 2>&1; then
    echo "ok   $row"
  else
    echo "FAIL $row"
    return 1
  fi
}
export -f run
export HERE OUT
only=" ${ROWS:-} "
for spec in "${rows[@]}"; do
  [ -z "${ROWS:-}" ] || [[ $only == *" ${spec%%|*} "* ]] && printf '%s\0' "$spec"
done | xargs -0 -P 4 -n 1 bash -c 'run "$1"' _ | tee "$OUT/batch.log"
! grep -q '^FAIL' "$OUT/batch.log"
