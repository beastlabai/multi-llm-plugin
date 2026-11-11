#!/bin/bash
# Host-side driver for the §6.2 row captures of codex, kilocode, kimi, opencode
# and qwen-code (README.md here). ONE container per (cli, row), --network none,
# no credential, the server started INSIDE the container by incontainer.py, and
# the CLI never wrapped in timeout(1): the outer `timeout` bounds `docker run`,
# and incontainer.py's own deadline (CAPTURE_DEADLINE_S, default 240) stops a CLI
# that retries forever and records `killed: true` in the row's meta.
#
#   run_row.sh <cli> <row> <stub|fail|script> [<failserver mode>]
#
# Env: MLLM_E2E_IMAGE (default mllm-e2e:local) is the image, as for the
# container lane; CAPTURE_OUT (default ~/.cache/multi-llm-capture/rows) is the
# scratch dir mounted at /work/out. The run flags are pi's capture kit's
# (tests/fixtures/pi/capture/run_row.sh), the provider lane's posture, with the
# workspace tmpfs mounted exec.
#
# Exit status: the capture's own (docker's, or the outer timeout's), never
# masked; a preparation failure stops before the launch. The row's previous
# files are deleted before the launch, and a failed run's meta is removed, so
# only a capture that exited 0 leaves `<row>.meta.json` (README.md here).
set -euo pipefail
if [ $# -lt 3 ] || [ $# -gt 4 ]; then
  echo "usage: run_row.sh <cli> <row> <stub|fail|script> [<failserver mode>]" >&2
  exit 2
fi
CLI=$1; ROW=$2; SERVER=$3; MODE=${4:-}
# The names become paths under $OUT/rows, and the deletions below use them.
for name in "$CLI" "$ROW"; do
  if [[ ! $name =~ ^[a-z0-9][a-z0-9-]*$ ]]; then
    echo "run_row.sh: '$name' is not a CLI or row name" >&2
    exit 2
  fi
done
HERE=$(cd "$(dirname "$0")" && pwd)
RIG=$(cd "$HERE/../../container" && pwd)
OUT=${CAPTURE_OUT:-$HOME/.cache/multi-llm-capture/rows}
ROWDIR=$OUT/rows/$CLI
# The kit's shared files are staged once: by batch.sh before its parallel runs
# (ROW_CAPTURE_STAGED=1), else here. A sibling run must never rewrite a file
# another container is reading, nor walk a tree a sibling is deleting from.
if [ -z "${ROW_CAPTURE_STAGED:-}" ]; then
  "$HERE/stage.sh" "$OUT"
fi
mkdir -p "$ROWDIR"
chmod a+rwX "$OUT/rows" "$ROWDIR"
for suffix in stdout.txt stderr.txt meta.json server.jsonl server.log calls.jsonl; do
  rm -f "$ROWDIR/$ROW.$suffix"
done
status=0
timeout --signal=KILL 900 docker run --rm --network none \
  --user 10001:10001 --cap-drop ALL --security-opt no-new-privileges --read-only \
  --pids-limit 512 --memory 4g --cpus 2.0 \
  --tmpfs /work/homes:rw,mode=0700,uid=10001,gid=10001 \
  --tmpfs /work/sample-project:rw,exec,mode=0755,uid=10001,gid=10001 \
  --tmpfs /tmp:rw,mode=1777 \
  -v "$RIG:/rig:ro" -v "$OUT:/work/out:rw" \
  -e MLLM_STUB_PORT=8099 -e MLLM_REDIRECT_YAML=/rig/redirect.yaml \
  -e MLLM_REAL_DIR=/opt/mllm/real -e MLLM_HOMES_ROOT=/work/homes \
  -e "MLLM_CALL_LOG=/work/out/rows/$CLI/$ROW.calls.jsonl" \
  -e CLI="$CLI" -e ROW="$ROW" -e SERVER="$SERVER" -e MODE="$MODE" \
  -e CAPTURE_DEADLINE_S="${CAPTURE_DEADLINE_S:-240}" \
  --entrypoint /bin/bash "${MLLM_E2E_IMAGE:-mllm-e2e:local}" -lc 'python3 /work/out/incontainer.py' \
  || status=$?
echo "docker exit: $status"
if [ "$status" -ne 0 ]; then
  rm -f "$ROWDIR/$ROW.meta.json"
  echo "run_row.sh: $CLI $ROW failed (exit $status); no meta kept" >&2
fi
exit "$status"
