#!/bin/bash
# Host-side driver for the pi json-mode capture set (and omp's compaction row):
# ONE container per (cli, row), --network none, no credential, the server started
# INSIDE the container by incontainer.py, and the CLI never wrapped in timeout(1)
# (the outer `timeout` bounds `docker run`, not the CLI).
#
#   run_row.sh <cli> <row> <stub|fail|script> [<mode>] [<drop_api_key 0|1>]
#
# Env: MLLM_E2E_IMAGE (default mllm-e2e:local) is the image, as for the
# container lane; CAPTURE_OUT (default ~/.cache/multi-llm-capture/pi-json) is the scratch
# dir mounted at /work/out; PART_BYTES sizes big/partN.txt for the compaction
# rows; PI_SETTINGS is written to the per-exec .pi/agent/settings.json.
# prompt.txt is the 4844-byte stdout-demanding review prompt the 0.80.6 set used.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
RIG=$(cd "$HERE/../../../container" && pwd)
OUT=${CAPTURE_OUT:-$HOME/.cache/multi-llm-capture/pi-json}
mkdir -p "$OUT/rows" "$OUT/scenarios"
cp "$HERE/incontainer.py" "$HERE/scriptserver.py" "$HERE/argv.json" "$HERE/prompt.txt" "$OUT/"
cp "$HERE/scenarios/"*.yaml "$OUT/scenarios/"
cp "$RIG/../../assets/omp/overlay.yml" "$OUT/overlay.yml"
chmod -R a+rwX "$OUT"
CLI=$1; ROW=$2; SERVER=$3; MODE=${4:-}; DROP=${5:-0}
timeout --signal=KILL 600 docker run --rm --network none \
  --user 10001:10001 --cap-drop ALL --security-opt no-new-privileges --read-only \
  --pids-limit 512 --memory 4g --cpus 2.0 \
  --tmpfs /work/homes:rw,mode=0700,uid=10001,gid=10001 \
  --tmpfs /work/sample-project:rw,mode=0755,uid=10001,gid=10001 \
  --tmpfs /tmp:rw,mode=1777 \
  -v "$RIG:/rig:ro" -v "$OUT:/work/out:rw" \
  -e MLLM_STUB_PORT=8099 -e MLLM_REDIRECT_YAML=/rig/redirect.yaml \
  -e MLLM_REAL_DIR=/opt/mllm/real -e MLLM_HOMES_ROOT=/work/homes \
  -e MLLM_CALL_LOG=/work/out/rows/$CLI/calls.jsonl \
  -e CLI="$CLI" -e ROW="$ROW" -e SERVER="$SERVER" -e MODE="$MODE" -e DROP_API_KEY="$DROP" \
  -e PART_BYTES="${PART_BYTES:-}" -e PI_SETTINGS="${PI_SETTINGS:-}" \
  --entrypoint /bin/bash "${MLLM_E2E_IMAGE:-mllm-e2e:local}" -lc 'python3 /work/out/incontainer.py'
echo "docker exit: $?"
