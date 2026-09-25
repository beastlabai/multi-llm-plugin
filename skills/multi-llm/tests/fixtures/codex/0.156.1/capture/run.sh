#!/bin/bash
# usage: run.sh NAME SCRIPT.json [extra codex args...]
set -u
D=$(cd "$(dirname "$0")" && pwd)
NAME=$1; SCRIPT=$2; shift 2
PORT=$((20000 + RANDOM % 20000))
W=$D/work/$NAME; rm -rf "$W"; mkdir -p "$W/home" "$W/tree"
printf 'line one\nline two\nline three\n' > "$W/tree/AGENTS.md"
(cd "$W/tree" && git init -q && git add . && git -c user.email=x@x -c user.name=x commit -qm init)
: > "$D/out/$NAME.stub.log"
python3 "$D/stub.py" $PORT "$SCRIPT" "$D/out/$NAME.stub.log" & SP=$!
sleep 0.5
cd "$W/tree"
CODEX_HOME="$W/home" STUB_KEY=dummy codex exec --json --skip-git-repo-check \
  -c model_provider=stub -c model_providers.stub.name=stub \
  -c model_providers.stub.base_url=http://127.0.0.1:$PORT/v1 \
  -c model_providers.stub.wire_api=responses -c model_providers.stub.env_key=STUB_KEY \
  "$@" 'Review the plan.' </dev/null > "$D/out/$NAME.jsonl" 2> "$D/out/$NAME.stderr"
echo "rc=$?"
kill $SP
