#!/bin/bash
# The whole 2026-09-25 capture session, in order. ~2.5 min wall, $0.
HERE=$(cd "$(dirname "$0")" && pwd)
run() { echo "=== $*"; "$HERE/run_row.sh" "$@" 2>&1 | tail -2; }
run pi success stub
run pi empty stub
run pi multi-assistant stub
run pi markers-in-tool-result stub
run pi truncated fail trunc
run pi error-envelope fail errbody
run pi exit-zero-failure fail 500
run pi auth-error fail 401
run pi rate-limit fail 429
run pi invalid-model fail 404
run pi context-overflow fail ctx
run pi zero-body fail empty
run pi server-error fail 500
run pi preflight-no-key stub "" 1
run pi length script length-full
run pi length-recovery script length-short
PART_BYTES=2000 PI_SETTINGS='{"compaction":{"keepRecentTokens":200}}' \
  run pi context-overflow-compacted script overflow-compact
PART_BYTES=2000 run omp compaction-summarized script overflow-compact
