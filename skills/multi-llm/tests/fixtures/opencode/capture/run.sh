#!/bin/bash
# Re-capture the opencode permission fixtures (plan §4, T4.0/T4.6). $0, offline.
#
# usage: run.sh [--bin PATH_TO_OPENCODE] [CASE ...]
#   default binary: ~/.opencode/bin/opencode; output: ./<its version>/
#   no CASE = every case in capture.py's table.
#
# An older build can be pulled out of the npm cache without touching the
# network (memory: npm-cacache-is-an-opencode-build-history):
#   I=$(grep -rh 'opencode-linux-x64/-/opencode-linux-x64-1.17.15.tgz' ~/.npm/_cacache/index-v5 \
#       | grep -o '"integrity":"sha512-[^"]*"' | head -1 | sed 's/.*sha512-//;s/"$//')
#   H=$(python3 -c "import base64,sys;print(base64.b64decode(sys.argv[1]).hex())" "$I")
#   mkdir -p /tmp/oc-1.17.15 && tar xzf ~/.npm/_cacache/content-v2/sha512/${H:0:2}/${H:2:2}/${H:4} -C /tmp/oc-1.17.15
#   ./run.sh --bin /tmp/oc-1.17.15/package/bin/opencode
#
# Every run happens under a cleared environment with a scratch HOME/XDG tree,
# stdin /dev/null and OPENCODE_DISABLE_MODELS_FETCH=1; no `timeout` wrapper
# (each case ends on its own because the stub answers the follow-up with a
# final text; capture.py kills a run only past a 90 s deadline and records
# exit=KILLED). Nothing under the user's ~/.config/opencode, ~/.local/share/opencode
# or /etc is read or written.
set -eu
exec python3 "$(cd "$(dirname "$0")" && pwd)/capture.py" "$@"
