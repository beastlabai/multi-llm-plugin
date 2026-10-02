#!/bin/bash
# Re-capture the opencode rig fixtures (plan §4, T4.0/T4.6, and the parser's
# stream cases). $0, offline.
#
# usage: run.sh [--bin PATH_TO_OPENCODE] [CASE ...]
#   default binary: ~/.opencode/bin/opencode; output: ./<its version>/
#   no CASE = every case in capture.py's table.
#
# Another build than the installed one: `make opencode-bump VERSION=<x.y.z>`
# (packaging/opencode/bump.py) installs it under a scratch npm prefix and passes
# --bin. A build already in the npm cache can also be pulled out without
# touching the network:
#   V=<x.y.z>
#   I=$(grep -rh "opencode-linux-x64/-/opencode-linux-x64-$V.tgz" ~/.npm/_cacache/index-v5 \
#       | grep -o '"integrity":"sha512-[^"]*"' | head -1 | sed 's/.*sha512-//;s/"$//')
#   H=$(python3 -c "import base64,sys;print(base64.b64decode(sys.argv[1]).hex())" "$I")
#   mkdir -p /tmp/oc-$V && tar xzf ~/.npm/_cacache/content-v2/sha512/${H:0:2}/${H:2:2}/${H:4} -C /tmp/oc-$V
#   ./run.sh --bin /tmp/oc-$V/package/bin/opencode
#
# Every run happens under a cleared environment with a scratch HOME/XDG tree,
# stdin /dev/null and OPENCODE_DISABLE_MODELS_FETCH=1; no `timeout` wrapper
# (each case ends on its own because the stub answers the follow-up with a
# final text; capture.py kills a run only past a 90 s deadline and records
# exit=KILLED). Nothing under the user's ~/.config/opencode, ~/.local/share/opencode
# or /etc is read or written.
set -eu
exec python3 "$(cd "$(dirname "$0")" && pwd)/capture.py" "$@"
