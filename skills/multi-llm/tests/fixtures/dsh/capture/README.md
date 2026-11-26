# dsh capture kit

How `tests/fixtures/dsh/` was produced. Offline and $0: one container per row
from this directory's `Dockerfile` (`node:24-bookworm-slim` plus
`@deepseek-ai/dsh@0.2.2-rc.5`; it predates the e2e image's dsh layer and
stays because its rows need a server per row and a forced tool call),
`--network none`, uid 10001, a read-only root, `--cap-drop ALL`, no credential.

| File | What it is |
| --- | --- |
| `Dockerfile` | the capture image: `docker build -t mllm-dsh-capture:0.2.2-rc.5 .` |
| `render.py <out>` | renders the production adapter's stdin argv, `get_env` and `DSH_PROFILE_PATCH` into `<out>/dsh/`, with only the patch's `baseURL` pointed at `http://127.0.0.1:8099/v1` |
| `incontainer.py` | in the container: a git workspace, one server (`tests/container/stub_server.py` with a `row_capture/scenarios` file, `tests/container/failserver.py --mode`, or `row_capture/scriptserver.py --tool`), a fresh `DSH_HOME` holding the rendered patch, then the rendered argv with `row_capture/prompt.txt` on stdin. Writes `<row>.stdout.txt`, `<row>.stderr.txt`, `<row>.meta.json` (exit code, wall time, request count, what changed in the workspace and in `/work/outside`) and the server log |
| `run_row.sh <row> <stub\|fail\|script> [<mode>]` | host side: the `docker run` for one row |
| `batch.sh` | every row, four containers at a time (`ROWS="a b"` limits it) |

Output goes to `$CAPTURE_OUT` (default `~/.cache/multi-llm-capture/dsh`;
Docker Desktop refuses `/tmp` mounts). The streams are copied into
`tests/fixtures/dsh/` byte for byte, and the metas are collected into
`captures.json`. Nothing from the host is in them: the paths are container
paths and the session ids are dsh's own.

The tool rows (`write-denied`, `write-outside-denied`, `bash-denied`) are the
read-only measurement in this container: `scriptserver.py` forces one call,
and `captures.json` records an unchanged workspace and outside directory.

A live check outside this kit, 2026-11-26: the production patch unchanged
(`https://openrouter.ai/api/v1`), the spend-capped e2e key, two calls to
`deepseek/deepseek-v4.1-flash-1017` (`:high` and no effort) from the same
image with the network on. Both read `calc.py` with the `read` tool and
answered the JSON array; the `:high` run streamed a `thinking` record, and
the second step of each reported `cacheReadTokens` beside `inputTokens`.
