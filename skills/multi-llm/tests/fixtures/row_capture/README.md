# Row-capture kit: §6.2 rows for codex, kilocode, kimi, opencode and qwen-code

The rig that produced the `rows/` captures graded by
`tests/fixtures/<provider>/manifest.yaml` for these five adapters
(`tests/test_provider_fixture_rows.py`). It is pi's capture kit
(`tests/fixtures/pi/capture/`) generalised to five CLIs. **Offline and $0**: one
container per (CLI, row) from the pinned image (`$MLLM_E2E_IMAGE`, default
`mllm-e2e:local`, the tag `make e2e-image` builds at the `cli-versions.lock`
pins), `--network none`, uid 10001, read-only root, no credential (the only key
anywhere is the PATH wrapper's per-exec `mllm-stub-<id>`, meaningful only on
127.0.0.1), the server running inside the container, and the CLI never wrapped
in `timeout(1)`.

| File | What it is |
| --- | --- |
| `run_row.sh <cli> <row> <stub\|fail\|script> [<mode>]` | host side: the `docker run` (pi's kit's flags, the provider lane's posture). It deletes the row's previous files before the launch, exits with the launch's status, and removes the meta of a run that exited non-zero, so only a capture that exited 0 leaves a `<row>.meta.json` |
| `incontainer.py` | in the container: seeds `/work/homes/<cli>` from the image's `home_seed`, makes `/work/sample-project` a git repository, starts the server, runs the CLI through `/opt/mllm/bin/<cli>` (the wrapper, so `redirect.yaml`'s row applies), and writes `<row>.stdout.txt`, `<row>.stderr.txt`, `<row>.meta.json` (exit code, wall time, request count), the server's request log and the wrapper's call log (`<row>.calls.jsonl`) |
| `argv.json`, `make_argv.py` | each CLI's argv: its adapter's `build_command` for `redirect.yaml`'s `model_arg` and a 230 s dispatch timeout (only qwen-code renders it, as `--max-wall-time 200`), rendered against the image's binaries rather than the host's PATH (kilocode's adapter would name `kilo` on a `kilo`-only host). `tests/test_row_capture_kit.py` holds the file to what the adapters render |
| `scenarios/` | `stub_server.py` scenarios: `success` (the two-finding array every manifest here pins) and `empty-answer` (the empty string) |
| `scriptserver.py`, `scripts.json` | the `multi-assistant` server: the tool turn is answered with a decoy text (`FIRST ANSWER (must NOT be selected)` and a `D999` array) **and** one call of the CLI's own read tool (`scripts.json`), the request carrying its result with the findings. The committed stub's `scripted_tool_call` answers that turn with a bare call and no text, which leaves one text message and says nothing about last-wins |
| `batch.sh [<cli> ...]` | the whole session, the five CLIs of one row in parallel (~20 min); exits 1 and lists the runs when any `run_row.sh` failed |
| `prompt.txt` | the 4,844-byte review prompt pi's set uses |

Rows and servers (`batch.sh`): `success`, `empty-answer` (stub);
`multi-assistant` (script); and failserver's `trunc` (`truncated`), `errbody`
(`error-envelope`), `500` (`server-error`), `401` (`auth-error`), `429`
(`rate-limit`), `404` (`invalid-model`), `ctx` (`context-overflow`) and `empty`
(`zero-body`). Each manifest maps them onto the twelve canonical rows and says
which capture a row uses, or why a row is not applicable.

A CLI still running at `CAPTURE_DEADLINE_S` (default 240) gets SIGTERM through
the wrapper, then SIGKILL 10 s later, and its meta records `killed: true`. A
killed run is a harness artifact, not a CLI behaviour, so no manifest grades
one; the row that would have used it is measured not-applicable instead.

Output goes to `$CAPTURE_OUT` (default `~/.cache/multi-llm-capture/rows`):
Docker Desktop refuses to bind-mount a directory under `/tmp`. The captures
are copied into `tests/fixtures/<provider>/rows/` by hand, byte for byte; the
paths the wrapper prints are container paths (`/tmp/mllm-inv/…`,
`/work/sample-project`), and nothing from the host is in them.
