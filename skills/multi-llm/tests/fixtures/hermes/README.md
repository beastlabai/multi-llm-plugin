# hermes fixtures and capture kit

Raw `hermes chat --format stream-json` captures from Hermes Agent v0.23.4,
graded by `tests/test_provider_fixture_rows.py` through `manifest.yaml`.
Offline and $0: no credential, `--network none`, the servers inside the
container.

| File | What it is |
| --- | --- |
| `probe.Dockerfile` | the probe image: the `mllm-e2e` image plus Hermes from the v0.23.4 release tarball (sha256-checked), `uv sync --locked` on a uv-managed Python 3.14 (the lockfile needs it), and an install stamp naming the tag commit so `hermes --version` prints `v0.23.4` instead of `vunknown`. Build it from a directory holding `hermes-v0.23.4.tar.gz` (`gh api repos/NousResearch/hermes-agent/tarball/v0.23.4 > hermes-v0.23.4.tar.gz`): `docker build -f probe.Dockerfile -t mllm-hermes-probe:0.23.4 .` |
| `capture.sh <row> <stub\|fail\|script> <mode> [options]` | host side: one container per row, read-only root, uid 10001; writes `$CAPTURE_OUT/<row>.json` (default `~/.cache/multi-llm-capture/hermes`, since Docker Desktop cannot mount `/tmp`) |
| `capture_probe.py` | in the container: imports the production adapter from the mounted skill, swaps only `hermes.BASE_URL` for the local server, and runs one dispatch exactly as a spawn site does (`private_home.dispatch_homes`, `hermetic_dispatch`, `resolve_env`, `build_child_env`, the stdin transport). It records the streams, request counts, whether the gateway daemon was still running after the dispatch closed, whether the private root was removed and what the workspace holds |
| `toolserver.py` | the tool-turn server: `multi` (decoy text plus a tool call, then the findings), `write` / `terminal` (forced calls the toolless lane must refuse), `loop` (a tool call on every turn) |
| `prompt.txt` | the 4,844-byte review prompt the row-capture kit uses |

The rows' stdout and stderr were copied out of the JSON byte for byte; the
manifest's `notes` say which server and mode produced each.
