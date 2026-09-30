# `jail_writes.out.jsonl` — a jailed codex turn whose every write was denied

The `codex exec --json` stream of the plugin-landlock preflight's `jail_writes` turn (plan `codex-sandbox-unavailable-findings-and-fixes`, preflight step 1), measured 2026-09-29 on codex 0.158.0 through the preflight rig (`run_step1.py` driving the real codex against the localhost stub, inside the step-1 wrapper, throwaway `CODEX_HOME`); `tests/test_landlock_dispatch.py` reads it for `jail_denied_edits`.
Recorded verbatim except three path prefixes, replaced consistently: the rig's scratch dir is `/tmp/preflight-scratch/step1/jail_writes` (so the turn's cwd is `…/tree`), another session's scratch dir is `/tmp/other-session-scratch`, and the user's home is `/home/user`.
