# Codex 0.163.0 offline recapture — 2026-10-22

The installed `codex-cli 0.163.0` emitted these four streams against the
repository's localhost Responses stub. Each turn used a fresh throwaway
`CODEX_HOME` and scratch git repository, `--model gpt-5.5 --sandbox
danger-full-access`. The model name selects the CLI's tool schema; every
response was scripted locally, with no API spend.

The unchanged `../0.156.1/capture/` rig was copied to
`/tmp/mllm-codex-compat-2026-10-22/capture`, with an `out/` directory created.
For each pair below, the invocation was:

```bash
HISTFILE=/dev/null timeout 180 bash /tmp/mllm-codex-compat-2026-10-22/capture/run.sh NAME /tmp/mllm-codex-compat-2026-10-22/capture/SCRIPT --model gpt-5.5 --sandbox danger-full-access
```

- `read_only_clean` / `script_r.json`
- `quoting_render` / `script_q.json`
- `writes_patches_sessions` / `script_a.json`
- `exec_session` / `script_s.json`

All four invocations reported `rc=0` and ended with `turn.completed` and an
`agent_message` containing `[]`. Event kinds and top-level/item keys are a
subset of the corresponding 0.156.1 captures (the older deprecation notice
is absent). The read-only audit accepts the clean reads and rejects the
other three streams, including the shell expansion in `quoting_render`.

Normalization: scratch tree/home prefixes became `/work/tree` and
`/work/home`; the two interactive bash outputs' abbreviated machine-specific
prompts became `user@host:/work/tree$ `. Thread IDs and scratch commit hashes
remain as emitted. No event, status, command or final answer was invented.

The temporary home produces a stderr warning about refusing PATH aliases
under `/tmp`; all scripted turns still completed. These files are stdout.
Collaboration was not rerun (no nested delegation); hosted tools and code mode
remain covered by their historical captures, not a new 0.163.0 measurement.

Separately, all 12 rows of `tests/test_codex_jail_rig.py` passed on 0.163.0
with `MULTI_LLM_REQUIRE_LANDLOCK=1`, including strace controls and the real
production wrapper. That evidence raises the jail measurement, not its
allow-list or its incidental protection grade. Full commands and results:
`plans/2026-10-22-three-host-racing-run/CODEX-COMPATIBILITY.md`.
