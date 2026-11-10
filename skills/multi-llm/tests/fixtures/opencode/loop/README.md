# opencode 1.19.5 looping on a failed provider response

Two stdout heads of `opencode run --format json` against `tests/container/failserver.py`,
driven by the capture kit (`tests/fixtures/row_capture/`: offline, `--network none`, a dummy
key), captured on 2026-11-10 for the `truncated` and `error-envelope` rows. Each file is the first 64 lines,
32 steps, of its capture, byte for byte. stderr was empty in both.

| Row | Server mode | What the server sent | Whole capture |
| --- | --- | --- | --- |
| `truncated` | `trunc` | a stream cut before its finish chunk | killed at the 240 s deadline (exit 143); 3,887 requests, 3,886 steps, 2,770,718 bytes |
| `error-envelope` | `errbody` | HTTP 200 carrying an error body | killed at the 240 s deadline (exit 143); 3,456 requests, 3,454 steps, 2,463,006 bytes |

Every step is a `step_start` and a `step_finish` with `reason: "unknown"`, zero tokens and no
`text` or `tool_use` event between them. `run` leaves its session loop only on a finish other
than `tool-calls` or `unknown`, so it re-sends the request until something kills it. The
whole captures stay out of git (2.7 MB of identical steps). The manifest's two rows stay
not applicable, since no finished run exists to grade. The adapter's `LoopWatch` stops
both at step 20 (`opencode.LOOP_EMPTY_STEPS`), 0.7 s and 1.6 s into the stream by opencode's
own timestamps. `tests/test_opencode_loop_watch.py` reads these files. The contract is in
`docs/development/providers.md`, section "A CLI that loops".
