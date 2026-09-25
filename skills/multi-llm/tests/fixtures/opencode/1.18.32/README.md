# opencode 1.18.32 denied-turn fixtures

Hand-graded `.jsonl` captures, consumed by
`tests/test_providers.py::TestOpenCodeStreamCapturesAcrossThePin` and
`TestOpenCodeDeniedTurn`, and by the `dispatch_task`-level test in
`tests/test_subagent_executor.py`. opencode is still in `MANIFEST_PENDING`
(`tests/test_provider_fixture_rows.py`), so these rows are graded by hand rather
than through a `manifest.yaml`. The stdout files are `.jsonl`, and this
directory is one level below `tests/fixtures/opencode/`, so
`test_stderr_diagnosis._capture_rows` (which globs `*.txt` in the provider
directory) does not pick them up.

The plan is `plans/opencode-validator-false-rate-limit-and-denied-turn.md`
(§2.3 T2.3, §4.2 step 4, T4.3).

| Fixture | Provenance | What it pins |
| --- | --- | --- |
| `denied_permission_turn_end` | **synthesized from a real run, redacted** | the b38de8bb signature: exit 0, the last `step_finish` has `part.reason: "tool-calls"`, a bash `tool_use` has `status: error` with "The user rejected permission to use this specific tool call.", and stderr has only the `auto-rejecting` line. Tool output quotes plan text containing `rate limit`, `rate_limit` and `429`. `classify_failure` is `runner_error`, never `rate_limited`. Before `vendor_text` it was `rate_limited` |
| `continue_loop_on_deny` | **measured**, byte for byte | the pinned negative. With `experimental.continue_loop_on_deny: true`, stderr still carries `auto-rejecting`, but the turn goes on to `FINAL ANSWER` and a final `part.reason: "stop"`. The structural check must answer `None` |

## Where `denied_permission_turn_end` came from

Run `b38de8bb-82c0-4570-a9f7-9b1434294e1a` (2026-09-24, opencode 1.18.32,
`opencode run --agent plan --format json --print-logs --log-level ERROR --pure`),
validator attempt `failed/validation-batch-0.attempt1.json`. That record
recorded the attempt `rate_limited`.

- **Whole events only.** The stored `stdout_tail` is the last 64 KB of the
  stream, and its first line is cut at its start. That line is dropped. The
  twelve whole events after it are kept, in order, with their `type`s, key
  order, timestamps, token counts and cost.
- **Redaction by value replacement**, following `tests/fixtures/cursor-agent/README.md`:
  - every session, part, message and call ID is zeroed to the same shape
    (`ses_000…`, `prt_000…N`, `msg_000…N`, `call_000…N`);
  - snapshot hashes are forty zeros;
  - repository paths became neutral stand-ins (`src/db/schema/10-core.sql`,
    `docs/plan.md`);
  - SQL, plan text, model prose and the model's reasoning metadata became
    neutral sentences. Those sentences deliberately still contain `rate limit`,
    `rate_limit` and `429`, because the defect is that such text reached the
    classifier.
- **Kept verbatim:** the rejected bash `tool_use` (a generic probe of
  `/var/run/postgresql`, `/etc/postgresql` and `psql`, with no repository
  content), both `step_finish reason: "tool-calls"` events, and the stderr file,
  which is the run's whole stderr: one `auto-rejecting` line behind ANSI colour
  codes.

## Where `continue_loop_on_deny` came from

It is the `cont` case of the offline reproduction in
`plans/opencode-validator-false-rate-limit-and-denied-turn/investigation/oc/`
(`run.sh`, `stub.py`). The real `~/.opencode/bin/opencode` 1.18.32 ran against a
localhost chat-completions stub, with `--agent plan --pure`, a scratch
HOME/XDG, `OPENCODE_CONFIG_CONTENT` carrying
`experimental.continue_loop_on_deny: true`, and the prompt
`please read /etc/hostname`. Exit 0. There were no model calls and nothing was
spent. The files are that run's `stdout.jsonl` and `stderr.txt`, unchanged.
