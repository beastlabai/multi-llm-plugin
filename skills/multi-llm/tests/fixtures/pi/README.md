# pi parser fixtures — `--mode json`, pi 0.87.1

Every `*.stdout.txt` / `*.stderr.txt` at this level is a **real capture** of pi
0.87.1's streams under `pi --no-session -p --mode json --model stub/stub-model
<prompt>`, taken on 2026-09-25 in the container (`mllm-e2e:local`,
`--network none`, **no credential**, one container per row, the server running
inside it, the CLI never wrapped in `timeout(1)`). Raw bytes, nothing scrubbed,
nothing synthesized. `manifest.yaml` is the authority on what each row asserts
and is graded by `tests/test_provider_fixture_rows.py`; this file says why the
set looks the way it does.

| Directory | What it is |
| --- | --- |
| `.` | the graded json-mode set (this file, `manifest.yaml`, the captures) |
| `capture/` | the rig that produced it: `run_row.sh` (host side), `incontainer.py`, `scriptserver.py` (the throwaway server for `length`, `length-recovery`, `context-overflow-compacted` and omp's `compaction-summarized`), the scenarios, the argv (`argv.json`, asserted equal to `PiProvider.build_command`) and the prompt. `batch.sh` reruns the whole session (~2.5 min, $0) |
| `json-mode/` | the 2026-09-25 **live** usage probe (one OpenRouter call, scrubbed) — the only row carrying real usage and cost |
| `text-mode-0.80.6/` | the historical 0.80.6 text-mode set, **ungraded** (see its README) |

## What changed from 0.80.6, row by row

| Row | 0.80.6 text mode | 0.87.1 json mode | class then → now |
| --- | --- | --- | --- |
| `success` | 151 B bare answer | stream; answer = last assistant `message_end` text | none → none |
| `auth-error` | stderr `401: {…}`, exit 1 | `error`, `errorMessage` `401: {…}`, **exit 0**, stderr empty | auth_failed → auth_failed |
| `rate-limit` | stderr `429: {…}`, 4 req | `error` after `auto_retry_*` ×3, 4 req, exit 0 | rate_limited → rate_limited |
| `server-error` | stderr `500: {…}`, 4 req | `error` `500:`, 4 req, exit 0 | **none → server_error** |
| `invalid-model` | stderr `404: {…}` | `error` `404: {…}` | invalid_model → invalid_model |
| `truncated` / `error-envelope` | stderr `Stream ended without finish_reason`, exit 1 | same sentence as `errorMessage`, exit 0 | none → none (recorded) |
| `zero-body` | stderr `Connection error.` | same, as `errorMessage` | connection_error → connection_error |
| `context-overflow` | **nothing on either channel, exit 0** | `error` with the vendor's `maximum context length` 400 | **none → context_length_exceeded** |
| `empty` | nothing, exit 0 | `stop` with `content: []` | none → none |
| `exit-zero-failure` | (ctx) nothing, exit 0 | (500) every in-stream failure exits 0 | none → server_error |
| `preflight-no-key` (new) | — | header-only stdout, stderr `No API key found for stub.`, exit 1 | → auth_failed |
| `length` (new) | not capturable | `stopReason: length` | → none, `Truncated output` |
| `length-recovery` (new) | not capturable | a recoverable length with nothing to compact: same as `length` | → none |
| `multi-assistant` (new) | not applicable | `toolUse` then `stop` | → none |
| `markers-in-tool-result` (new) | not capturable | a clean run whose tool result quotes 401 / 429 / rate limit | → none, **only through `vendor_text`** |
| `context-overflow-compacted` (new) | not capturable | 400 → compaction (one paid summarization, `result.usage`) → retry → answer | → none (a success) |

No cell got worse; `tests/test_stderr_diagnosis.py::TestNoPiClassifyCellWorsened`
pins that against the 0.80.6 literal.

## The four facts the parser rests on

1. **Success is structural.** Every in-stream failure exits 0 with stderr empty,
   so `PiProvider.parse_output` reads the terminal assistant `message_end`'s
   `stopReason` and never the exit code.
2. **The status lives only in the `errorMessage` prefix.** pi's assistant message
   has no `errorStatus` (omp's does), so `401`/`429`/`5xx` are parsed from
   `NNN: {json}` — the string 0.80.6 wrote to stderr.
3. **A thrown pre-flight is the one non-zero exit**, and it leaves the `session`
   header on stdout: `describe_failure` treats a header-only stream as silent and
   answers from stderr (omp's `_never_started`).
4. **The stream carries our prompt and pi's system prompt**, so
   `PiProvider.vendor_text` hands the shared tables the terminal message's
   `errorMessage` and nothing earlier.

## Retry and compaction numbers (T7.14, T7.15)

- Every retryable row (429, 500, cut stream, 200-with-error, zero-byte) is **4
  requests over ~14.4 s** (`auto_retry_start` at 2/4/8 s, `maxAttempts: 3`), 401
  and 404 are 1. That is `PiProvider.internal_retries`.
- `context-overflow-compacted` is 7 requests: 4 tool turns, the 400, **one**
  summarization (no tools, `max_tokens` 8192), the retried answer. Its usage is
  `compaction_end.result.usage`, which `parse_usage` adds.
- omp's twin row (`capture/` `compaction-summarized`, committed as
  `tests/fixtures/omp/compaction-summarized.*`) is 8 requests: omp's compaction
  made **two** summarization calls (8192 and 512 `max_tokens`) and its
  `auto_compaction_end.result` carries **no** usage.
