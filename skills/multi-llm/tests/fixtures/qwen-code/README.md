# qwen-code parser fixtures

Every `*.stdout.json` here is a **real capture** of `qwen -o json` stdout, and
every `*.stderr.txt` the same run's stderr. They are raw bytes as qwen wrote
them, not hand-authored shapes.

**How they were made (2026-10-08, `@qwen-code/qwen-code@0.25.0`).** Re-captured
in the pinned image built from `tests/container/Dockerfile` at the
`cli-versions.lock` pins, `--network none`, uid 10001, no credential, with qwen
driven through the PATH wrapper (`/opt/mllm/bin/qwen`, so `redirect.yaml`'s
qwen-code row applies: the session-prefixed stub base URL, `QWEN_HOME`). argv is
`QwenCodeProvider.build_command`'s output (`qwen --bare --chat-recording false
-o json --max-tool-calls 0 --max-session-turns 25 [--json-schema …] -m
stub-model <prompt>`), env is `get_env`'s; `wall-time-exceeded` and
`rate-limit-wall-time` insert `--max-wall-time 5` / `100` ahead of `-m`, and
`multi-assistant` keeps the probe's own argv (`--yolo --max-session-turns 5`,
no tool cap, so the stub's tool call can run). The answering rows ran against
the committed `stub_server.py`; the failure rows against the §6 probe's own
`probestub.py`, copied unchanged (same error bodies, same `ANSWER=OK`). The
first capture (2026-08-17, 0.21.13) used a throwaway probe image and argv
without the tool cap; its two exit-0 shapes are kept below.

**The vendored ripgrep.** The pinned image installed qwen's bundled `rg`
without its exec bit, root-owned, so qwen's own repair (it `chmod`s the file
itself on a user-owned install) failed and every run printed `Ripgrep not
available: spawn …/rg EACCES. Falling back to built-in grep.` on stderr. The
main captures were taken with an executable copy of that `rg` bind-mounted in
its place, which is what a normal install looks like; re-checked on the image
rebuilt with the bit set (`success`, `auth-error`, `truncated`,
`wall-time-exceeded`: same exit codes, request counts and stdout sizes, stderr
byte-identical, `success`'s empty). `rate-limit-wall-time.root-owned-install.stderr.txt`
is the one capture taken with the stock 0644 `rg` on purpose: it is what a
root-owned install (`sudo npm i -g`, a distro package) prints, and
`QwenCodeProvider.vendor_text` exists because that `EACCES` used to classify
every run `permission_denied`.

The two `NOT-APPLICABLE.md` files are §6.2 rows that do not exist for this CLI,
each recording *why* — "not applicable, because …" is an acceptable cell; an
untested one is not.

| Fixture | What it pins |
| --- | --- |
| `success.stdout.json` | the envelope shape; answer is the bare string `OK`, so the unwrap correctly reports "no JSON document"; stderr empty |
| `success-document.stdout.json` | the **real** success case: the answer is a suggestion-list JSON array, and the test asserts the unwrapped `data` is that document — not `success is True` alone |
| `structured-output-success.stdout.json` | `--json-schema` really constrains the output (OQ2), and composes with `--max-tool-calls 0` |
| `exit-zero-failure.{stdout.json,stderr.txt}` | 500-forever, 28 requests, 107 s. **No longer exit 0 at 0.25.0**: exit 1, `subtype: error_during_execution`, `is_error: true`, the text under `error.message`, no `result` key, and an `AlreadyReportedError` envelope on stderr. The name is the §6.2 row's, kept |
| `exit-zero-failure.0.21.13.stdout.json` | **captured on 0.21.13** (the adapter's `min_version`): rc=0, `subtype: success`, `is_error: false` — a failure detected only by the `[API Error:` marker |
| `auth-error.{stdout.json,stderr.txt}` | 401, one request, exit 1, the upstream text verbatim. It classifies `auth_failed` since 2026-10-08 through `QwenCodeProvider`'s `[api error: 401` pattern (qwen's own status prefix). Before that it classified `server_error` on 0.21.13 and 0.25.0 alike: OpenAI's "Incorrect API key provided" is in no base auth pattern and a bare 401 is not one, so the `api error` rung took it |
| `invalid-model.{stdout.json,stderr.txt}` | 404 **naming the model**, which is what makes a stray `:effort` suffix findable; classifies `invalid_model` |
| `context-overflow.{stdout.json,stderr.txt}` | 400 context-length, terminal, one request; classifies `context_length_exceeded` |
| `truncated.{stdout.json,stderr.txt}` | a stream cut after its first frame, 5 requests, 21 s, exit 1. 0.25.0 writes one `system`/`retry` record per retry (`preserveText: false`, 8 of them for 5 requests) and one `Retrying provider attempt …` stderr line each, and the assistant text is only the last attempt's partial `OK` ahead of the marker — the cross-retry concatenation is gone |
| `truncated.0.21.13.stdout.json` | **captured on 0.21.13**: partial text concatenated across retries (`OKOKOKOKOK[API Error: …]`) inside a clean-marked record — the muse trap by a different route, and why the marker is searched rather than prefix-matched |
| `no-finish-reason.{stdout.json,stderr.txt}` | the same shape as `truncated` from a stream with no finish reason |
| `empty-stream.{stdout.json,stderr.txt}` | a zero-delta stream: 5 requests, then `[API Error: Model stream ended with empty response text.]`, exit 1 |
| `multi-assistant.stdout.json` | three assistant records; the `result` record is last-wins by construction |
| `stdout-error.stdout.json` | `--json-schema` against a plain-text answer: rc=1, stderr empty, the diagnosis on **stdout** inside the envelope, and no `result` key — the shape that would otherwise reach `parse_result_envelope` and be served as the answer |
| `wall-time-exceeded.{stdout,stderr}.txt` | exit 55: zero bytes on stdout, structured `FatalBudgetExceededError` on stderr (byte-identical to 0.21.13's) |
| `rate-limit-wall-time.{stdout,stderr}.txt` | 429-forever ended by `--max-wall-time 100`: exit 55, 28 requests, zero bytes on stdout, and stderr names the 429 (`Retrying in 60s (attempt 1/10): [API Error: 429 rate limit exceeded]`) **ahead of** the `FatalBudgetExceededError` envelope, so it classifies `rate_limited` and the envelope is found after the retry line |
| `rate-limit-wall-time.root-owned-install.stderr.txt` | the same row (`--max-wall-time 120`) with the stock 0644 `rg`: the ripgrep advisory line first. Its stdout was zero bytes |
