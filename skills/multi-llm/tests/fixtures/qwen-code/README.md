# qwen-code parser fixtures

Every `*.stdout.json` here is a **real capture** of `qwen -o json` stdout, taken
against a stub in the containerized §6 probe (`@qwen-code/qwen-code@0.21.13`,
`--network none`, no credential). They are raw bytes as qwen wrote them, not
hand-authored shapes.

The two `NOT-APPLICABLE.md` files are §6.2 rows that do not exist for this CLI,
each recording *why* — "not applicable, because …" is an acceptable cell; an
untested one is not.

| Fixture | What it pins |
| --- | --- |
| `success.stdout.json` | the envelope shape; answer is the bare string `OK`, so the unwrap correctly reports "no JSON document" |
| `success-document.stdout.json` | the **real** success case: the answer is a suggestion-list JSON array, and the test asserts the unwrapped `data` is that document — not `success is True` alone |
| `structured-output-success.stdout.json` | `--json-schema` really constrains the output (OQ2) |
| `exit-zero-failure.stdout.json` | rc=0, `subtype: success`, `is_error: false`, empty stderr — and the run is still a **failure**, detected only by the `[API Error:` marker |
| `auth-error.stdout.json` | 401 in the same exit-0 envelope; the upstream text survives verbatim so `auth_error_patterns` matches |
| `invalid-model.stdout.json` | 404 **naming the model**, which is what makes a stray `:effort` suffix findable |
| `context-overflow.stdout.json` | 400 context-length, terminal, one request |
| `truncated.stdout.json` | **partial text concatenated across retries** (`OKOKOKOKOK[API Error: …]`) — the muse trap by a different route, and why the marker is searched rather than prefix-matched |
| `no-finish-reason.stdout.json` | the same concatenation from a stream with no finish reason |
| `empty-stream.stdout.json` | a zero-delta stream: 5 requests, then the API-error envelope, rc=0 |
| `multi-assistant.stdout.json` | three assistant records; the `result` record is last-wins by construction |
| `stdout-error.stdout.json` | rc=1 with the diagnosis on **stdout** inside the envelope, and no `result` key — the shape that would otherwise reach `parse_result_envelope` and be served as the answer |
| `wall-time-exceeded.{stdout,stderr}.txt` | exit 55: zero bytes on stdout, structured `FatalBudgetExceededError` on stderr |
