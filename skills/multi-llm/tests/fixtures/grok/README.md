# grok parser fixtures

Every `*.stdout.txt` / `*.stderr.txt` here is a **real capture** of `grok`'s
streams under the argv `GrokProvider.build_command` emits, taken against a stub
in the containerized §6 probe (`grok 0.2.93`, `--network none`, **no
credential** — `XAI_API_KEY` was the PATH wrapper's fake per-exec session key,
which only has to be non-empty: with it unset grok opens an interactive OAuth
device flow against accounts.x.ai and blocks until timeout). They are raw bytes
as grok wrote them; nothing here is hand-authored, and there is no synthesized
row.

They are graded by `tests/test_provider_fixture_rows.py` through `manifest.yaml`.
The manifest is the authority on what each row asserts; this file says *why* each
one is here, and what the set found.

> **Nine of the twelve graded rows declare `expect.success: true` on a run that
> failed.** That is the measurement, not a weakened assertion. Read defect 1
> before reading the manifest.

Five facts govern every row:

- **`parse_output` reports success on every failure.** grok's failure output is a
  well-formed JSON object `{"type":"error","message":…}`. It has no `"text"` key,
  so `GrokProvider.parse_output` falls through to `return {"success": True,
  "data": wrapper}` and hands the error envelope back as the model's answer.
- **The failure text lands on both channels** — the JSON envelope on stdout, an
  `Error: <same message>` line on stderr. stderr is 0 bytes on the two answering
  rows only.
- **Three requests per exec**: `GET /v1/models` (the catalog grok validates `-m`
  against), a concurrent `POST /v1/responses` session-title call with the
  **hardcoded** model `grok-build` — which also carries tools, so only the model
  id separates it — and then the real `POST /v1/chat/completions`.
- **The internal-retry envelope is two-tiered**: 401, 404 and 400 are terminal at
  one chat request; 429 gets exactly one retry (2 requests, 2.2 s); every
  transport-shaped failure gets **15 chat requests over 335 s**, backing off to a
  ~30 s cap.
- **During those 335 s grok writes zero bytes to either channel.** The first pass
  of this probe bounded five rows at 300 s and captured exactly the opencode
  silent-hang shape from a CLI that would have reported at 335 s. Those rows were
  re-run unbounded; the bounded artifacts are not committed.

| Fixture | What it pins |
| --- | --- |
| `success` | the `{"text", "stopReason", "sessionId", "requestId"}` envelope, unwrapped by `parse_output`. The test asserts the unwrapped `data` **is** the suggestion-list document, and the committed server log beside it shows all three requests |
| `ephemeral-only` | an empty assistant message is **not** accepted: 15 retries over 335 s, then `empty response from model (no_visible_content)`. The right verdict reached through the wrong channel, since `parse_output` then reports that verdict as an answer |
| `multi-assistant` | **last**-wins, and the last document is the answer; the first is the decoy (`FIRST ANSWER (must NOT be selected)`), still in the bytes so the row tests selection — see defect 4 |
| `truncated` | a **real** cut mid-record on the *chat* surface (second SSE frame, the one carrying the text) → 15 retries / 335 s → `no_visible_content`. **Byte-identical** to `ephemeral-only` and `error-envelope` (md5 `07923ec7…`) |
| `error-envelope` | a provider error object on a **200**: grok finds no assistant content, retries 15 times and reduces it to the same `no_visible_content`. The provider's text is lost, not mis-served |
| `stdout-error` | 5xx-forever: a nonzero exit whose diagnosis is on **stdout**, parsed as valid JSON and returned as the answer — §6 step 3's opencode trap, in its worse form. (Deviation recorded: stderr is not empty, it duplicates the message) |
| `auth-error` | 401 → **one** chat request, terminal. grok quotes the vendor body plus its own `Model:` / `Auth: ApiKey` / `Version:` / `Available:` diagnostic block, so `Unauthorized (401)` reaches the classifier and `auth_failed` is right |
| `rate-limit` | 429-forever → grok retries **once** (2 requests, 2.2 s) and gives up with its own sentence. It does **not** quote the vendor's `try again in 20s`, so the capture carries no wait to honour |
| `invalid-model` | shape 1 of 2 — an **upstream 404** for a model that IS in the catalog. Names the model and the whole catalog, but classifies **`server_error`** (defect 3) |
| `invalid-model-preflight` | shape 2 of 2 — grok's own pre-flight rejection, **one request, no completion**: `-m` is validated against `GET /v1/models` before anything is dispatched. `unknown model id`, classified correctly. This is the shape a stray `:effort` suffix produces |
| `context-overflow` | 400 context-length, passed through unedited, so `maximum context length` wins over the `API error (status 400 …)` wrapper that made the 404 row a `server_error`. One request, not retried |
| `zero-byte-response` | beyond §6.2's twelve, and the measurement behind the `empty` cell: rust's own `reqwest error stream: error sending request for url (…)` — the ONE transport failure grok distinguishes from `no_visible_content`, and it still classifies as `None` |

## Two rows that are "not applicable, because …"

Both carry their measurement in `manifest.yaml` rather than being absent (§6.3).

- **`exit-zero-failure`** — ten injected failure shapes, ten exits of **1**. No
  producer at 0.2.93, so the adapter's empty `exit_zero_failure_markers` is
  correct. This is the most load-bearing measurement in the set: with
  `parse_output` returning `success: true` on all ten, the exit code is the only
  thing separating a failed grok run from an answered one.
- **`empty`** — zero bytes on both channels is true for **335 seconds** and then
  stops being true. Capturing the silent window measures the harness's patience,
  not the CLI; the first pass of this probe did exactly that at a 300 s bound and
  the results were discarded. The 335 s window is itself the finding.

## Three rows are the same bytes, and that is the point

`truncated`, `ephemeral-only` and `error-envelope` have identical stdout
(md5 `07923ec777189786019712114bc55f94`) and identical stderr
(`f10e38e768d9b8a1ea871cbacae77df3`). A cut stream, an empty assistant message
and a 200 carrying a provider error object are **indistinguishable** at grok's
output: all three become `Internal error: "empty response from model
(no_visible_content)"` after 15 retries. All three files are committed anyway —
unlike aider's zero-byte/truncated pair, which is one file plus a note — because
here the collapse spans three *different* §6.2 rows, and a future release that
started distinguishing them should show up as a diff on the row that changed.

## What this set found

Four measured defects, recorded here rather than silently fixed. Defect 4 has
since been fixed in the shared extractor, and its entry records what changed.

1. **Every failed grok run parses as a success, with the error envelope as the
   answer.** `parse_output` handles `{"text": …}` and then ends with a bare
   `return {"success": True, "data": wrapper}` for any other object. grok's
   failure shape is another object, so ten distinct failures — 401, 429, 404,
   400, 5xx, cut stream, zero-byte response, error-on-200, empty answer, and its
   own pre-flight model rejection — all come back `success: True` with
   `data = {"type": "error", "message": …}`. Downstream that is a review result
   containing one dict instead of a suggestion list. Only the nonzero exit code
   saves it. The fix is one branch: an object whose `type` is `"error"` (or which
   carries `message` and no `text`) is a failure, and `message` is the error
   string. Every failure row here would then flip to `expect.success: false` with
   a real `error_contains`, which is the visible edit that proves the fix.
2. **The silent window is 335 seconds.** Five different transport-shaped failures
   each retry 15 times with a ~30 s backoff cap, writing **nothing** to either
   channel until they give up. `GrokProvider.default_timeout` is 600 s so the
   product survives it, but any shorter slot — a configured `providers.yaml`
   timeout, an executor deadline with little left — turns a CLI that would have
   reported into one that reports zero bytes, which is the
   `opencode-run-silent-hangs` shape. The adapter declares no
   `internal_retries` envelope; the measurement for one is in this directory
   (`max_requests` 15 on the chat surface, `max_wall_s` ≈ 335,
   `retries_rate_limit` **true but only once**).
3. **An upstream 404 classifies as `server_error`.** grok wraps every HTTP
   failure in `API error (status <n> <reason>)`, and `\bapi error\b` is a
   `SERVER_ERROR_MARKERS` entry consulted *before* `INVALID_MODEL_MARKERS` — so
   the row that carries the model name loses to the wrapper. The pre-flight row
   classifies correctly only because that message has no `API error` wrapper at
   all. `context-overflow` escapes the same fate only because the context table
   is consulted *before* the server table. Same root cause as claude-code's
   `API Error:` prefix, in a second CLI.
4. **`multi-assistant` selected the FIRST document, and it was the decoy —
   FIXED** — `extract_json_from_text(prefer_arrays=True)` returned the first
   array it found. grok concatenates every turn into one `text`, so there was no
   boundary for a per-adapter rule and the fix had to be one rule in the shared
   helper: **the last document wins, at every level**, with shape preference
   consulted before position and a contained candidate never outranking the one
   that contains it. It is aider's documented last-wins rule generalised from
   the envelope to the text inside it, and it now covers all three CLIs in this
   batch. The row moved with the fix and still carries the decoy in
   `raw_contains`.

## Reproducing

```
docker run --rm --network none --user 10001:10001 --read-only \
  --tmpfs /work/homes:rw --tmpfs /work/sample-project:rw --tmpfs /tmp:rw \
  -v "$PWD/skills/multi-llm/tests/container:/rig:ro" \
  -v "$PWD/plans/harness-expansion/probe-out/grok:/work/out:rw" \
  -e MLLM_STUB_PORT=8099 -e MLLM_REDIRECT_YAML=/rig/redirect.yaml \
  -e MLLM_REAL_DIR=/opt/mllm/real -e MLLM_HOMES_ROOT=/work/homes \
  --entrypoint /bin/bash mllm-e2e:local -lc 'python3 /work/out/capture.py'
```

Three preconditions, each of which fails **silently** if wrong: (1) the `-m`
model must appear in the stub's `GET /v1/models` catalog or grok exits 1 with
`unknown model id` before dispatching anything — which is what
`invalid-model-preflight` deliberately triggers; (2) `XAI_API_KEY` must be
non-empty or grok opens an interactive OAuth device flow and blocks; (3) the
concurrent `POST /v1/responses` title call uses the hardcoded model `grok-build`
and carries tools, so a stub keyed on `has_tools` cannot separate it from the
real turn. Use `stub-model`, never providers.yaml's `grok-4.5:high` — the
`:effort` suffix adds `--reasoning-effort` and changes the argv under test. Do
**not** wrap the run in `timeout`: at 300 s five rows are zero bytes on both
channels. Total wall time ≈ 28 minutes if the five 335 s rows run in sequence;
the probe ran them as five one-row containers in parallel instead, which is what
`result.<row>.json` beside `results.json` exists for.
