# grok parser fixtures

Every `*.stdout.txt` / `*.stderr.txt` here is a **real capture** of `grok`'s
streams under the argv `GrokProvider.build_command` emits, taken against a stub
in the containerized §6 probe (`grok 1.0.46`, re-captured 2026-10-08 by the same
rig that first captured them on 0.2.93; `--network none`, **no credential** —
`XAI_API_KEY` was the PATH wrapper's fake per-exec session key, which only has
to be non-empty: with it unset grok opens an interactive OAuth
device flow against accounts.x.ai and blocks until timeout). They are raw bytes
as grok wrote them; nothing here is hand-authored, and there is no synthesized
row.

Re-run once more the same day (review 24, pins #1/#2) with the two switches
`GrokProvider.get_env` now sets on every dispatch, in the row's `env_extra`:
`GROK_TURN_SUMMARY=0` and `GROK_CONFIG={"models":{"max_retries":2}}`. Eight rows
came back byte-identical, four differ only in per-exec ids; the request counts
and walls in `manifest.yaml` are that run's, and the unbounded numbers below are
what grok does out of the box.

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
- **Five requests per answered exec at 1.0.46** (three at 0.2.93): `GET
  /v1/models` (the catalog grok validates `-m` against), `GET /` (a reachability
  probe of the base URL, new in 1.0.x), a concurrent `POST /v1/responses`
  session-title call with the **hardcoded** model `grok-4.6` (`grok-build` at
  0.2.93) — which also carries tools, so only the model id separates it — the
  real `POST /v1/chat/completions`, and a NEW tool-less `POST
  /v1/chat/completions` on the `-m` model asking for "an ultra-short dashboard
  line that captures the AGENT'S REPLY": a second, billed call on the user's own
  model after every answered turn. A failed turn makes no such call.
  `GrokProvider.get_env` switches it off (`GROK_TURN_SUMMARY=0`): four requests
  per answered exec as dispatched, the answer's bytes unchanged.
- **The internal-retry envelope is two-tiered**: 401, 404 and 400 are terminal at
  one chat request; 429 gets exactly one retry (2 chat requests, 5.2 s); a cut
  stream, a 200 carrying an error object and an empty answer get **15 chat
  requests over 326 s** (335 s at 0.2.93), backing off to a ~30 s cap; a 5xx or a
  zero-byte response (15 / 335 s at 0.2.93) now gets **four such ladders back to
  back, 60 chat requests over ~1,340 s**. As dispatched (`models.max_retries`
  = 2) those are 2 chat requests over 5.1 s and 8 over 54 s (47 s for the
  zero-byte response); the 429 is unchanged.
- **During those windows grok writes zero bytes to either channel.** The first pass
  of this probe bounded five rows at 300 s and captured exactly the opencode
  silent-hang shape from a CLI that would have reported at 335 s. Those rows were
  re-run unbounded; the bounded artifacts are not committed. The 2026-10-08
  re-capture hit the same wall one size up: a 900 s bound fired on the 5xx and
  zero-byte rows, and they were re-run under 3,600 s.

| Fixture | What it pins |
| --- | --- |
| `success` | the `{"text", "stopReason", "sessionId", "requestId"}` envelope, unwrapped by `parse_output`. The test asserts the unwrapped `data` **is** the suggestion-list document; the stub's log of the re-run shows four requests (five before the summary call was switched off). `stopReason` is `end_turn` (`EndTurn` at 0.2.93) |
| `ephemeral-only` | an empty assistant message is **not** accepted: 15 chat requests over 326 s, then `empty response from model (no_visible_content)`. The right verdict reached through the wrong channel, since `parse_output` then reports that verdict as an answer |
| `multi-assistant` | **last**-wins, and the last document is the answer; the first is the decoy (`FIRST ANSWER (must NOT be selected)`), still in the bytes so the row tests selection — see defect 4 |
| `truncated` | a **real** cut mid-record on the *chat* surface (second SSE frame, the one carrying the text) → 15 chat requests / 326 s → `no_visible_content`. **Byte-identical** to `ephemeral-only` and `error-envelope` (md5 `07923ec7…`) |
| `error-envelope` | a provider error object on a **200**: grok finds no assistant content, sends 15 chat requests and reduces it to the same `no_visible_content`. The provider's text is lost, not mis-served |
| `stdout-error` | 5xx-forever: a nonzero exit whose diagnosis is on **stdout**, parsed as valid JSON and returned as the answer — §6 step 3's opencode trap, in its worse form. (Deviation recorded: stderr is not empty, it duplicates the message.) 60 chat requests over 1,341 s at 1.0.46 unbounded; 8 over 54 s as dispatched |
| `auth-error` | 401 → **one** chat request, terminal. grok quotes the vendor's error (`invalid_request_error: <message>` at 1.0.46, the whole JSON body at 0.2.93) plus its own `Model:` / `Auth: ApiKey` / `Version:` / `Available:` diagnostic block, so `Unauthorized (401)` reaches the classifier and `auth_failed` is right |
| `rate-limit` | 429-forever → grok retries **once** (2 chat requests, 5.2 s) and gives up. At 0.2.93 it replaced the vendor's words with its own sentence and dropped the wait; at 1.0.46 it passes them through (`rate_limit_exceeded: Rate limit reached … Please try again in 20s.`), so the capture now carries the wait |
| `invalid-model` | shape 1 of 2 — an **upstream 404** for a model that IS in the catalog. Names the model and the whole catalog, but classifies **`server_error`** (defect 3) |
| `invalid-model-preflight` | shape 2 of 2 — grok's own pre-flight rejection, **no completion** (the catalog GET, plus the root probe at 1.0.46; byte-identical across the two releases): `-m` is validated against `GET /v1/models` before anything is dispatched. `unknown model id`, classified correctly. This is the shape a stray `:effort` suffix produces |
| `context-overflow` | 400 context-length, passed through unedited, so `maximum context length` wins over the `API error (status 400 …)` wrapper that made the 404 row a `server_error`. One request, not retried |
| `zero-byte-response` | beyond §6.2's twelve, and the measurement behind the `empty` cell: rust's own `reqwest error stream: error sending request for url (…)` — the ONE transport failure grok distinguishes from `no_visible_content`, and it still classifies as `None`. 60 chat requests over 1,334 s at 1.0.46 |

## Two rows that are "not applicable, because …"

Both carry their measurement in `manifest.yaml` rather than being absent (§6.3).

- **`exit-zero-failure`** — ten injected failure shapes, ten exits of **1**. No
  producer at 0.2.93 or at 1.0.46, so the adapter's empty
  `exit_zero_failure_markers` is correct. This is the most load-bearing measurement in the set: with
  `parse_output` returning `success: true` on all ten, the exit code is the only
  thing separating a failed grok run from an answered one.
- **`empty`** — zero bytes on both channels is true for **326 seconds** (335 at
  0.2.93), or ~1,340 s on a 5xx or a zero-byte response at 1.0.46, and then
  stops being true. Capturing the silent window measures the harness's patience,
  not the CLI; the first pass of this probe did exactly that at a 300 s bound and
  the results were discarded. The window is itself the finding.

## Three rows are the same bytes, and that is the point

`truncated`, `ephemeral-only` and `error-envelope` have identical stdout
(md5 `07923ec777189786019712114bc55f94`) and identical stderr
(`f10e38e768d9b8a1ea871cbacae77df3`), at 1.0.46 exactly as at 0.2.93 — the
re-capture moved none of the three files. A cut stream, an empty assistant message
and a 200 carrying a provider error object are **indistinguishable** at grok's
output: all three become `Internal error: "empty response from model
(no_visible_content)"` after 15 chat requests. All three files are committed anyway —
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
2. **The silent window is 335 seconds — and at 1.0.46, ~1,340 s on a 5xx —
   BOUNDED 2026-10-08.** At
   0.2.93 five different transport-shaped failures each retry 15 times with a
   ~30 s backoff cap, writing **nothing** to either channel until they give up.
   At 1.0.46 three of them still do (326 s), while a 5xx and a zero-byte response
   run four such ladders back to back (60 chat requests, ~1,340 s).
   providers.yaml gives grok a 1200 s slot (`grok.default_timeout`; the class's
   own 600 s is the fallback for a config without the row), so the product
   survived the 0.2.93 window but did **not** survive the 1.0.46 5xx one: a
   fail-forever 5xx was killed with zero bytes on both channels, and its
   transport retry re-ran the same silent ladder. And any shorter slot — a
   configured `providers.yaml` timeout, an executor deadline with little left —
   turns a CLI that would have reported into one that reports zero bytes, which
   is the `opencode-run-silent-hangs` shape. **Bounded since 2026-10-08**
   (review 24, pins #2): every dispatch sets grok's own `models.max_retries` = 2
   through `GROK_CONFIG` (merged over the user's own overlay), so the 5xx is 8
   chat requests over 54 s and reports its error, and the adapter declares
   `internal_retries` at that measurement — 11 requests, `max_wall_s` 55,
   `retries_rate_limit` **true but only once** — so §7.5 rule 4 refuses, with
   a reason, an attempt the slot has no room left for.
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
  --tmpfs /work/homes:rw,mode=0700,uid=10001,gid=10001 \
  --tmpfs /work/sample-project:rw,mode=0755,uid=10001,gid=10001 \
  --tmpfs /tmp:rw,mode=1777 \
  -v "$PWD/skills/multi-llm/tests/container:/rig:ro" \
  -v "$PWD/plans/harness-expansion/probe-out/grok:/work/out:rw" \
  -e MLLM_STUB_PORT=8099 -e MLLM_REDIRECT_YAML=/rig/redirect.yaml \
  -e MLLM_REAL_DIR=/opt/mllm/real -e MLLM_HOMES_ROOT=/work/homes \
  -e CAP_CLI=grok -e CAP_BIN=grok -e CAP_MODEL=stub-model \
  -e CAP_ROWS=<rows> -e CAP_BOUND_S=<bound> \
  --entrypoint /bin/bash <image built from tests/container/Dockerfile> -lc \
  'python3 /rig/seed_workspace.py --spec grok:stub-model --runner opencode:stub-model && python3 /work/out/capture.py'
```

(the 2026-08-19 capture ran in `mllm-e2e:local` on 0.2.93; the 2026-10-08
re-run in the image built at the cli-versions.lock pins, 1.0.46, with the out
dir a fresh copy of the probe's `capture.py`, `rows.json`, `prompt.txt` and
scenarios; the second 2026-10-08 re-run the same way, one container per row,
with `zero-byte-response` added to `rows.json` as the `empty` row's spec. The
tmpfs owners are posture.yaml's: without `uid=10001` the seed step cannot write
the workspace.)

Three preconditions, each of which fails **silently** if wrong: (1) the `-m`
model must appear in the stub's `GET /v1/models` catalog or grok exits 1 with
`unknown model id` before dispatching anything — which is what
`invalid-model-preflight` deliberately triggers; (2) `XAI_API_KEY` must be
non-empty or grok opens an interactive OAuth device flow and blocks; (3) the
concurrent `POST /v1/responses` title call uses a hardcoded model (`grok-4.6`
at 1.0.46, `grok-build` at 0.2.93) and carries tools, so a stub keyed on
`has_tools` cannot separate it from the real turn. Use `stub-model`, never providers.yaml's `grok-4.5:high` — the
`:effort` suffix adds `--reasoning-effort` and changes the argv under test. Do
**not** wrap the run in `timeout`: at 300 s five rows are zero bytes on both
channels. With the grok row's `env_extra` in force (the shipped dispatch) the
slowest row is 54 s; without it, total wall time at 1.0.46 ≈ 65 minutes if the
five slow rows run in sequence (three of ~326 s, two of ~1,340 s), which is why
the probe ran them as one-row containers in parallel, what `result.<row>.json`
beside `results.json` exists for. Give the slow rows a bound well above their
wall (`CAP_BOUND_S`); a bound that fires is a manufactured hang, not a capture.
