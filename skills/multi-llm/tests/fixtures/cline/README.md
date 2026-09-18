# cline parser fixtures

Every `*.stdout.txt` / `*.stderr.txt` here is a **real capture** of `cline`'s
streams under the argv `ClineProvider.build_command` emits, taken against a stub
in the containerized §6 probe (`cline 3.0.55`, `--network none`, **no
credential** — the only key anywhere is the wrapper's per-exec
`mllm-stub-<id>` string, which means something on 127.0.0.1 and nowhere else).
They are raw bytes as cline wrote them, with the one labelled exception below.

They are graded by `tests/test_provider_fixture_rows.py` through
`manifest.yaml` — the generic §6.2 harness, not a hand-written class. The
manifest is the authority on what each row asserts; this file says *why* each
one is here.

Three facts hold across **every** row and are the reason several assertions look
the way they do:

- cline exits **1** on all nine measured failures. That is a **change** from the
  measurement `ClineProvider`'s own comment records (`cline **exits 0**`, 3.0.39,
  2026-08-16) and it is why `exit-zero-failure` is not-applicable here;
- the diagnosis rides **both** channels — stdout carries the JSONL `run_result`
  with `finishReason: "error"`, stderr carries a one-line
  `{"type":"error","message":…}` copy of the same text;
- stderr on any run that reached the model also carries three AI SDK
  deprecation warnings (938–1029 bytes) that say nothing about the outcome. The
  empty-looking channel is never empty and never the only carrier, so a parser
  reading stdout alone loses nothing.

| Fixture | What it pins |
| --- | --- |
| `success` | the nine-line JSONL shape and the whole unwrap: the answer is the `text` field of the **last** `run_result`, and the test asserts the unwrapped `data` **is** the suggestion-list document — plus `classify_failure is None`, the assertion that catches an adapter reading its own happy path as a failure |
| `truncated` | a **real** cut mid-body (200 headers promising a full `Content-Length`, 60 bytes, socket closed): cline retries three times in 2.4 s and then erases the wire condition, reporting `Model returned empty response` |
| `ephemeral-only` | an all-noise run is a **failure**, not an empty-string success. Everything on disk is a `hook_event`/`agent_event` record the parser filters; cline itself refuses to call an empty completion an answer |
| `multi-assistant` | **last**-wins, not first-wins and not concatenation. The first `run_result` reads `FIRST ANSWER (must NOT be selected)`. The one synthesized row — see below |
| `error-envelope` | a provider error object on a **200**. Worse than "served as the answer": the text is **lost**. cline sees a completion with no choices, calls it `Model returned empty response`, and no vendor byte reaches any pattern table |
| `auth-error` | 401 → **one** request, terminal, upstream wording intact. Classified `runner_error` when captured; now `auth_failed` from this row's own words. See below |
| `rate-limit` | 429-forever → cline **does** retry rate limits internally: 3 requests over 7.8 s, so an executor-level re-dispatch doubles a wait the CLI already served |
| `invalid-model` | 404 **naming the model**, and naming it `stub-model` — the bare id `-m` carried, not the `openai/stub-model` spec. That is the shape a stray `:effort` suffix produces |
| `context-overflow` | 400 context-length, handled **well**: cline runs its own overflow-recovery path (three `notice` events), raises `ContextWindowOverflowError`, and **quotes the provider's sentence in full**, so `CONTEXT_OVERFLOW_MARKERS` matches and the terminal classification costs one request |
| `zero-body` (extra) | a zero-byte response is a *distinct* event at cline, unlike at aider: `Cannot connect to API: … (ECONNRESET)`. Classified `runner_error` — see below |
| `server-error` (extra) | 5xx-forever, §6.1.1's experiment 2 and the whole internal-retry envelope in one capture: **3 requests over 8.0 s** at ~0 s / ~1 s / ~6 s |

## The one synthesized row

`multi-assistant.stdout.txt` is **synthesized from the real record**: cline emits
exactly one `run_result` per invocation and a second requires a tool loop the
committed stub deliberately never drives. The decoy line was built by
`json.loads`-ing cline's own `run_result`, replacing only its `text` value and
re-serialising with `separators=(',',':')` — verified byte-identical to the
original before the substitution, so the shape is cline's and only the answer is
ours. Recorded here rather than faked silently; the rule it pins (last-wins) is
what the muse trap cost this repo before.

## Three rows that are "not applicable, because …"

All three carry their measurement in `manifest.yaml` rather than being absent
(§6.3).

- **`exit-zero-failure`** — at 3.0.55 the exit code is **1** on every failure the
  rig can produce (nine modes, nine captures). The exit-0 shape the adapter's
  comment records at 3.0.39 no longer has a producer. The declared
  `exit_zero_failure_markers` are **not** dead: `classify_streams` consults them
  regardless of exit code and they are what classifies five of the rows here.
- **`stdout-error`** — half-satisfied by construction (cline exits non-zero with
  its whole diagnosis on stdout, which is every failure row here) and the missing
  half was measured: stderr is **never** empty, carrying at minimum a 132-byte
  copy of the same error line. What the row exists to prevent is pinned by all
  eight failure rows, each asserting `success: false`.
- **`empty`** — zero bytes on both channels is unreachable while cline runs:
  `agent_start` and `iteration_start` are written before the first HTTP request,
  and the smallest capture here is **1360 bytes**. Producing one would mean
  killing the process, i.e. the `timeout(1)` zero-byte-hang artifact §6 step 3
  forbids manufacturing.

## Three shapes collapse onto one message

`truncated`, `error-envelope` and `ephemeral-only` are three different things on
the wire — a body cut at 60 bytes, a 200 carrying an error object, and a
completion whose content is the empty string — and cline reports all three as
`Model returned empty response`, with three retries and ~2.4 s apiece. They are
committed as three files because the *producers* are distinct and a future
release may separate them; today the only difference between the captures is the
agent/task id and the timestamp. Note also that `"errorClass":` is present on the
401/404/400/500/429/ECONNRESET rows and **absent** on these three, so the
adapter's second declared marker (`"errorclass"\s*:`) does not fire on them.

## What this set found

Four measured defects, recorded here rather than silently fixed — the fixtures
pin today's behaviour, so each fix is a visible edit to both the adapter and the
manifest. Defect 1 has since been fixed on the adapter, and its entry below
records what changed rather than being deleted.

1. **A 401 classified as `runner_error`, not `auth_failed` — FIXED.** OpenAI's
   own 401 sentence — *"Incorrect API key provided: …"* — matched none of
   `base.auth_error_patterns`' ten entries, and cline's own added
   `user not found` is a different provider's wording, so the run fell through
   every specific table to the exit-0 marker and a credential failure sat on a
   retryable rung. `incorrect api key` is now a `ClineProvider`
   `auth_error_patterns` entry, driven by this row's own bytes rather than by
   new ones, and auth is consulted before the marker (§7.2 step 4 vs step 6), so
   `auth-error` reads `auth_failed`. It is cline's **second** measured auth
   wording, not a replacement for the first: `user not found` came from 3.0.39
   with an invalid OPENROUTER_API_KEY. cline glosses neither — both are the
   upstream provider's sentence passed through — which is why this set needs one
   entry per upstream rather than one per CLI. Checked against every capture
   here: the phrase occurs in `auth-error` alone, so no other cell moved.
2. **`exit_zero_failure_markers` has no exit-0 producer left.** cline exits 1 at
   3.0.55. The markers still earn their place as the last-resort classifier, but
   the adapter comment's premise ("the exit code carries no verdict") is now
   false for this CLI and should be re-worded rather than trusted.
3. **A 5xx classifies as `runner_error`, not `server_error`.** OpenAI's
   *"The server had an error while processing your request"* matches no member of
   `base.SERVER_ERROR_MARKERS`, whose entries all want the literal "server
   error", "internal server error" or "api error". Both categories are retryable
   so the ladder still behaves; the `Reason:` line a user acts on is wrong.
4. **`ECONNRESET` classified as `runner_error`, not `connection_error` —
   FIXED.** `CONNECTION_ERROR_MARKERS` matched `connection (?:refused|reset|
   closed|…)` as one token and cline writes *"connection was closed"*, so the
   sentence fell past the connection table and was caught one rung lower by
   `exit_zero_failure_markers` — the last-resort "this run failed at all"
   verdict rather than a reading of the cause. The shared row is now
   `connection (?:was )?(?:refused|reset|closed|…)`, driven by this row's own
   bytes, and `zero-body` reads `connection_error`. The widening was deferred
   while omp landed, because omp's `empty` row carries the **identical**
   Bun/undici sentence and re-pinning two adapters at once is the whole change;
   both were re-pinned together on 2026-09-18. The table still carries
   `ECONNREFUSED` and not `ECONNRESET`, which no longer matters here: the
   sentence in front of it matches. Retry behaviour for this row did not move —
   cline declares no `describe_failure`, so Lane A's haystack is the bare
   `cline exited with code 1` either way — but the `Reason:` line a user acts on
   went from *"the runner reported a failure"* to *"Connection/network error"*.

## Reproducing

One container per row, no credential, no network — cline spawns a background hub
daemon and self-updates, so a shared container would let one row's daemon serve
the next. The driver is a throwaway script in a scratch mount (deliberately not
committed; the two servers it drives are):

```
docker run --rm --network none \
  --user 10001:10001 --cap-drop ALL --security-opt no-new-privileges --read-only \
  --tmpfs /work/homes:rw --tmpfs /work/sample-project:rw --tmpfs /tmp:rw \
  -v "$PWD/skills/multi-llm/tests/container:/rig:ro" -v "<scratch>:/work/out:rw" \
  -e MLLM_STUB_PORT=8099 -e MLLM_REDIRECT_YAML=/rig/redirect.yaml \
  -e MLLM_REAL_DIR=/opt/mllm/real -e MLLM_HOMES_ROOT=/work/homes \
  --entrypoint /bin/bash mllm-e2e:local -lc 'python3 /work/out/incontainer.py'
```

Per row the script copies `/opt/mllm/rig/home_seed/cline/.` into
`/work/homes/cline/` (the wrapper's seed source — a HAND-WRITTEN
`providers.json` is silently ignored, which is why the committed seed is what
`cline auth` itself writes), starts one server on 127.0.0.1:8099 —
`stub_server.py --scenario <inline answer>` for `success`/`ephemeral-only`,
`failserver.py --mode {trunc,errbody,500,401,429,404,ctx,empty}` for the rest —
polls `/__health` **with urllib** (there is no curl in the image), then runs
`/opt/mllm/bin/cline` (which *is* `wrapper.py`, and applies the whole
redirect.yaml row itself: the seed copy with `{{base_url}}` substitution, the
`CLINE_NO_AUTO_UPDATE=1` child env, a fresh per-invocation `$HOME`). Do **not**
wrap the run in `timeout`. Total wall time ≈ 45 s for the ten rows; the
`requests:` counts in `manifest.yaml` are read out of the server's own `--log`,
excluding the `/__health` poll.

Raw probe output, including the per-row server logs and the wrapper call log:
`plans/harness-expansion/probe-out/cline/`.
