# goose parser fixtures

Every `*.stdout.txt` / `*.stderr.txt` here is a **real capture** of `goose run
--output-format json`'s streams under the argv `GooseProvider.build_command`
emits, taken against a stub in the containerized §6 probe (`goose 1.41.0`,
`--network none`, **no credential** — the only key anywhere is the wrapper's
per-exec `mllm-stub-<id>` string in `OPENAI_API_KEY`, which means something on
127.0.0.1 and nowhere else). They are raw bytes as goose wrote them, with the one
labelled exception below.

They are graded by `tests/test_provider_fixture_rows.py` through
`manifest.yaml` — the generic §6.2 harness, not a hand-written class. The
manifest is the authority on what each row asserts; this file says *why* each one
is here.

Three facts hold across **every** row:

- goose exits **0** on all nine measured failures, so the exit code carries no
  verdict at all and only a stream marker does;
- stderr is **0 bytes** on all eleven captures — the whole envelope and the whole
  diagnosis ride stdout (the opencode trap, in a third CLI). The empty
  `*.stderr.txt` files are committed rather than omitted, because "stderr was
  empty" is the measurement;
- every run makes at least **three** requests: `GET /v1/models`, then a
  SESSION-TITLE completion goose fires even under `--no-session`, then the real
  turn. Under a fail-forever server both completions are retried to four
  attempts apiece — 7 completions, ~6.7 s — which is the internal-retry envelope
  for this CLI.

## The prompt is part of the fixture, and that is a finding

goose echoes the user message back inside its own JSON envelope, so
`classify_failure` reads **our prompt** as well as goose's answer. The product's
own `prompts/plan_review.txt` example block contains the word *"timeout"*, and
`\btime[ _-]?out\b` is a member of `base.TIMEOUT_MARKERS` — which alone makes
`classify_streams` answer `timeout` on a run that **succeeded**.

So the canonical rows below were captured with a 4689-byte review prompt whose
example block is reworded to carry no marker word; they measure goose. The extra
`prompt-echo` row is the **same successful run** captured with the product-shaped
prompt, committed so the defect is pinned by a test (`success: true` *and*
`classify_failure: "timeout"` on one set of bytes) rather than by prose. Fixing
it flips that cell to `null`.

| Fixture | What it pins |
| --- | --- |
| `success` | the envelope shape and the whole unwrap: `{"messages": […], "metadata": {…}}`, answer = the concatenated `type == "text"` parts of the **last** `role == "assistant"` message. The test asserts the unwrapped `data` **is** the suggestion-list document and that `classify_failure is None` |
| `truncated` | a **real** cut mid-body. goose does **not** retry it (3 requests, 0.15 s) and reports `Network error: Stream decode error: … EOF while parsing a string at line 1 column 54`. Neither `parse_output`'s failure branch nor any marker table matches it — see below |
| `ephemeral-only` | an all-noise run is a **failure**, not an empty-string success: a complete, well-formed envelope with `"status": "completed"` and **no assistant message in it at all** |
| `multi-assistant` | **last**-wins, not first-wins and not concatenation. The first assistant message reads `FIRST ANSWER (must NOT be selected)`. The one synthesized row — see below |
| `error-envelope` | a provider error object on a **200**. The text is **lost**, and so is any sign that something went wrong: the envelope is byte-shape identical to the empty-completion run (5236 bytes each, differing only in the `created` timestamp) |
| `exit-zero-failure` | exit **0** on a run that produced nothing, with the measured `exit_zero_failure_markers` phrase `Ran into this error:` present — and §6.1.1's envelope in one capture: 8 requests (4 attempts at the real turn, 3 at the session-title call), ~6.7 s |
| `auth-error` | 401 → **no retry**. Classified from goose's own word `Unauthorized`; OpenAI's *"Incorrect API key provided: …"* matches nothing in `base.auth_error_patterns`, which is the same hole that leaves cline's identical 401 classified `runner_error` |
| `rate-limit` | 429-forever → goose **does** retry rate limits internally: 8 requests over 7.2 s, so an executor-level re-dispatch doubles a wait the CLI already served |
| `invalid-model` | 404 **naming the model** — and a **retry ladder on a 404**: 7 completions over 6.9 s on a condition no retry can fix, plus a second `GET /v1/models` to render `Available models for this provider: …` |
| `context-overflow` | 400 context-length, **swallowed**: goose substitutes its own compaction report and classifies as **nothing** — two separate misses, below |
| `zero-body` (extra) | a zero-byte response, which goose *does* retry to the full 8 requests / 6.8 s (unlike the cut stream it gives up on immediately), reported as `Network error — check your network connection and try again.` |
| `prompt-echo` (extra) | the classifier reading our own prompt: a **successful** run whose `classify_failure` is `timeout` because the prompt says "timeout" |

## The one synthesized row

`multi-assistant.stdout.txt` is **synthesized from the real record**: a second
assistant message requires a tool loop, and the committed stub deliberately never
emits tool calls. goose's own message block was `json.loads`-ed, its single text
part replaced with a decoy array, re-rendered with `json.dumps(indent=2)` and
re-indented to goose's 4-space base, so every byte around it is unchanged.
Recorded here rather than faked silently — the rule it pins (last-wins) is what
the muse trap cost this repo before.

## Two rows that are "not applicable, because …"

Both carry their measurement in `manifest.yaml` rather than being absent (§6.3).

- **`stdout-error`** — the row needs a *nonzero* exit and goose exits **0** on
  every failure the rig can produce (nine modes, nine captures). The other half
  of the row is true of every capture here and is pinned by all of them: the
  diagnosis is on stdout with stderr at exactly **0 bytes**.
- **`empty`** — zero bytes on both channels is unreachable while goose runs: the
  envelope is written unconditionally and always contains at least the echoed
  user message, so the smallest capture here is **5236 bytes**. Producing one
  would mean killing the process, i.e. the `timeout(1)` zero-byte-hang artifact
  §6 step 3 forbids manufacturing. The nearest real shape — an envelope with
  nothing in it the parser can use — *is* captured, as `ephemeral-only`.

## What this set found

Four measured defects, recorded here rather than silently fixed.

1. **The classifier reads our own prompt** (`prompt-echo`). goose puts the user
   message in its output, so any marker word in a shipped prompt template decides
   the verdict. `prompts/plan_review.txt` contains "timeout"; a successful goose
   run therefore classifies as `timeout` today. The fix is on the adapter side —
   classify against the *selected assistant text* rather than against raw stdout
   — and it is the only defect here that a prompt edit would merely paper over.
2. **`Ran into this error:` misses two of goose's five failure vocabularies.**
   The declared marker is `re.compile(r"ran into this error:")`, and goose writes
   `Ran into this error TRYING TO COMPACT:` for a context overflow and
   `Network error:` for both stream failures. On those three rows `parse_output`
   never takes its explicit-failure branch — it falls through to
   `extract_json_from_text` and reports `No valid JSON found in text`, which
   names neither the condition nor the CLI.
3. **`context-overflow` classifies as nothing.** goose replaces the vendor's
   sentence with `Failed to compact: context limit exceeded even after removing
   all tool responses`, and `context limit exceeded` matches no member of
   `CONTEXT_OVERFLOW_MARKERS` (every entry wants "length", "window",
   "conversation context" or a token count). Combined with defect 2 the verdict
   is `None` — transport-shaped — so the executor spends its whole bounded-retry
   budget on the one condition §7.2 declares terminal, after goose already burned
   five compaction retries on it.
4. **Both stream failures classify as nothing.** `Network error: …` matches no
   member of `base.CONNECTION_ERROR_MARKERS`, whose entries want
   `connection refused/reset/closed`, `getaddrinfo`, `ECONNREFUSED` and friends.
   A one-word addition (`network error`) would cover both `truncated` and
   `zero-body`.

## Reproducing

One container per row, no credential, no network:

```
docker run --rm --network none \
  --user 10001:10001 --cap-drop ALL --security-opt no-new-privileges --read-only \
  --tmpfs /work/homes:rw --tmpfs /work/sample-project:rw --tmpfs /tmp:rw \
  -v "$PWD/skills/multi-llm/tests/container:/rig:ro" -v "<scratch>:/work/out:rw" \
  -e MLLM_STUB_PORT=8099 -e MLLM_REDIRECT_YAML=/rig/redirect.yaml \
  -e MLLM_REAL_DIR=/opt/mllm/real -e MLLM_HOMES_ROOT=/work/homes \
  --entrypoint /bin/bash mllm-e2e:local -lc 'python3 /work/out/incontainer.py'
```

goose needs no seeded home (`seed_files: []`): `OPENAI_BASE_URL`,
`GOOSE_PROVIDER=openai` and `GOOSE_DISABLE_KEYRING=1` come from redirect.yaml's
row and the wrapper applies them. Per row the script starts one server on
127.0.0.1:8099 — `stub_server.py --scenario <inline answer>` for the answering
rows, `failserver.py --mode {trunc,errbody,500,401,429,404,ctx,empty}` for the
rest — polls `/__health` **with urllib** (there is no curl in the image), then
runs `/opt/mllm/bin/goose` (which *is* `wrapper.py`). `stub-model` must be in the
stub's `CATALOG_MODELS`, because goose fetches `GET /v1/models` before its first
completion and fails the row if it is absent. Do **not** wrap the run in
`timeout`. Total wall time ≈ 30 s for the eleven rows; the `requests:` counts in
`manifest.yaml` are read out of the server's own `--log`, excluding the
`/__health` poll.

Raw probe output, including the per-row server logs, the request timelines that
back the retry counts, and the archived product-prompt captures under
`realistic-prompt/`: `plans/harness-expansion/probe-out/goose/`.
