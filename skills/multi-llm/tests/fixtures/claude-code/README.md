# claude-code parser fixtures

Every `*.stdout.txt` / `*.stderr.txt` here is a **real capture** of `claude`'s
streams under the argv `ClaudeCodeProvider.build_command` emitted at capture
time (see `manifest.yaml`'s `method:` for the one pair it has gained since, and
why these bytes are still the measurement), taken against a
stub in the containerized §6 probe (`@anthropic-ai/claude-code 2.1.232`,
`--network none`, **no credential** — `ANTHROPIC_API_KEY` was the PATH wrapper's
fake per-exec session key `mllm-stub-<id>`, which is meaningful only on
127.0.0.1). They are raw bytes as claude-code wrote them; nothing here is
hand-authored, and there is no synthesized row.

They are graded by `tests/test_provider_fixture_rows.py` through
`manifest.yaml` — the generic §6.2 harness, not a hand-written class. The
manifest is the authority on what each row asserts; this file says *why* each one
is here.

Four facts hold across **every** row and are the reason several assertions look
the way they do:

- **stderr is 0 bytes on all eleven** — success and failure alike. The whole
  diagnosis rides stdout, inside the result envelope. The empty `*.stderr.txt`
  files are committed rather than omitted, because "stderr was empty" is the
  measurement. Every failure row here is therefore also the `stdout-error` trap
  (§6 step 3's opencode shape) in miniature.
- **The exit code is authoritative at 2.1.232.** Eight injected failure modes,
  eight exits of **1**; both answering rows exit **0**. That is what makes the
  adapter's empty `exit_zero_failure_markers` correct *today*, measured rather
  than assumed — and it is the cell that has to move first if a later release
  stops reporting through the exit code.
- **Two requests per exec, not one.** The first is claude-code's own
  session-title generator: tool-free, **same model**, same surface, so only
  `has_tools` separates the two in the stub's log. Every `requests:` count is
  both calls, and on a failure-injection run the injected error hits the *title*
  call first.
- **The envelope measured is the pinned one.** `get_env` sets
  `CLAUDE_CODE_MAX_RETRIES=3` (adapter declares `max_requests=4`) and
  `API_TIMEOUT_MS=300000`, and the wrapper's `ENV_CARRY_EXACT` carries both into
  the child — so these numbers are the product's configuration, not the CLI's
  out-of-the-box 11.

| Fixture | What it pins |
| --- | --- |
| `success` | the single-line result envelope, unwrapped by `parse_result_envelope`'s fast path. The test asserts the unwrapped `data` **is** the suggestion-list document and that `classify_failure` is `None`. Also pins that the `haiku` alias is resolved to `claude-haiku-4-5-20251001` **before the request leaves** — the wire never carries the alias |
| `ephemeral-only` | an all-noise run is a **failure**, not an empty-string success. The CLI emits one record and everything in it except `result` is bookkeeping the parser discards; the stub answered with an empty assistant message, so 1,164 bytes on disk are 1,164 bytes of filtered noise. `is_error: false`, `subtype: "success"`, **exit 0** — the only thing wrong is that the answer is empty |
| `multi-assistant` | **first**-wins, and the first document is the decoy. Two JSON arrays in one `result`, the first reading `FIRST ANSWER (must NOT be selected)` — see defect 1 |
| `truncated` | a **real** cut mid-record on the *anthropic* surface (SSE cut inside `content_block_start`), then claude-code's own **non-streamed retry**, cut again. 4 requests = 2 calls × (streamed + non-streamed); exit 1; gloss `Connection to the API was lost (ECONNRESET)` |
| `error-envelope` | a provider error object on a **200**. claude-code does *not* serve it as the answer — it detects the shape and substitutes `API returned an empty or malformed response (HTTP 200) …`, so the provider's text is lost rather than mis-served. Note the self-contradicting envelope: `is_error: true` beside `subtype: "success"` |
| `stdout-error` | 5xx-forever: nonzero exit, diagnosis entirely on stdout, stderr 0 bytes — and the whole pinned retry envelope in one capture, 8 requests over 4.4 s |
| `auth-error` | 401 → **retried 8 times**, and classified **`server_error`**, not `auth_failed`. The most consequential finding in this set — see defect 3 |
| `rate-limit` | 429-forever → claude-code **does** retry rate limits internally (7 requests / 4.4 s), so an executor-level re-dispatch of a rate-limited slot repeats a chain the CLI already ran (§7.5 rule 6b). The one failure row classified on the vendor's words rather than claude-code's prefix |
| `invalid-model` | 404 **naming the model** — and naming the *resolved* id `claude-haiku-4-5-20251001`, not the `haiku` spec. Measured classification: **`None`** — see defect 4 |
| `context-overflow` | 400 context-length, passed through **unedited**, so `maximum context length` reaches the classifier and the terminal category wins. **2 requests, not 8**: a 400 is not retried |
| `zero-byte-response` | beyond §6.2's twelve, and the measurement behind the `empty` cell: a server that writes zero bytes still leaves 853 bytes of well-formed envelope. Distinct wording from `truncated` (`Connection dropped` vs `Connection to the API was lost`) and a different retry count (8 vs 4), which is why it is its own file |

## Two rows that are "not applicable, because …"

Both carry their measurement in `manifest.yaml` rather than being absent (§6.3).

- **`exit-zero-failure`** — no producer at 2.1.232: 401, 429, 404, 400, 500,
  a truncated stream, a zero-byte response and a 200 carrying an error object all
  exit **1**. The nearest exit-0 non-answer is `ephemeral-only`, and it prints no
  error at all, so `exit_zero_failure_markers` would have nothing to match.
- **`empty`** — zero bytes on both of the CLI's channels is unreachable while
  claude-code runs at all: `--output-format json` always prints the envelope, and
  the smallest capture here is **853 bytes**. Manufacturing one would mean
  killing the process, the `timeout(1)` artifact §6 step 3 forbids. The
  server-side half is captured as `zero-byte-response`.

## What this set found

Five measured defects, recorded here rather than silently fixed — the fixtures
pin today's behaviour, so each fix is a visible edit to both the adapter and this
manifest.

1. **`multi-assistant` selects the FIRST document, and it is the decoy.**
   `parse_result_envelope` falls through to
   `extract_json_from_text(prefer_arrays=True)`, which returns the first array it
   finds. Two answers in one `result` — a reasoning model that "shows its work"
   as a draft array, or a retry appended after a correction — are resolved
   first-wins. aider documents and implements **last**-wins (`rsplit` on its
   ANSWER marker); claude-code has no rule at all. This is the muse trap in a
   third CLI.
2. **`parse_output` ignores `is_error`.** The envelope claude-code prints on a
   failure carries `is_error: true`, `terminal_reason: "api_error"` and an
   `api_error_status` — three unambiguous machine-readable verdicts — and the
   parser reads none of them; it looks only at `result`. It is safe today only
   because the substituted error sentences carry no JSON. One `wrapper.get(
   "is_error")` check in `parse_result_envelope` would make the whole failure set
   self-reporting instead of relying on "no JSON was extractable".
3. **A 401 classifies as `server_error`.** claude-code's gloss is
   *"Failed to authenticate. API Error: 401 …"*; none of `base.auth_error_patterns`'
   ten entries matches it (`unauthorized` is absent from the text, `invalid api
   key` is spelled *Incorrect API key provided*), while `\bapi error\b` in
   `SERVER_ERROR_MARKERS` matches the `API Error:` prefix claude-code puts on
   **every** failure. Since auth is consulted before the server table, adding
   `"failed to authenticate"` (and/or `"incorrect api key"`) to the adapter's
   `auth_error_patterns` fixes it — the same shape as aider's added pair. Until
   then a logged-out claude-code slot is retried as a transient server fault.
   The 401 is *also* retried 8 times by the CLI itself, so the wasted budget is
   doubled.
4. **`invalid-model` classifies as nothing.** claude-code replaces the vendor's
   `The model \`X\` does not exist` with *"There's an issue with the selected
   model (X). It may not exist or you may not have access to it."*, and no
   `INVALID_MODEL_MARKERS` entry matches that wording — so a wrong model slug is
   transport-shaped and burns the whole retry budget. The one-line fix is a
   marker for the CLI's own phrasing, the way vibe's `conversation context
   exceeds` was added.
5. **`\bapi error\b` makes `server_error` the default verdict for this CLI.**
   Every failure claude-code prints starts `API Error:`, and that string is a
   `SERVER_ERROR_MARKERS` entry, so the server table absorbs the connection
   failures (`truncated`, `zero-byte-response`) and the auth failure alike. Three
   of the five rows that reach it are not server errors. Defects 3 and 4 are the
   two that change the ladder's behaviour; this one is why the `Reason:` line a
   user acts on is wrong on five rows out of eight.

## Reproducing

One container, no credential, no network. The driver is a throwaway script under
`plans/harness-expansion/probe-out/claude-code/` — deliberately not committed
here, because the two servers it drives are.

```
docker run --rm --network none --user 10001:10001 --read-only \
  --tmpfs /work/homes:rw --tmpfs /work/sample-project:rw --tmpfs /tmp:rw \
  -v "$PWD/skills/multi-llm/tests/container:/rig:ro" \
  -v "$PWD/plans/harness-expansion/probe-out/claude-code:/work/out:rw" \
  -e MLLM_STUB_PORT=8099 -e MLLM_REDIRECT_YAML=/rig/redirect.yaml \
  -e MLLM_REAL_DIR=/opt/mllm/real -e MLLM_HOMES_ROOT=/work/homes \
  --entrypoint /bin/bash mllm-e2e:local -lc 'python3 /work/out/capture.py'
```

Per row the driver: seeds `/work/homes/claude-code/` from
`tests/container/home_seed/claude-code/` (the `.claude.json` that makes the run
non-interactive — the row's `seed_files: []` is misleading, the wrapper copies
the whole directory), starts one server on 127.0.0.1:8099 — `stub_server.py
--scenario …` for the three answering rows, `failserver.py --mode
{trunc,errbody,500,401,429,404,ctx,empty}` for the rest — polls `/__health` with
**urllib** (there is no `curl` in the image), then runs
`ClaudeCodeProvider.build_command`'s argv through the PATH wrapper
`/opt/mllm/bin/claude`, which applies redirect.yaml's whole claude-code row
itself. Do **not** wrap the run in `timeout`. Total wall time ≈ 25 s; the two
fail-forever rows are ~4.5 s each. The per-row `requests:` counts in
`manifest.yaml` are read out of the server's own `--log`, with the driver's
`/__health` poll excluded.
