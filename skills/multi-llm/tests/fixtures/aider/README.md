# aider parser fixtures

Every `*.stdout.txt` / `*.stderr.txt` here is a **real capture** of `aider`'s
streams under the argv `AiderProvider.build_command` emits, taken against a stub
in the containerized §6 probe (`aider-chat 0.86.2`, `--network none`, **no
credential** — `OPENAI_API_KEY` was the literal string
`stub-key-not-a-credential`, which litellm only requires to be non-empty). They
are raw bytes as aider wrote them; nothing here is hand-authored, and there is no
synthesized row.

They are graded by `tests/test_provider_fixture_rows.py` through
`manifest.yaml` — the generic §6.2 harness, not a hand-written class. The
manifest is the authority on what each row asserts; this file says *why* each
one is here.

Three facts hold across **every** row and are the reason several of the
assertions look the way they do:

- aider exits **0** on all eleven measured outcomes, success and failure alike;
- stderr is **0 bytes** on all eleven — the entire diagnosis rides stdout (the
  opencode trap, in a second CLI). The empty `*.stderr.txt` files are committed
  rather than omitted, because "stderr was empty" is the measurement;
- offline, litellm's import-time fetch of `model_prices_and_context_window.json`
  fails and aider prints that failure as the first line of every capture. Its
  text ends in *"Temporary failure in name resolution"*, a literal member of
  `base.CONNECTION_ERROR_MARKERS` — so `AiderProvider._TOLERATED_STDOUT_LINES` is
  load-bearing on every row here, not just the success one. `base.classify_streams`
  on these same bytes answers `connection_error` for eight of the ten rows.

| Fixture | What it pins |
| --- | --- |
| `success` | the **no-marker** shape: a non-reasoning reply is printed bare, with no `► **ANSWER**` at all, so the fallback path (`extract_json_from_text` over the whole of stdout) is what a normal run takes. The test asserts the unwrapped `data` **is** the suggestion-list document, and that `classify_failure` is `None` — which is the assertion that catches the tolerated-line excision breaking |
| `ephemeral-only` | an all-noise run is a **failure**, not an empty-string success: a reasoning block, then `► **ANSWER**` with nothing after it, then the `Tokens:` line the parser strips. Everything on disk is filtered; the verdict is `Empty answer after ANSWER marker` |
| `multi-assistant` | **last**-wins, not first-wins and not concatenation. Two THINKING/ANSWER segments, the first reading `FIRST ANSWER (must NOT be selected)` |
| `truncated` | a **real** cut mid-body (200 headers promising a full `Content-Length`, 60 bytes, socket closed): 9 aider attempts / **27 HTTP requests** / 78 s, then exit 0 with no answer. It classifies `rate_limited`, not `connection_error` — see below |
| `error-envelope` | a provider error object on a **200**. Worse than "served as the answer": litellm renders it `litellm.APIError: APIError: OpenAIException - ` with an **empty message**, so no provider text reaches any pattern table. **The one row where `exit_zero_failure_markers` is load-bearing** (`runner_error`) |
| `exit-zero-failure` | 5xx-forever: exit **0** on a run that produced nothing, and the whole §6.1.1 envelope in one capture — 27 requests, backoff 0.2 s → 32 s, 78.8 s wall |
| `auth-error` | 401 → **one** request, terminal. Not one of the ten `base.auth_error_patterns` appears anywhere in the capture; the classification comes entirely from the adapter's own added pair, matching aider's gloss *"The API provider is not able to authenticate you"* |
| `rate-limit` | 429-forever → aider **does** retry rate limits internally: 27 requests over 78.7 s, so an executor-level re-dispatch doubles a wait that already ran for a minute and a quarter |
| `invalid-model` | 404 **naming the model** — and naming it `stub-model`, not the `openai/stub-model` spec that was passed to `--model`, because litellm strips the provider prefix before dispatch. That is the shape a stray `:effort` suffix produces |
| `context-overflow` | 400 context-length, **swallowed**: aider catches litellm's `ContextWindowExceededError` and prints its own token-limit help, so `maximum context length` never appears. Measured classification: **`None`** — see below |

## Two rows that are "not applicable, because …"

Both carry their measurement in `manifest.yaml` rather than being absent (§6.3).

- **`stdout-error`** — the row wants a *nonzero* exit whose diagnosis is on
  stdout. aider exits **0** on every failure mode the rig can produce (401, 404,
  400, 500-forever, 429-forever, truncated body, zero-byte response, 200 carrying
  an error object), so the nonzero half has no producer at 0.86.2. The half that
  is the actual trap — diagnosis on stdout, stderr empty — is pinned by all seven
  failure rows.
- **`empty`** — zero bytes on both channels is unreachable while aider runs at
  all: the banner block is written before the first HTTP request, and the
  smallest capture in this set is **594 bytes**. Producing one would mean killing
  the process, which is the `timeout(1)` zero-byte-hang artifact §6 step 3
  forbids manufacturing.

## The zero-byte response is not a separate file, on purpose

`failserver --mode empty` (accept the request, write zero bytes, close) produces
stdout **byte-identical** to `truncated.stdout.txt` — md5
`6ab5e8a2fc5439c05732532716ed55d5`, same 27 requests, same 79 s. At aider's
stdout a truncated body and a zero-byte response are the same event
(`OpenAIException - Connection error.`), so a second file would be a duplicate
rather than a distinct case.

## What this set found

Three measured defects, recorded here rather than silently fixed — the fixtures
pin today's behaviour, so each fix is a visible edit to both the adapter and this
manifest.

1. **`context-overflow` classifies as nothing.** aider replaces the vendor's
   text with its own help, so no `CONTEXT_OVERFLOW_MARKERS` entry matches and the
   verdict is `None` — transport-shaped. The executor therefore spends its whole
   bounded-retry budget on the one condition §7.2 declares terminal. The one-line
   fix is a marker for aider's own wording (`exhausted context window`, or
   `hit a token limit`), the way vibe's `conversation context exceeds` was added.
2. **Every 5xx and every truncated stream reported as `rate_limited`.**
   **FIXED (C7).** aider's gloss for a server error is *"The API provider's
   servers are down or overloaded"*, and `overloaded` is a member of
   `base.rate_limit_patterns`, consulted before the server and connection
   tables. `AiderProvider._ADAPTER_GLOSS_LINES` now excises that one
   runner-written line before classification, so the fix is narrowed to where
   the gloss is *written* rather than weakening a table every adapter shares —
   a vendor that really says `overloaded` (Anthropic's 529) still classifies
   `rate_limited`, and `base.py` is byte-identical. `truncated` is now
   `connection_error` and `exit-zero-failure` `runner_error`; both rows pin it.
   One residual, recorded not fixed: `exit-zero-failure` lands on
   `runner_error` rather than the more specific `server_error`, because
   `base.SERVER_ERROR_MARKERS` spells `internal server error` with separators
   while litellm prints `InternalServerError` and the run carries no bare
   `500`. Both map to the same rung, so only the `Reason:` line differs;
   closing it is an edit to a shared table and out of C7's scope.
3. **The parser's error message is 500 characters of unrelated noise.**
   `parse_output` snippets `stdout[:500]`, and offline the first ~460 of those
   are litellm's cost-map DNS failure plus the banner — so the operator-facing
   error for a failed auth run reads
   `No JSON found in aider output: HTTPSConnectionPool(host='raw.githubusercontent.com' …`
   and never reaches `litellm.AuthenticationError`. `_drop_tolerated_lines` is
   already the fix; it is just not applied on this path.

## Reproducing

One container, no credential, no network. The driver is a throwaway shell
script written into a scratch mount — deliberately not committed, because the
two servers it drives are, which is the whole point of `failserver.py` living in
`tests/container/` instead of a scratch directory:

```
docker run --rm --network none \
  -v "$PWD/skills/multi-llm/tests/container:/rig:ro" \
  -v "$PWD/<scratch>:/work/cap" \
  --entrypoint /bin/bash mllm-e2e:local /work/cap/capture.sh
```

Per row the script: starts one server on 127.0.0.1:8080 — `python3
/rig/stub_server.py --port 8080 --log … --scenario …` for the three answering
rows, `python3 /rig/failserver.py --port 8080 --mode
{trunc,errbody,500,401,429,404,ctx,empty} --log …` for the rest — polls
`/__health`, then runs `AiderProvider.build_command`'s argv verbatim with
`OPENAI_API_BASE=http://127.0.0.1:8080/v1`, `OPENAI_API_KEY=<any non-empty
string>` and `get_env`'s `BROWSER=true` / `COLUMNS=10000`, redirecting stdout and
stderr to separate files and recording `$?` and the wall time. `cwd` is an
empty `git init` tree, `HOME` is a fresh directory. Do **not** wrap the run in
`timeout`. Total wall time ≈ 8 minutes; the seven failure rows are where all of
it goes, and the per-row `requests:` counts in `manifest.yaml` are read out of
the server's own `--log`.
