# gemini parser fixtures

Every `*.stdout.txt` / `*.stderr.txt` here is a **real capture** of `gemini`'s
streams under the argv `GeminiProvider.build_command` emits, taken against a stub
in the containerized §6 probe (`@google/gemini-cli 0.25.2`, `--network none`,
**no credential** — `GEMINI_API_KEY` was the PATH wrapper's fake per-exec session
key, meaningful only on 127.0.0.1). They are raw bytes as gemini wrote them;
nothing here is hand-authored, and there is no synthesized row.

They are graded by `tests/test_provider_fixture_rows.py` through `manifest.yaml`.
The manifest is the authority on what each row asserts; this file says *why* each
one is here, and what the set found.

Four facts govern every row:

- **On every failure gemini writes ZERO bytes to stdout** and puts the whole
  diagnosis on stderr — the exact inverse of the opencode trap, and the reason
  the operator-facing error is identical (`JSON parse error: Expecting value:
  line 1 column 1 (char 0)`) for a 401, a 429, a 404 and a 400 alike:
  `parse_output` never reads stderr.
- **stderr carries two things**: free-form JS prose with a Node stack into
  `@google/genai` / `@google/gemini-cli-core`, then a machine-readable envelope
  `{"session_id", "error": {"type", "message", "code"}}` whose **`message` is
  always the literal `[object Object]`**. The only structured field gemini offers
  is destroyed by an upstream stringification bug.
- **The exit code is the envelope's `code` truncated to 8 bits**: 401 → **145**,
  400 → **144**, 500 → **244**, `code: 1` → 1. An exit code from this CLI is a
  status, not a category; `!= 0` is the only safe reading.
- **One request per exec** — gemini fires no session-title side call — except on
  the two fail-forever rows, where its own backoff runs **10 requests over ~4
  minutes** inside the one awaited subprocess.

| Fixture | What it pins |
| --- | --- |
| `success` | the pretty-printed `{"session_id", "response", "stats"}` envelope, unwrapped by `parse_output`. The test asserts the unwrapped `data` **is** the suggestion-list document. `stats` records `totalCalls: 0` and `totalLinesAdded: 0` — answered on stdout, wrote nothing |
| `ephemeral-only` | an all-noise run is a **failure**, not an empty-string success: `"response": ""` inside 802 bytes of `stats` the parser discards, exit 0, stderr empty. `totalRequests: 4` — gemini retried the empty answer four times and still called it a success |
| `multi-assistant` | **last**-wins, and the last document is the answer; the first is the decoy (`FIRST ANSWER (must NOT be selected)`), still in the bytes so the row tests selection — see defect 4 |
| `truncated` | a **real** cut mid-record on the *gemini* surface: the whole answer is ONE 291-byte SSE frame and the cut lands inside it, so `@google/genai` reports `Incomplete JSON segment at the end`. **One request** — a cut stream is not retried |
| `error-envelope` | a provider error object on a **200**, which gemini never sees as an error at all: it asked for `text/event-stream`, got one JSON object, and failed at the SSE framing layer with the same `Incomplete JSON segment` as `truncated`. The provider's text is lost, not mis-served |
| `exit-zero-failure` | the one gemini failure that **exits 0 while reporting success** — a `write_file` tool call that the registry rejects, after which the model's confirmation sentence is printed as the answer. See defect 1 |
| `auth-error` | 401 → one request, terminal, exit **145**. Measured classification **`None`**: *Incorrect API key provided* is not `invalid api key`, and `UNAUTHENTICATED` is not `unauthorized` |
| `rate-limit` | 429-forever → **10 requests over ~200 s**, each announced as `Attempt N failed with status 429. Retrying with backoff...`. The §6.1.1 measurement for this CLI, and the reason §7.5 rule 6(b) forbids an identical re-dispatch |
| `invalid-model` | 404 **naming the model** — `gemini-2.5-flash`, correct only because failserver now reads the model from the PATH (see "A second rig fix" below). One of only four categories this CLI's failures reach |
| `context-overflow` | 400 context-length, passed through unedited, so `maximum context length` reaches the classifier. Exit **144**, and the one envelope carrying a real status rather than `code: 1` |
| `server-error` | beyond §6.2's twelve: 5xx-forever, **10 requests / ~225 s**, exit **244**, 14 KB on stderr and 0 on stdout — and classified as **nothing** (defect 5) |
| `zero-byte-response` | beyond §6.2's twelve, and the measurement behind the `empty` cell: a server writing zero bytes still leaves 1,582 bytes of stderr, whose words are undici's generic `exception TypeError: fetch failed sending request` |

## Two rows that are "not applicable, because …"

Both carry their measurement in `manifest.yaml` rather than being absent (§6.3).

- **`stdout-error`** — the row wants error text on stdout with stderr empty.
  gemini is the mirror image: eight injected failure modes, eight captures of
  **0 bytes on stdout**, everything on stderr. What the row exists to catch is
  real here and worse; see defect 2.
- **`empty`** — zero bytes on *both* channels never happened. Every failure is
  0 bytes on stdout and non-empty on stderr; every success is the reverse. One
  channel is always empty and neither run type leaves both empty. The nearest
  producer, `failserver --mode empty`, is committed as `zero-byte-response`.

## §14.7 gap 5, closed: `supports_file_output = True` is **wrong** for gemini

gemini was the last unmeasured `supports_file_output = True` adapter. It is
measured now, offline, at $0.00, and the declaration does not hold.

`stub_server.py` never emits a tool call (its own docstring), so the §6.2 session
alone can only show gemini answering on stdout — which measures the stub, not the
CLI. The other half is a throwaway probe kept beside the captures at
`plans/harness-expansion/probe-out/gemini/gap5_probe.py`, run twice:

1. **observe** — record the request body. Under the adapter's exact argv
   (`gemini --output-format json --model M <prompt>`, **no** `--yolo`, no
   `--approval-mode`) gemini declares **nine** tools to the model, and the
   complete list is: `list_directory`, `read_file`, `search_file_content`,
   `glob`, `activate_skill`, `save_memory`, `google_web_search`, `write_todos`,
   `delegate_to_agent`. There is **no `write_file`**, no `replace`, no
   `run_shell_command`. The saved body is
   `probe-out/gemini/gap5-observe/req-1.json`.
2. **call** — answer turn 1 with a `functionCall` for `write_file` anyway. The
   result is the committed `exit-zero-failure` row:

   ```
   Error executing tool write_file: Tool "write_file" not found in registry.
   Tools must use the exact names that are registered. Did you mean one of:
   "read_file", "write_todos", "activate_skill"?
   ```

   `stats.tools.byName.write_file` = `{count: 1, success: 0, fail: 1}`,
   `files.totalLinesAdded: 0`, the target file does not exist — and gemini
   **exits 0** printing the model's `JSON written to …` confirmation as its
   answer.

So gemini belongs in the same class as `cursor-agent --mode ask`, `aider /ask`,
`qwen-code --max-tool-calls 0` and `vibe --agent plan`: a CLI whose shipped
invocation cannot write the file the `{output_location}` block demands. It is
worse than those four in one respect — they refuse in words, gemini exits 0
having *confirmed* a write that never happened.

**The fix is the declaration, not the argv.** `supports_file_output = False`
makes `render_output_location` ask for the array on stdout, which is the channel
gemini does have and the one `invoke_with_file_output` already falls back to.
Adding `--yolo` or `--approval-mode=yolo` to buy the write back is the write
bypass §3.1 removed and must not be the answer. This is left unchanged here on
purpose: the roster in `tests/test_provider_housekeeping.py` (`STOPS_FILE_WRITES`
and its sibling table) is keyed on *restriction-shaped argv atoms*, and gemini's
argv carries none — the read-only tool set is the **default**, not a flag — so
flipping the declaration needs that table taught about a fourth reason, which is
a change to a file this capture does not own.

## A second rig fix this set forced

`failserver.py`'s error bodies formatted `{model}` from `body["model"]`. gemini is
the one surface that carries its model in the **path**
(`/v1beta/models/<model>:streamGenerateContent`) and never in the body, so the
404 named the `stub-model` default — and the whole point of the `invalid-model`
row is that the failure NAMES the model. `failserver.model_of()` now reads the
path for that surface. Both fixes to that file (this one and the `trunc`
per-surface rendering) leave the chat surface byte-identical, which is what keeps
aider's committed `truncated.stdout.txt` grading unchanged.

## What this set found

Five measured defects, recorded rather than silently fixed. Defect 4 has since
been fixed in the shared extractor, and its entry records what changed.

1. **`supports_file_output = True` is wrong** — the whole of the section above.
   The adapter asks gemini to write a file with a tool gemini does not register,
   and the run reports success.
2. **`parse_output` ignored stderr, where 100 % of gemini's diagnosis lives.**
   **FIXED (C4).** Nine failure captures used to yield nine copies of
   `JSON parse error: Expecting value: line 1 column 1 (char 0)` as the
   operator-facing error, with the vendor's 401/429/404/400 text sitting in the
   bytes and never reaching the user. `GeminiProvider.describe_failure` now
   reads stderr through a three-tier precedence — the last `ApiError` blob's
   `error.message`/`error.code` (4 rows), else the prose sentence following the
   `Full report available at:` path (4 rows), else the first stderr line with
   the stack stripped (1 row: `exit-zero-failure`, the only capture with bytes
   on both channels) — and `parse_output`'s error branch names the same text.
   The Node stack and the per-run `/tmp/mllm-inv/` path are dropped and the
   result is capped to one line, so a 15 KB stderr cannot become the error
   string. `error.message` is never reported: at 0.25.2 it is the literal
   `[object Object]`. `classify_failure` already read both streams, which is
   why four of the nine landed on a category even before this — the parser was
   the half that was blind.
3. **A 401 classified as `None`.** **FIXED (C3).** Google's status word is
   `UNAUTHENTICATED` (not `unauthorized`) and the vendor sentence is *Incorrect
   API key provided* (not `invalid api key`), so nothing in
   `base.auth_error_patterns` matched and a logged-out gemini burned the whole
   bounded-retry budget. Two patterns were added to
   `GeminiProvider.auth_error_patterns`, the way aider's pair was — and swept
   against all 231 committed capture files across 16 providers to confirm they
   match nothing else, including every echoing adapter's success capture.
4. **`multi-assistant` selected the FIRST document, and it was the decoy —
   FIXED** — `extract_json_from_text(prefer_arrays=True)` returned the first
   array it found. gemini has no message boundary to hang a per-adapter rule on
   (every turn is concatenated into one `response` string), which is why the fix
   is one rule in the shared helper: **the last document wins, at every level**,
   with shape preference consulted before position and a contained candidate
   never outranking the one that contains it. That is aider's documented
   last-wins rule generalised from the envelope to the text inside it, so
   claude-code, grok and gemini now select the answer instead of the decoy. The
   row moved with the fix and still carries the decoy in `raw_contains`.
5. **A 5xx classifies as `None`.** gemini surfaces *"The server had an error
   while processing your request"*, which matches no `SERVER_ERROR_MARKERS` entry
   (the phrase `server error` never appears, and the `5\d{2}` pattern wants the
   status adjacent to the word). 10 requests and 232 s of the CLI's own retrying
   are then followed by the harness treating the run as transport-shaped.

## Reproducing

```
docker run --rm --network none --user 10001:10001 --read-only \
  --tmpfs /work/homes:rw --tmpfs /work/sample-project:rw --tmpfs /tmp:rw \
  -v "$PWD/skills/multi-llm/tests/container:/rig:ro" \
  -v "$PWD/plans/harness-expansion/probe-out/gemini:/work/out:rw" \
  -e MLLM_STUB_PORT=8099 -e MLLM_REDIRECT_YAML=/rig/redirect.yaml \
  -e MLLM_REAL_DIR=/opt/mllm/real -e MLLM_HOMES_ROOT=/work/homes \
  --entrypoint /bin/bash mllm-e2e:local -lc 'python3 /work/out/capture.py'
```

Per row the driver starts one server on 127.0.0.1:8099 — `stub_server.py
--scenario …` for the three answering rows, `failserver.py --mode
{trunc,errbody,500,401,429,404,ctx,empty}` for the rest — polls `/__health` with
**urllib** (there is no `curl` in the image), then runs
`GeminiProvider.build_command`'s argv through the PATH wrapper
`/opt/mllm/bin/gemini`, which applies redirect.yaml's gemini row itself. `$HOME`
must exist or gemini crashes before its first request. Do **not** wrap the run in
`timeout`. Total wall time ≈ 8 minutes, of which 7 are the two fail-forever rows. The `exit-zero-failure` row uses `gap5_probe.py` instead, twice
(`--mode observe`, then `--mode call`); the `requests:` counts come from each
server's own `--log`, with the driver's `/__health` poll excluded.
