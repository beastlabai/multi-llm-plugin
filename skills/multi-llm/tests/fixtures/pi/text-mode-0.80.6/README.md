# pi parser fixtures — text mode, 0.80.6 (HISTORICAL, UNGRADED)

> **Historical.** This is the 2026-08-19 capture of pi **0.80.6** under the old
> text-mode argv (`pi --no-session -p --model …`). pi now runs `--mode json`
> (plans/opencode-validator-false-rate-limit-and-denied-turn.md §7), and the
> graded set is the json-mode one in the parent directory. Nothing here is
> collected by the capture-row globs (`test_provider_fixture_rows._discover`,
> `test_stderr_diagnosis._capture_rows`, `test_providers._provider_capture_rows`
> read only `tests/fixtures/pi/*`), so this manifest is no longer graded. The one
> exception is `success.*`, which `tests/test_pi_json_mode.py` reads **by explicit
> path** to grade `PiProvider.parse_output`'s text-mode back-compat leg (a pi that
> ignores `--mode`). Kept, not deleted, because it is the measurement the json-mode
> set is compared against (`TestNoPiClassifyCellWorsened`).

Every `*.stdout.txt` / `*.stderr.txt` here is a **real capture** of `pi`'s
streams under the argv `PiProvider.build_command` emits, taken against a stub in
the containerized §6 probe (`pi 0.80.6`, `--network none`, **no credential** —
the only key anywhere is the wrapper's per-exec `mllm-stub-<id>` string in
`STUB_API_KEY`, which means something on 127.0.0.1 and nowhere else). They are
raw bytes as pi wrote them; nothing here is hand-authored and there is no
synthesized row.

They are graded by `tests/test_provider_fixture_rows.py` through
`manifest.yaml` — the generic §6.2 harness, not a hand-written class. The
manifest is the authority on what each row asserts; this file says *why* each one
is here.

**One fact dominates the whole set, and it is the inverse of the opencode trap:
pi writes ZERO bytes to stdout on every failure.** The diagnosis goes to stderr,
prefixed with the HTTP status and carrying the vendor's JSON error object
verbatim, and the exit code is 1 — precisely what `PiProvider`'s docstring
claims, now measured rather than asserted. Ten of the eleven `*.stdout.txt` files
here are therefore **0 bytes**, and they are committed rather than omitted
because "pi wrote nothing on this channel" is the measurement. The practical
consequence: `parse_output` answers `Empty output` on every failure row, and
every distinction between the rows lives in `classify_failure`, which reads
stderr.

| Fixture | What it pins |
| --- | --- |
| `success` | the whole shape — 151 bytes of stdout that **are** the answer, with no envelope, no banner, no marker and no accounting footer, and an empty stderr. One HTTP request; pi fetches no catalog. The test asserts the unwrapped `data` **is** the suggestion-list document and that `classify_failure is None` |
| `truncated` | a **real** cut mid-body: pi retries **four** times over 14.6 s and reports the lot in five words — `Stream ended without finish_reason` — with stdout empty. Classified as **nothing**; see below |
| `error-envelope` | a provider error object on a **200**, indistinguishable from the cut stream: same five words, same four retries, and not one byte of the vendor's message anywhere |
| `exit-zero-failure` | the worst possible version of the shape: exit **0**, stdout 0 bytes, stderr 0 bytes, one request, 0.5 s. No marker, because there are no bytes — and `PiProvider` declares no `exit_zero_failure_markers` that could fire |
| `empty` | zero bytes on both channels, **real** and produced with no `timeout(1)` anywhere. §6.2 wants it "distinguishable from a valid empty answer"; at pi it is not — see below |
| `auth-error` | 401 → **one** request, terminal. Classified from the machine-readable `"code":"invalid_api_key"`, not from the English sentence: *"Incorrect API key provided: …"* matches nothing in `base.auth_error_patterns` |
| `rate-limit` | 429-forever → pi **does** retry rate limits internally: 4 requests over 14.6 s, so an executor-level re-dispatch doubles a wait the CLI already served |
| `invalid-model` | 404 **naming the model** — `stub-model`, the id half of the `stub/stub-model` spec, since pi resolves the provider segment against `models.json` before dispatch |
| `context-overflow` | 400 context-length, **swallowed whole**: no bytes on either channel, exit **0**, 0.5 s. The terminal classification §7.2 requires is unreachable |
| `zero-body` (extra) | the one failure pi separates from a truncated body — `Connection error.` — and the one text in this set that lands on a shared marker table rather than on vendor JSON |
| `server-error` (extra) | 5xx-forever, §6.1.1's experiment 2 and pi's internal-retry envelope in one capture: **4 requests over 14.6 s**. Classified as nothing; see below |

## Three rows that are "not applicable, because …"

All three carry their measurement in `manifest.yaml` rather than being absent
(§6.3).

- **`ephemeral-only`** — pi has no noise channel to be all of. Its stdout is the
  final assistant text and nothing else: no event records, no progress lines, no
  thinking block, no `Tokens:` footer. Measured directly: driven against a stub
  whose answer is the **empty string**, pi wrote 0 bytes to both channels and
  exited 0, i.e. the all-noise shape degenerates into `empty`, which *is*
  committed.
- **`multi-assistant`** — there is no envelope to hold a second message. pi print
  mode emits the final assistant text as bare bytes, so first-wins, last-wins and
  concatenation are the same operation. Measured on a **multi-turn** run rather
  than assumed: the live-lane capture in
  `tests/container/runs/20260819T145308Z/live-pi/live-provider.json` is a run in
  which pi called its write tool and *then* answered, and its entire stdout is
  the one line `JSON written to …`.
- **`stdout-error`** — inverted at pi. The row wants the diagnosis on stdout with
  stderr empty; pi does the exact opposite on all nine failure modes. What the
  row exists to prevent — a diagnosis being read as an answer — cannot happen
  here, because the parser is never handed one.

## Three rows, one set of bytes

`empty`, `context-overflow` and `exit-zero-failure` are three separate real
captures that are **byte-identical**: 0 bytes on stdout, 0 bytes on stderr,
exit 0. That identity is the finding, not an accident of the capture:

- a model that legitimately answers with the empty string,
- a 400 `context_length_exceeded` that pi swallows entirely,
- and any other silent exit-0 failure

are the same event at pi 0.80.6, on every signal available to the executor. Each
row is committed separately because each records a different thing the absence of
bytes costs.

## What this set found

Three measured defects, recorded here rather than silently fixed.

1. **A swallowed context overflow is invisible.** `--mode ctx` produces exit 0
   and no output at all, so `classify_streams` short-circuits on the empty
   haystack and answers `None`. The executor then spends its whole bounded-retry
   budget on the one condition §7.2 declares terminal, and the operator sees
   `Empty output` with no cause. There is no adapter-side pattern that can fix
   this — the fix has to be upstream in pi, or the adapter has to treat
   *stdout-empty + exit-0* as its own terminal category.
2. **A cut stream, a 200-with-an-error-object and a 5xx all classify as
   nothing.** `Stream ended without finish_reason` matches no marker table, and
   OpenAI's *"The server had an error while processing your request"* matches no
   member of `base.SERVER_ERROR_MARKERS` either (the bare `500: ` prefix pi adds
   is not a status-code pattern). Three of pi's five retryable failures are
   therefore transport-shaped to the ladder.
3. **`parse_output` ignored stderr entirely.** **FIXED (C4).** Every failure row
   used to return the same two words — `Empty output` — while the vendor's full
   JSON error object sat unused on the other stream, making the operator-facing
   message for a 401, a 429 and a 404 identical and uninformative.
   `PiProvider.describe_failure` now strips the bare `NNN: ` HTTP-status prefix,
   parses the remainder as the vendor's JSON object and reports its `message`
   with its `code` and the stripped status (4 rows: `auth-error`,
   `invalid-model`, `rate-limit`, `server-error`); where the remainder is not
   JSON it reports the stderr text verbatim, which is already the whole
   diagnosis (3 rows: `error-envelope`, `truncated`, `zero-body`).
   `parse_output`'s error branch names the same text. **The three rows with
   nothing on either channel keep `Empty output` and return `None` from
   `describe_failure`** — `context-overflow` (pi's silent 400, recorded as an
   unfixable measurement in C11), plus `empty` and `exit-zero-failure`, both
   byte-identical to it. That is the measurement, not a gap: nothing is
   invented from an exit code.

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

Per row the script copies `/opt/mllm/rig/home_seed/pi/.` into `/work/homes/pi/`
(the wrapper's seed source; `.pi/agent/models.json` is the **only** carrier of
the base URL, and its `apiKey` must be present even as a dummy or the model is
filtered out of the selectable set entirely), starts one server on
127.0.0.1:8099 — `stub_server.py --scenario <inline answer>` for the answering
rows, `failserver.py --mode {trunc,errbody,500,401,429,404,ctx,empty}` for the
rest — polls `/__health` **with urllib** (there is no curl in the image), then
runs `/opt/mllm/bin/pi` (which *is* `wrapper.py`, and applies the whole
redirect.yaml row itself, including `PI_CODING_AGENT_DIR`, which pi reads *ahead
of* `$HOME`). The prompt must **demand stdout**: pi declares
`supports_file_output = True`, and a file-output prompt makes it print only
`JSON written to …`. Do **not** wrap the run in `timeout`. Total wall time
≈ 90 s for the eleven rows, almost all of it in the four retrying rows; the
`requests:` counts in `manifest.yaml` are read out of the server's own `--log`,
excluding the `/__health` poll.

Raw probe output, including the per-row server logs and the wrapper call log:
`plans/harness-expansion/probe-out/pi/`.
