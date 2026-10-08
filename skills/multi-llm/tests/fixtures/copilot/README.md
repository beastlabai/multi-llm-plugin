# copilot parser fixtures

Every `*.jsonl` here is a **real capture** of `copilot --output-format json`
stdout, taken against a stub (`--network none`, **no GitHub token and no
credential**): first in the containerized §6 probe on `@github/copilot@1.0.80`
(2026-08-17), and re-captured on **1.0.93** (2026-10-08) in the pinned image
through its PATH wrapper — every file except `exit-zero-failure.jsonl`, which
is still 1.0.80's bytes because 1.0.93 no longer produces that shape (see
`empty-response.jsonl`). They are raw bytes as copilot wrote them, not
hand-authored shapes — with the labelled exceptions below. `manifest.yaml`
records how each was produced and what moved between the two releases.

| Fixture | What it pins |
| --- | --- |
| `success.jsonl` | the envelope shape and the whole unwrap: the answer is a fenced suggestion-list array, and the test asserts the unwrapped `data` **is that document** — not `success is True` alone |
| `success-streamed.jsonl` | `--stream on` adds two ephemeral records and still emits the same terminal `assistant.message`, so **one parser serves both** stream modes |
| `success-silent.txt` | the `-s` bare-text shape. Kept as the record of why JSON mode is what ships: `-s` loses `exitCode`, the model echo and the whole structured error channel |
| `truncated.jsonl` | a **real** kill mid-flight: the stream ends at `model.call_start` with **no `result` record at all**, so the parser must not assume one exists |
| `truncated-mid-echo.jsonl` | a cut **inside** the 61 696-byte `user.message` record: the fragment `json.loads` rejects still carries our prompt, so `vendor_text` drops an unparseable line whose opener is that record's |
| `ephemeral-only.jsonl` | an all-noise run is a **failure**, not an empty-string success |
| `multi-assistant.jsonl` | **last**-wins, not first-wins and not concatenation. The first assistant message reads `FIRST ANSWER (must NOT be selected)` |
| `exit-zero-failure.jsonl` | **1.0.80's bytes**: the shape no marker can catch: exit 0, `result.exitCode: 0`, an empty assistant message, **no error record**. Detected structurally |
| `empty-response.jsonl` | the same empty 200 on 1.0.93: copilot now adds a `session.error` (`errorType: query`, "No response was returned…") and exits 1 |
| `auth-error.jsonl` | 401 → `session.error` with `errorType: authentication`, **on stdout with stderr empty** |
| `rate-limit.jsonl` | 429 → `errorType: rate_limit`; the message self-reports "retried 5 times (total retry wait time: 56.31 seconds)" (85.66 s at 1.0.80), which is the independent cross-check on `max_requests=6` |
| `invalid-model.jsonl` | 404 **naming the model** (`Model 'gpt-5.5' not found on provider …`), which is what makes a stray `:effort` suffix findable |
| `context-overflow.jsonl` | 400 context-length; the upstream text survives verbatim so `CONTEXT_OVERFLOW_MARKERS` matches and no retry budget is spent |
| `empty.jsonl` | zero bytes on both channels, distinguishable from a valid empty answer (no run wrote it; see the manifest's provenance) |

## The synthesized rows

`multi-assistant.jsonl` is **synthesized from the real record**: copilot emits a
single `assistant.message` per turn, and producing a second requires a tool loop
that the committed stub deliberately never drives. Recorded here rather than
faked silently — the rule it pins (last-wins) is real and is what the muse trap
cost this repo before.

`truncated-mid-echo.jsonl` is a **byte prefix** of the real `success.jsonl` —
every byte is copilot's own, at its own offset — cut 1 129 bytes into the echoed
`user.message` line (2 890 bytes in all at 1.0.93). `ephemeral-only.jsonl` is the
real `success.jsonl` with every non-ephemeral line removed, the rest verbatim.
All three are re-derived from each re-capture by the same recipe. `truncated.jsonl` is a real kill whose cut happened to land
on a record boundary, so no measured capture carries a mid-record cut, and the
offset here is chosen rather than measured. What it pins is live: those bytes
classified `rate_limited` off our own plan text until the unparseable-line rule
landed.

## `stdout-error` is not a separate file, on purpose

§6.2's `stdout-error` row — error text on stdout with stderr empty and a nonzero
exit — is the shape of **every** error fixture above, because `session.error` is
copilot's only error channel and it rides stdout. A separate file would be a
duplicate rather than a distinct case.
