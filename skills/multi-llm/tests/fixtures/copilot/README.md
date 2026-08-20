# copilot parser fixtures

Every `*.jsonl` here is a **real capture** of `copilot --output-format json`
stdout, taken against a stub in the containerized §6 probe
(`@github/copilot@1.0.80`, `--network none`, **no GitHub token and no
credential**). They are raw bytes as copilot wrote them, not hand-authored
shapes — with the one labelled exception below.

| Fixture | What it pins |
| --- | --- |
| `success.jsonl` | the envelope shape and the whole unwrap: the answer is a fenced suggestion-list array, and the test asserts the unwrapped `data` **is that document** — not `success is True` alone |
| `success-streamed.jsonl` | `--stream on` adds two ephemeral records and still emits the same terminal `assistant.message`, so **one parser serves both** stream modes |
| `success-silent.txt` | the `-s` bare-text shape. Kept as the record of why JSON mode is what ships: `-s` loses `exitCode`, the model echo and the whole structured error channel |
| `truncated.jsonl` | a **real** kill mid-flight: the stream ends at `model.call_start` with **no `result` record at all**, so the parser must not assume one exists |
| `ephemeral-only.jsonl` | an all-noise run is a **failure**, not an empty-string success |
| `multi-assistant.jsonl` | **last**-wins, not first-wins and not concatenation. The first assistant message reads `FIRST ANSWER (must NOT be selected)` |
| `exit-zero-failure.jsonl` | the shape no marker can catch: exit 0, `result.exitCode: 0`, an empty assistant message, **no error record**. Detected structurally |
| `auth-error.jsonl` | 401 → `session.error` with `errorType: authentication`, **on stdout with stderr empty** |
| `rate-limit.jsonl` | 429 → `errorType: rate_limit`; the message self-reports "retried 5 times (total retry wait time: 85.66 seconds)", which is the independent cross-check on `max_requests=6` |
| `invalid-model.jsonl` | 404 **naming the model** (`Model 'gpt-5.5' not found on provider …`), which is what makes a stray `:effort` suffix findable |
| `context-overflow.jsonl` | 400 context-length; the upstream text survives verbatim so `CONTEXT_OVERFLOW_MARKERS` matches and no retry budget is spent |
| `empty.jsonl` | zero bytes on both channels, distinguishable from a valid empty answer |

## The one synthesized row

`multi-assistant.jsonl` is **synthesized from the real record**: copilot emits a
single `assistant.message` per turn, and producing a second requires a tool loop
that the committed stub deliberately never drives. Recorded here rather than
faked silently — the rule it pins (last-wins) is real and is what the muse trap
cost this repo before.

## `stdout-error` is not a separate file, on purpose

§6.2's `stdout-error` row — error text on stdout with stderr empty and a nonzero
exit — is the shape of **every** error fixture above, because `session.error` is
copilot's only error channel and it rides stdout. A separate file would be a
duplicate rather than a distinct case.
