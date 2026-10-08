# `ephemeral-only` — not applicable to the shipped output mode

§6.2's `ephemeral-only` row pins that an all-noise run classifies as a **failure**
rather than as an empty-string success. qwen has no such stdout to produce in
`-o json`, which is the mode this adapter ships:

* `-o json` writes **one complete JSON array** terminating in a `result` record,
  or — on a hard abort such as the wall-clock bound — **zero bytes** (that shape
  is captured as `wall-time-exceeded.stdout.txt` and is the `empty` row instead).
  Re-checked on 0.25.0 (2026-10-08): its new `system`/`retry` records ride inside
  that array, ahead of an assistant record and the terminal `result` record
  (`truncated.stdout.json`), so they are noise beside that record, never in its
  place;
* the `stream_event` records that *would* be the filterable noise exist only in
  `-o stream-json`, which is not the shipped mode. `stream-json` was measured and
  rejected for a separate reason: its `result` record **drops the `stats` block**
  that corroborates the API-error classification (still so at 0.25.0).

There is therefore no "records present, no answer among them" stdout to capture.
The parser still tolerates the case defensively — an array with no `result`
record returns the failure branch naming that — and it is covered by a unit test
rather than by a fixture, because no run of the real binary produces it.
