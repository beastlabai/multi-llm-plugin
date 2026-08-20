# cursor-agent parser fixtures

Two rows on disk, ten written reasons, and **the ten are permanent**. Every other
fixture set in this tree was produced by pointing a CLI at
`tests/container/stub_server.py` under `--network none`. cursor-agent cannot be
pointed at it, ever, and the reason is not effort — it is a wire format. Quoting
`tests/container/redirect.yaml`'s own cursor-agent row rather than paraphrasing
it:

> The redirect itself WORKS — `CURSOR_API_ENDPOINT` (default
> `https://api2.cursor.sh`) sent 100% of traffic to the stub and nothing reached
> Cursor. It is unserviceable anyway: after a JSON handshake
> `POST /auth/exchange_user_api_key`, every call is
> `POST /aiserver.v1.<Service>/<Method>` with `content-type: application/proto`
> and `connect-protocol-version: 1` — binary protobuf over Connect-RPC. None of
> the three model-API surfaces are involved, so serving it means reimplementing
> Cursor's protobuf schema, which is not a stub. `bedrock` mode is not an escape:
> its own self-test validates against the backend.

So the only lane that can produce cursor-agent bytes is the **credentialed live
lane**, and every §6.2 failure shape on it needs Cursor's backend to fail a paid
dispatch at a moment of its own choosing. `manifest.yaml` carries that
measurement per row; this file says why the two rows that *do* exist look the way
they do.

`manifest.yaml` is the authority on what each row asserts —
`tests/test_provider_fixture_rows.py` grades it.

| Fixture | Provenance | What it pins |
| --- | --- | --- |
| `success` | **harvested from the live lane**, not from a stub | the nine-key result envelope as Cursor really wrote it: `type`, `subtype`, `is_error`, `duration_ms`, `duration_api_ms`, `result`, `session_id`, `request_id`, `usage` — and `usage` with **no cost field**, which is what `usage_support = "partial"` declares. `parse_output` unwraps `result` to `["ok"]`; `classify_failure` is `None` |
| `exit-zero-failure` | **synthesized**, one field | `exit_zero_failure_markers` firing on `"is_error": true` — and, unintentionally but importantly, that `parse_output` **ignores** the flag |

## Where `success` came from, exactly

`tests/container/runs/20260819T145322Z/live-cursor-agent/live-provider.json`,
the `stdout` key, byte for byte. That record was written by
`tests/container/live_provider.py` inside the image on the credentialed lane:
`--print --trust --mode ask --output-format json --model auto` against
`api2.cursor.sh`, with the developer's own `~/.config/cursor/auth.json` read
host-side and staged into a disposable `HOME`. exit **0**, stderr **0 bytes**,
11.16 s wall, 6341 input / 31 output tokens.

Two properties of that record are what make it usable as a fixture rather than
as an anecdote: it is **untruncated** (`live_provider.py` writes the whole
stream, not a snippet), and it is **redacted at write time** by value
replacement — and the committed bytes contain no `<redacted>` marker, so nothing
in them was actually replaced. The `session_id` and `request_id` are Cursor's
own per-run UUIDs for a session that is long over; they are not credentials.

## The one synthesized row, and the finding it turned up

`exit-zero-failure.stdout.txt` is `success.stdout.txt` with **one field
changed**: `"is_error":false` → `"is_error":true`. Nothing else was touched, and
`"subtype":"success"` is deliberately still sitting next to it.

Synthesizing it is legitimate because the *shape* is documented as measured, not
guessed — `cursor_agent.py`'s `exit_zero_failure_markers` records that
cursor-agent "exited **0** in all six auth states measured, so for this CLI the
exit code carries no verdict whatsoever and only a stream field does", naming
`"is_error"\s*:\s*true` as that field (2026-08-16). What cannot be harvested is
the *vendor-side condition* that sets it: that needs a real failing dispatch
against Cursor's backend, on credits, when the backend decides to fail. Flipping
one documented flag and labelling it is a smaller fiction than inventing a
failure around it.

The row then measured something nobody had asserted:

> **`parse_output` never consults `is_error`.** It is
> `parse_result_envelope`, which branches only on the presence of a `result`
> key — so an envelope flagged as an error whose `result` still holds a
> parseable document comes back `success: True` with that document served as the
> answer. The only part of the adapter that sees the flag is `classify_failure`.

Hence the two assertions on those bytes look contradictory and are both correct:
`expect.success: true` **and** `classify_failure: runner_error`. That division of
labour is real and load-bearing — §7.2 step 6 consults the exit-0 markers last,
after the parser — and it was invisible before this row existed. It is recorded,
not fixed: the fixture pins today's behaviour, so any change to it is a visible
edit to both the adapter and this manifest.

## The row that is *almost* capturable, and why it is not here

`auth-error`. A second failure shape is documented as measured on the same date:
a plain-text `Error: Authentication required. Please run 'agent login' first, …`
printed *instead of* the envelope when auth fails before the session starts. It
is not committed because **the record of it ends in an ellipsis** — byte-exact
bytes for it exist nowhere in this tree, so a fixture would be inventing the
tail rather than flipping one documented field. Re-harvesting it means a live
run with a deliberately broken credential, which is a credit spend this capture
was scoped not to make. The classification it would pin is already covered:
`authentication required` is a member of `base.auth_error_patterns`.

## Reproducing

There is no free reproduction, and that is the finding. The `success` row is
re-harvested by running the live lane
(`tests/test_container_live_provider.py`, `live-cursor-agent`) and copying the
`stdout` key of the new run's `live-provider.json` — a credentialed,
credit-spending run against `api2.cursor.sh`. The `exit-zero-failure` row is
regenerated from whatever `success.stdout.txt` then holds by replacing
`"is_error":false` with `"is_error":true`, once.
