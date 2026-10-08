# muse parser fixtures

Every `*.stdout.txt` / `*.stderr.txt` here is a **real capture** of
`muse exec --json`'s streams under the argv `MuseProvider.build_command` emits,
taken against a stub in the containerized §6 probe (Muse Code `1.4.3-R5018.1`,
`--network none`, **no credential** — the PATH wrapper's own fake `META_API_KEY`,
`mllm-stub-<exec_id>`, which is meaningful only on `127.0.0.1:8099`). They are
raw bytes as muse wrote them, and **there is no synthesized row** — including
`multi-assistant`, which every other fixture set in this tree had to fake. The
set was first captured on `0.1.0-R708.1` (2026-08-19) and re-captured on
`1.4.3-R5018.1` (2026-10-08) by the same rig in the pinned image built at
`tests/container/cli-versions.lock`'s pins, with today's argv (which gained
`--disable-sandbox` on 2026-09-11); where a fact below moved between the two,
it says so.

They are graded by `tests/test_provider_fixture_rows.py` through
`manifest.yaml` — the generic §6.2 harness. The manifest is the authority on
what each row asserts; this file says *why* each one is here.

## Five facts that hold across every row

1. **Eighteen requests for one generate call, and they are not retries.** One
   catalog GET on `/muse-code/models`, then seventeen POSTs on `/responses` — of
   which exactly **one** is the real `model.meta.response`. The other sixteen
   are four background reminder observers (`reminder.agent.skill-reminder`,
   `scope-reminder`, `goal-reminder`, `verify-reminder`; 0.1.0 named them
   `reminder.agent.plugin:tbh-reminders:*`), four calls each, each with a
   one-tool `submit_reminder_decision` namespace, versus the main task's
   26-tool `muse` namespace. Every scenario here answers by **fallback** rather
   than by anchored pattern, because all seventeen prompts quote ours. (0.1.0:
   two GETs and fifteen POSTs, a 20-tool namespace.)
2. **No reminder fails in the green run any more.** `success.stdout.txt` and
   `ephemeral-only.stdout.txt` have byte-identical record-type histograms: 44
   records, fifteen payload types, same counts, and no `task.lifecycle.failed`.
   At 0.1.0 both held exactly one, reading *"model did not reach a terminal
   state within 2 step(s)"* — a reminder that always failed against a stub,
   green run included — which is why muse declares no
   `exit_zero_failure_markers` and must never grow one keyed on a task failure:
   a build whose observer fails would fail green runs.
3. **The session prefix is partial**, exactly as `redirect.yaml` records: the
   POSTs arrive at `/s/<exec_id>/v1/responses`, the catalog GET at
   `/muse-code/models` with the prefix dropped, because muse rebuilds that URL
   from the origin.
4. **stderr is never empty, and at 1.4.3 it carries the cause too.** 123 bytes
   on a run that reaches `completed` (`muse: workspace root: … (cwd default)`
   plus `muse: Agent delegation: auto unavailable: workspace is untrusted.`),
   203–642 on a failed one — the same two lines plus `run ended with Failed:
   <reason>`, the `run.terminal.failed` record's `reason` verbatim. At 0.1.0
   that line was the content-free `run ended with Failed` and the cause was on
   stdout alone, which is what `describe_failure` exists to recover; it still
   reads stdout.
5. **Exit 1 on every real failure.** All nine failure rows exit 1.

| Fixture | What it pins |
| --- | --- |
| `success` | the baseline and the envelope's floor: 18 requests, no retries, 5.5 s. One `run.output.delta` carries the whole answer, so `tail` and `full` coincide — the single-turn case |
| `multi-assistant` | **the muse trap, captured rather than synthesized** — see below |
| `ephemeral-only` | an all-noise run is a **failure**, not an empty-string success. Same 44 records as `success`, `"text":""` — muse exits **0** and reports `completed`, so only the parser says this produced nothing |
| `auth-error` | 401 → **terminal, one attempt**. muse swallows the vendor text; once a **defect** (classified `None`), now `auth_failed` — see below |
| `invalid-model` | 404 **naming the model** — the best of the six failure shapes. At 0.1.0 the vendor sentence came through verbatim and classified `server_error`; at 1.4.3 muse glosses it (``model `…` does not exist or you lack access``) and it classifies `invalid_model` — see below |
| `context-overflow` | 400 context-length, **passed through**: `maximum context length` survives, `CONTEXT_OVERFLOW_MARKERS` matches, the verdict is terminal and correct. The positive control for the two rows above |
| `error-envelope` | a 200 carrying a provider error object is **never read**: muse requires `text/event-stream` and rejects it as `non-SSE response`. An error object can never be served as an answer here — and can never be classified either |
| `stdout-error` | §6 step 3's opencode trap as 0.1.0 had it, plus the §6.1.1 envelope in one capture: exit 1, 30 POSTs, **251 s**, `opening meta model stream attempt N/10` ten times, the whole diagnosis on stdout. At 0.1.0 stderr said only `run ended with Failed`; at 1.4.3 that line ends with the reason |
| `rate-limit` | 429-forever → **the same envelope as the 5xx**: 10 attempts, 30 POSTs, 249 s (to two decimals at 0.1.0). muse's ladder does not distinguish a rate limit from a server error, so an executor-level re-dispatch repeats a chain that already ran for four minutes |
| `truncated` | the **default** `--trunc-bytes 60` cut of the Responses **SSE** stream: four rounds of the ten-attempt ladder, 120 POSTs, **985 s**, ended by a **12-minute wall-clock ceiling** muse enforces itself. The richest diagnosis muse produces — and it classifies as `None`. See "Three cuts, three behaviours" |
| `truncated-after-deltas` | **not one of the twelve** — the same fault cut one frame later (inside `response.completed`): terminal in **one** attempt, 5.3 s. Partial text recovered, **not** served as the answer; since C8 it fails at `parse_output`'s first branch with muse's own reason |
| `truncated-json-answer` | **not one of the twelve** — the one shape in this set where the parser gets it wrong. A cut one frame earlier again, with a complete JSON array already delivered, so `parse_output` returns **success** for a run that exited 1 after ten failed attempts |

## `multi-assistant`: the row this whole set exists for

§6.2 names this row *"the muse trap"*, and both previous attempts at it in this
repo are labelled fakes (copilot's `multi-assistant.jsonl`, kimi's
`multi-assistant.SYNTHESIZED.md`). The reason is mechanical: producing a second
assistant message needs a `tool.result` record between two turns, and
`stub_server.py` deliberately never drives a tool loop — *"Making the stub drive
a tool loop would be simulating an agent, which is the host lane's job."*

So the capture used a **throwaway probe-local Responses-API server** instead
(`plans/harness-expansion/probe-out/muse/probestub.py`, uncommitted, in the
probe's own directory — the committed servers were not touched). It answered the
main task's first request with a message item carrying a **draft** array plus a
`function_call` to muse's own `read_file`; muse really executed that tool against
a seeded `NOTES.md`; and the second request got the real array. Every byte on
disk is muse's own JSONL.

What it pins, exactly — and this is the documented rule at `muse.py:206-232`:

```
run.output.delta   "Draft: [{"id": "DRAFT", "title": "FIRST ANSWER (must NOT be selected)", …}]"
tool.result        read_file → NOTES.md            ← the only message boundary muse publishes
run.output.delta   "[{"id": "S001", "title": "Bound the retry ladder", …}]"
run.terminal.completed  text = the two glued together, with NO separator
```

Feeding that terminal `text` to `extract_json_from_text` now returns the
**answer** (verified: `id: S001`), because the extractor takes the last document
at every level; it used to return the draft. Taking the deltas after the last
`tool.result` returns the answer too, so the two routes agree **on this
capture** and the tail rule is no longer what separates them here. The rule
still ships, and what it pins is narrower: a `tool.result` boundary is
authoritative rather than inferred from document order — text from an earlier
turn is never the answer, even when it is the last thing before the boundary and
even when it is the only thing that parses. This row is still the only capture
in the repo that exercises that boundary at all.

## Three cuts, three behaviours — and the ceiling underneath them

**How far the SSE stream got before the socket closed decides everything**: how
many times muse retries, how long it takes, and whether the parser gets it
right. All three cut points are committed, because no one of them tells the
story. The offsets come from the surface's own payload — the Responses stream
for a two-character answer is 2354 bytes with `response.completed` starting at
1911; for the JSON-array answer it is 2759 bytes with `response.output_item.done`
starting at 1873.

| Row | Cut lands in | Attempts | Wall | Outcome |
| --- | --- | --- | --- | --- |
| `truncated` | `response.created` (60 B, the default) | 4 × 10 | **985 s** | terminal, no text, the 12-minute-ceiling diagnosis — and `classify_failure` is `None` |
| `truncated-json-answer` | `response.output_item.done` (2100 B) | 10 | 70 s | terminal, and the salvaged text used to be served as a **success** — refused since C8 (defect 4 below) |
| `truncated-after-deltas` | `response.completed` (2000 B) | 1 | 5.3 s | terminal, partial text recovered and correctly refused |

The canonical row's own words are the most useful thing in this directory:

> model call chain exceeded the **12m wall-clock ceiling** measured from its
> first failed attempt: 4 failed attempts (40 provider requests) over 16m19s
> since the first counted provider request, all
> `[transport_eof_before_terminal_event]`. dominant failure
> `[transport_eof_before_terminal_event]` — next step: check network
> connectivity (VPN/proxy) and retry; if it persists, re-authenticate with the
> provider — a very large context can also repeatedly break streams, so consider
> compacting the session

(That is 1.4.3's wording; 0.1.0 wrote "over 12m16s" for the same four rounds,
timed from a later point.) So the ladder is bounded, but by **wall clock, not
by attempts**: ten provider
attempts per round, rounds repeated until twelve minutes have passed since the
first failure. Two operational consequences follow, and both are in the
manifest: 985 s is past muse's own `default_timeout` of **600** and far past
`redirect.yaml`'s `row_timeout_s: 300`, so **in production this shape is a
harness timeout and that diagnosis is never seen**; and it classifies as
`None`, because nothing in any base table matches
`transport_eof_before_terminal_event` or the "check network connectivity"
sentence after it.

`failserver --mode empty` (accept, write zero bytes, close) is the same event —
identical request counts through the 900 s mark, where an earlier probe's outer
bound cut it (measured at 0.1.0 and not re-run). That is aider's *"a truncated body and a zero-byte response are
the same event at this CLI's output"* result reached from the other side, which
is why it is not a fourth file.

## What this set found

Five measured defects, recorded here rather than silently fixed — the fixtures
pinned the behaviour, so each fix was a visible edit to both the adapter and this
manifest. The fourth is the serious one. **Four have since landed** and each
entry below says which and how the same bytes now read; the fifth is still open
and still pinned.

1. **A rejected muse credential classifies as nothing.** `MuseProvider`
   deliberately adds `api error 401` and `api error 403` to
   `auth_error_patterns`, and **neither fires**: on a real 401 muse swallows the
   vendor text completely — the strings `401`, `API error` and `Incorrect API key
   provided` appear *nowhere* in the 17 KB capture (nor in 1.4.3's) — and writes its own gloss,
   *"your API key from META_API_KEY was rejected — update or unset it"*. That
   matched none of the twelve patterns, so `classify_failure` returned `None`,
   the failure was transport-shaped, and the executor spent its whole bounded
   retry budget on a condition §7.2 makes terminal. **Fixed (C3), from these
   bytes and no new ones**: `auth_error_patterns` gained `api key from` and
   `update or unset it` — two independent anchors on that one sentence, so a
   reworded half is still caught by the other — and the `auth-error` row's cell
   now reads `auth_failed`. `was rejected` on its own was deliberately not
   added: `vendor_text` drops muse's prompt-echoing payloads but keeps every
   `run.output.delta`, so a pattern a model's own prose can write would fail
   whole batches. Neither phrase occurs in any other row of this set.
2. **Every muse vendor error outranks `invalid_model`.** muse prefixes what it
   does pass through with `API error <status>:`, and `\bapi error\b` is a member
   of `base.SERVER_ERROR_MARKERS`, which `classify_streams` consults **before**
   `INVALID_MODEL_MARKERS`. So the 404 — which is otherwise the best failure
   shape in the set, keeping the vendor sentence and *naming the model* — landed
   on `server_error`, a retryable rung, when the fix is a config edit. The 402
   and 429 rungs are unaffected, because rate limits are checked first. **Fixed
   (C2) in the base ladder rather than here**: `INVALID_MODEL_MARKERS` is now
   consulted before `SERVER_ERROR_MARKERS` — a model-named 404 is strictly more
   specific than "api error" — and the `invalid-model` row's cell reads
   `invalid_model`. At 1.4.3 the point is moot for this row: muse glosses the
   404 too (``model `…` does not exist or you lack access``), no `API error`
   prefix survives, and the base `\bmodel .{0,40}does not exist` marker
   classifies it on its own.
3. **A 5xx or a 429 costs four minutes before the harness sees a byte.** Ten
   internal provider attempts, 30 POSTs, 248 s, measured identically for both
   at 0.1.0 (250.7 s and 249.0 s at 1.4.3, the same ladder).
   muse narrates its own ladder, so the envelope is readable straight out of the
   capture's `task.lifecycle.status` messages: `opening meta model stream
   attempt N/10` with `retrying meta model stream in 1000ms, 2000ms, 4000ms,
   8000ms, 16000ms, 32000ms, 60000ms, 60000ms, 60000ms` — exponential, **capped
   at 60 s**, ten attempts, 243 s of sleeping. That is muse's `internal_retries`
   envelope and it means `retries_rate_limit` is true for this CLI:
   re-dispatching a rate-limited muse slot at the executor level doubles a wait
   that already ran for four minutes (§7.6(b) rule 6b). **Declared (C12)**:
   `InternalRetryEnvelope(max_requests=40, max_wall_s=736.0,
   retries_rate_limit=True)` — 40 because the canonical `truncated` row's own
   sentence is *"4 failed attempts (40 provider requests)"*, and 736 s because
   that is muse's own 12-minute ceiling, overshot to 12m16s (0.1.0's figure;
   1.4.3 reports the same four rounds as 16m19s since the first counted
   provider request, and the declaration was left as it is). Read the budget
   consequence the right way round: declaring `retries_rate_limit` **removes**
   the harness's own rate-limit backoff for muse (rule 6(b),
   `STOP_VENDOR_ALREADY_RETRIED`). It is a smaller retry budget, not a larger
   one — the multiplication is what the rule exists to stop.
4. **A failed run's salvaged fragment was served as the answer whenever it
   happened to parse.** `truncated-json-answer.stdout.txt` is a run that exited
   **1** after ten attempts, with `run.terminal.failed` reading *"model failed
   after 10 attempts: stream ended before terminal event"* — and `parse_output`
   returned `success: True` with the array. One guard caused it: the first branch
   failed a non-completed run only `and not full`, so with any text present the
   function fell through to `_direct_json(tail)` and returned before the
   terminal state was consulted again. Blast radius, checked in both callers
   rather than assumed: **Lane A is safe by accident** — `llm_client`
   short-circuits on `returncode != 0` and never calls `parse_output` — but the
   **subagent executor was not**, because it documents *"exit code is evidence,
   never the verdict"* and accepts a schema-valid result from a non-zero exit
   with only an annotation. **Fixed (C8), in the adapter and nowhere else**: the
   `and not full` conjunct is gone, so terminal state comes before salvaged text
   *in one direction only* — a recorded non-`completed` terminal fails the run
   whatever parsed, while a **missing** terminal record still parses, because
   the absence of a verdict is not a verdict. The row's `success` cell now reads
   `false` with muse's own reason in `error_contains`; `raw` stays the whole
   stream, so Lane B lands it on rung 5 (`unparseable_result`) rather than rung
   4. `utils/subagent_executor.py` is byte-identical: giving *it* the exit-code
   check `llm_client` has by accident was rejected, not deferred — that would
   change how every adapter's exit code is read to fix one adapter, and the
   evidence condemning this run is the terminal record only this adapter parses.
   `tests/test_muse_terminal_state.py` drives both directions through the
   executor's own parse path.
5. **A sixteen-minute transport failure classifies as nothing.** The default
   truncation ends with `transport_eof_before_terminal_event` and an explicit
   *"check network connectivity (VPN/proxy) and retry"* — and
   `classify_failure` returns `None`, because no member of
   `CONNECTION_ERROR_MARKERS`, `TIMEOUT_MARKERS` or any other base table matches
   either string. So the ladder treats it as generic transport noise and spends
   its whole budget on top of the 985 s muse already spent. In practice the
   600 s `default_timeout` fires first and the run is reported as a timeout
   instead, which is the right *category* by accident and loses the diagnosis
   entirely.

## Two rows that are "not applicable, because …"

Both carry their measurement in `manifest.yaml` rather than being absent (§6.3).

- **`exit-zero-failure`** — muse exits **1** on every failure mode the rig can
  produce (401, 404, 400 context-length, a 200 error object, 500-forever,
  429-forever, and both terminating truncations), at 0.1.0 and at 1.4.3, so the "still exiting 0" half has no producer. The one exit-0 run with
  no answer is `ephemeral-only`, and it is not a failure of the CLI: muse reached
  `run.terminal.completed` legitimately and the model returned `""`. The
  tempting marker (`task.lifecycle.failed`) was **disqualified by measurement**
  at 0.1.0 — `success.stdout.txt` contained one too — and is not needed at
  1.4.3, where every failure exits 1.
- **`empty`** — muse writes its workspace banner (123 bytes at 1.4.3, 57 at 0.1.0) to stderr before the
  first HTTP request and `runtime.command.accepted` to stdout immediately after,
  so the smallest capture here is **16,215 bytes**. Zero bytes on both channels
  would mean killing the process, which is the `timeout(1)` zero-byte-hang
  artifact §6 step 3 forbids manufacturing.

## Reproducing

One container per batch, no credential, no network. The driver is a throwaway
Python script in the probe's own directory — deliberately not committed, because
the two rig servers it drives are.

```
docker run --rm --network none \
  --user 10001:10001 --cap-drop ALL --security-opt no-new-privileges --read-only \
  --tmpfs /work/homes:rw,mode=0700,uid=10001,gid=10001 \
  --tmpfs /work/sample-project:rw,mode=0755,uid=10001,gid=10001 \
  --tmpfs /tmp:rw,mode=1777 \
  -v "$PWD/skills/multi-llm/tests/container:/rig:ro" -v "$OUT:/work/out:rw" \
  -e MLLM_STUB_PORT=8099 -e MLLM_REDIRECT_YAML=/rig/redirect.yaml \
  -e MLLM_REAL_DIR=/opt/mllm/real -e MLLM_HOMES_ROOT=/work/homes \
  -e MLLM_CALL_LOG=/work/out/calls.jsonl \
  --entrypoint /usr/bin/python3 <pinned image> /work/out/capture.py '<rows>'
```

Per row the driver: copies `/opt/mllm/rig/home_seed/muse/.` into
`/work/homes/muse/`, starts one server on `127.0.0.1:8099` — `python3
/rig/stub_server.py --port 8099 --log … --scenario …` for `success` and
`ephemeral-only` (`multi-assistant` uses the probe-local server described
above), `python3 /rig/failserver.py --port 8099 --mode
{401,404,ctx,errbody,500,429,trunc} --log …` for the rest — polls `/__health`
with **urllib** (curl is not in the image), then runs
`MuseProvider.build_command`'s argv verbatim through **`/opt/mllm/bin/muse`**,
capturing stdout and stderr into separate files with `$?` and the wall time.

Three things are not optional:

- **Dispatch through the wrapper, not `/opt/mllm/real/muse`.** The wrapper is
  what applies `redirect.yaml`'s `argv_inject` — muse's base URL is **argv-only**
  (`--base-url` after `exec`; six config keys were proved negative) and
  `MuseProvider.build_command` emits none — and its `env_extra`,
  `MUSE_NO_AUTO_UPDATE=1`, without which the `muse` on PATH (a bash *launcher*,
  not the binary) auto-updates from `api.meta.ai` synchronously and fatally.
- **The catalog envelope must be `data`, not `models`.** muse decodes it with a
  typed deserializer and fails the whole run with ``missing field `data` `` on
  the wrong shape. `stub_server.catalog("muse")` is already right and
  `failserver.py` delegates to it; do not "simplify" either.
- **Never wrap the child in `timeout(1)`** inside the container. The outer bound
  is a host-side `timeout --signal=KILL`, and it has to be generous: the 5xx and
  429 rows take **~250 s each** and the default `truncated` row takes **985 s**,
  because muse's ceiling on that shape is twelve minutes of wall clock measured
  from its first failure. An earlier probe bounded at 900 s recorded it as a
  kill; the committed capture is the one that was allowed to finish.

`truncated` needs no extra flags at all — it is `--mode trunc` at the default
`--trunc-bytes 60`, and it takes **985 s**. The two extra truncation rows do:
`--trunc-bytes 2000` for `truncated-after-deltas`, and `--trunc-bytes 2100` plus
a `--trunc-answer` holding a complete JSON array for `truncated-json-answer`.
Recompute those offsets from `failserver.success_frames('responses', …)` if the
renderer ever changes — the table above says which frame each cut is supposed to
land in, and landing in a different one changes the behaviour being captured.
