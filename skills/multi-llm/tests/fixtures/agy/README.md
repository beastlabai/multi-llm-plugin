# agy parser fixtures

One row on disk, eleven written reasons, and the eleven all reduce to one
measured fact: **agy's auth resolves before its first RPC**, so no dispatch of
any kind reaches a model surface a stub could answer. The bytes here are raw
`agy` output as the CLI wrote them (agy 1.3.1, `--network none`, no credential,
no server — none was needed). Nothing is synthesized. First captured on 1.1.15
(2026-08-19) and re-captured on 1.3.1 (2026-10-08) in the pinned image built at
`tests/container/cli-versions.lock`'s pins, with today's argv (`--print-timeout
20m`, where the first capture ran `90s`): stdout is byte-identical across the
two releases; stderr gained one closing line.

`manifest.yaml` is the authority on what the row asserts —
`tests/test_provider_fixture_rows.py` grades it. This file says why there is
only one.

## What the one capture settled

`agy.py`'s class docstring claims that on error (exit 1) agy emits
`{"status":"ERROR","response":"","error": …}`. That claim had never been
measured. It is now, and **it is right**:

```
exit 1, stdout 238 bytes:
{"conversation_id":"","status":"ERROR","response":"","error":"authentication failed or timed out","duration_seconds":0,"num_turns":0,"usage":{…}}
```

`parse_output` reads it exactly as documented — `success: False` with
`error = "authentication failed or timed out"` — so the machine-readable failure
channel really is stdout.

What the docstring does **not** say, and what the capture adds:

- agy also writes **939 bytes to stderr** on the same path: an interactive
  Google OAuth consent URL, `Waiting for authentication (timeout 60s)...`,
  `Error: authentication timed out.` and `error: authentication failed or timed
  out` (new at 1.3.1; 897 bytes without it at 1.1.15) — and it **blocks for the
  full 60 s** first.
  This is not a CLI that fails fast when logged out.
- **The useful diagnosis and the classifiable string are on different
  channels.** `classify_failure` returns `auth_failed`, but not from the
  envelope: the envelope's own wording (`authentication failed or timed out`)
  matches nothing in any base table. What matches is stderr's
  `Authentication required.`, via `base.auth_error_patterns`. `parse_output`
  reads only stdout; `classify_failure` reads both. Drop stderr anywhere in the
  chain and a logged-out agy classifies as nothing at all.
- Every **non-print** failure is the mirror image — 0 bytes on stdout,
  everything on stderr: `agy --help` (rc 0, 2901 bytes on stderr at 1.3.1,
  2510 at 1.1.15), an unknown flag (rc 2), `--print-timeout 1170` → Go's
  `missing unit in duration` (rc 2), and `agy models` (rc 1, `Error: Please
  sign in to view available models.`).

The stderr bytes carry a Google OAuth URL. Its `code_challenge` and `state` are
per-run PKCE nonces for a login that never completed, and the `client_id` is the
public installed-app id compiled into the binary — nothing there is a
credential (gitleaks agrees). They are, however, why a re-capture is **not**
byte-identical on stderr. stdout is.

## Why the other eleven rows are permanent, not pending

Re-measured here rather than quoted from `redirect.yaml`, because the point of a
fixture set is a measurement:

1. The same argv with `--model definitely-not-a-model-xyz` produced **stdout
   byte-identical** to `auth-error.stdout.txt` (md5
   `dc30d77eed870733053e445f556afa14`), exit 1, 60.15 s. The model string is
   never validated, because the run never gets past authentication.
2. The real binary run with `CLOUD_CODE_URL`, `GEMINI_API_KEY`, `GOOGLE_API_KEY`
   **and** `GOOGLE_GEMINI_BASE_URL` all pointed at a live `stub_server.py` on
   `127.0.0.1:8099` produced the **same** byte-identical envelope — and the
   stub's request log recorded **zero agy requests**. The only lines in it are
   the probe's own `/__health` polls.

That independently reproduces `redirect.yaml`'s agy row: *"it fails at the
ELIGIBILITY gate on `:loadCodeAssist`, whose response schema is Google-internal,
and its AUTH RESOLVES BEFORE THE FIRST RPC — so the failure happens with zero
requests leaving the process."* There is no environment variable that selects an
API-key path a stub could answer, so `success`, `truncated`, `error-envelope`,
`rate-limit`, `context-overflow` and `invalid-model` have no free producer at any
effort.

The remaining five are shape facts rather than lane facts:

- **`ephemeral-only`** — `--output-format json` emits exactly one object; there
  is no filtered-noise record class for an all-noise run to consist of.
- **`multi-assistant`** — the envelope carries a single `response` string, so two
  assistant messages have no representation in what `parse_output` sees.
- **`exit-zero-failure`** — no agy failure was observed exiting 0 (1 for print
  mode and `models`, 2 for flag errors), so the conditional row has no producer.
- **`stdout-error`** — agy *does* put its diagnosis on stdout on a nonzero exit,
  which is the trap this row names and is pinned by `auth-error`; but agy never
  leaves stderr empty on that path, so the literal byte pattern has no producer.
- **`empty`** — the 238-byte envelope is written on every path that reaches
  `--print`. Zero bytes on both channels would mean killing the process, i.e. the
  `timeout(1)` zero-byte-hang artifact §6 step 3 forbids manufacturing.

## What this set found

1. **The `agy.py` docstring's error-envelope claim is correct** — now verified
   rather than asserted. No adapter change needed.
2. **The docstring is incomplete about the channel split.** It says the error
   envelope carries the diagnosis; it does not say the *classification* depends
   entirely on stderr, nor that agy blocks 60 s waiting for an interactive login
   before producing either. Both are now in the adapter docstring.
3. **The tree disagrees with itself about an invalid `--model`.** `agy.py` says
   it is "SILENTLY ignored" (fallback to the default, status SUCCESS);
   `providers.yaml` says it is "rejected outright" with an ERROR envelope listing
   the valid names (verified 2026-07-29). Offline neither is reachable — the auth
   gate wins — so **this set cannot settle it**, and it is recorded here rather
   than guessed at. Settling it costs one credentialed run.

## Reproducing

One container, no credential, no network, no server; ~60 s per run because agy
waits out its own OAuth timeout.

```
docker run --rm --network none \
  --user 10001:10001 --cap-drop ALL --security-opt no-new-privileges --read-only \
  --tmpfs /work/homes:rw,mode=0700,uid=10001,gid=10001 \
  --tmpfs /work/sample-project:rw,mode=0755,uid=10001,gid=10001 \
  --tmpfs /tmp:rw,mode=1777 \
  -v "$PWD/skills/multi-llm/tests/container:/rig:ro" -v "$OUT:/work/out:rw" \
  -e MLLM_REDIRECT_YAML=/rig/redirect.yaml -e MLLM_REAL_DIR=/opt/mllm/real \
  -e MLLM_HOMES_ROOT=/work/homes -e MLLM_CALL_LOG=/work/out/calls.jsonl \
  --entrypoint /bin/bash mllm-e2e:local -lc '<script>'
```

The script runs `AgyProvider.build_command`'s argv verbatim through the PATH
wrapper `/opt/mllm/bin/agy` — **not** `/opt/mllm/real/agy` — because the wrapper
is what applies the row's `env_extra`. That variable matters: it must be
`AGY_CLI_DISABLE_AUTO_UPDATE=true`, the string, not `=1`; with `=1` agy was
measured spawning a background updater that rewrote the binary 1.1.13 → 1.1.15
inside a single run, and agy releases daily. Do **not** wrap the run in
`timeout`. The whole outer `docker run` is bounded by
`timeout --signal=KILL 300` on the host side instead.
