# `rate-limit` — no terminal capture of qwen's own, and that IS the measurement

§6.2 asks for a 429 fixture pinning `rate_limit_patterns`. qwen produces no
terminal output of its own on that path, because **it never terminates**:
§6.1.1's experiment 3 ran a 429-forever backend for over 180 s and logged 44
completion POSTs with the run still going, on a backoff that ramps to 60 s and
then *resets* rather than exhausting. With `Retry-After: 60` honoured exactly the
shape is the same, just slower (12 requests in the same window). Re-measured on
0.25.0 (2026-10-08): the same 44 and 12 requests in 180 s, still running when
the capture's bound killed it, so this is unchanged.

So the honest record is:

* the retry evidence lives in the request logs, not in a stdout fixture;
* `internal_retries.retries_rate_limit = True` is declared from those logs, which
  is what makes §7.5 rule 6(b) suppress a duplicate harness-side backoff;
* `internal_retries.max_wall_s = None` is declared for the same reason — this
  path has no bound qwen imposes on itself;
* `--max-wall-time`, derived from the slot timeout, is what actually ends this
  run in production, and *that* shape is captured: `wall-time-exceeded.*` (a
  stalled backend) and, since 0.25.0, `rate-limit-wall-time.*` — this very
  429-forever path ended at 100 s with exit 55 and zero bytes on stdout. At
  0.25.0 its stderr names the 429 (`Retrying in 60s (attempt 1/10): [API Error:
  429 rate limit exceeded]`) ahead of the `FatalBudgetExceededError` envelope,
  so it classifies `rate_limited`. That line appears only once qwen's jittered
  backoff reaches its 60 s step, near the 100 s mark in these captures (one 100 s
  run printed it, another did not), so a shorter wall time ends the run with the
  envelope alone.

A 429 that the *upstream* reports inside a completed response is a different
shape and is covered by the universal API-error envelope the other error rows
pin, with the upstream text preserved verbatim so `rate_limit_patterns` still
matches.
