# `rate-limit` — no terminal capture exists, and that IS the measurement

§6.2 asks for a 429 fixture pinning `rate_limit_patterns`. qwen produces no
terminal output on that path to capture, because **it never terminates**:
§6.1.1's experiment 3 ran a 429-forever backend for over 180 s and logged 44
completion POSTs with the run still going, on a backoff that ramps to 60 s and
then *resets* rather than exhausting. With `Retry-After: 60` honoured exactly the
shape is the same, just slower (12 requests in the same window). Both runs ended
because the probe's own bound fired, so `stdout` is empty and `stderr` carries
only that kill — an artifact of the harness, not of qwen.

So the honest record is:

* the retry evidence lives in the probe's request logs
  (`plans/harness-expansion/probe-out/qwen-code/logs/`), not in a stdout fixture;
* `internal_retries.retries_rate_limit = True` is declared from those logs, which
  is what makes §7.5 rule 6(b) suppress a duplicate harness-side backoff;
* `internal_retries.max_wall_s = None` is declared for the same reason — this
  path has no bound qwen imposes on itself;
* `--max-wall-time`, derived from the slot timeout, is what actually ends this
  run in production, and *that* shape is captured as
  `wall-time-exceeded.{stdout,stderr}.txt` (exit 55, zero bytes on stdout, a
  structured `FatalBudgetExceededError` envelope on stderr).

A 429 that the *upstream* reports inside a completed response is a different
shape and is covered: it arrives as `[API Error: 429 …]` in the `result` record,
i.e. the same universal API-error envelope `exit-zero-failure.stdout.json` pins,
with the upstream text preserved verbatim so `rate_limit_patterns` still matches.
