## Short answer

The weakest part of the rate limiter is `src/security.py`: the attempt store is
an unbounded module-level dict, the lockout comparison is off by one, and token
comparison is not constant time.

## Detail

- **Unbounded store.** `record_attempt` only ever appends. Nothing evicts an
  address once seen, so a process that has handled many logins keeps every one
  of them for its lifetime. Evicting outside `WINDOW_SECONDS` on write bounds it
  to the active set.
- **Off-by-one lockout.** `is_locked_out` compares `len(recent) > MAX_ATTEMPTS`,
  so an account is locked only after `MAX_ATTEMPTS + 1` failures. The plan's
  wording calls for `>=`.
- **Timing leak.** `compare_tokens` returns `a == b`, which short-circuits on
  the first differing byte. `hmac.compare_digest` is the drop-in fix.

## What the plan does not say

The plan does not state whether the limiter is per-process or shared. As written
the store is per-process, so the limit is effectively multiplied by the worker
count — worth pinning before this ships.
