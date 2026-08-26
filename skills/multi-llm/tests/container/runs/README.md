# `runs/` — raw evidence bundles (ephemeral, gitignored)

Each container row binds `/work/out` here, at `runs/<utc>/<row-id>/`, and leaves:

- `stub.jsonl` — every request the stub server received, including the ones it
  could not answer (`unhandled: true`) and the ones no pattern matched
  (`unmatched: true`);
- `calls.jsonl` — the PATH wrapper's call log: one redacted argv per real CLI
  invocation, plus the scrub receipt (child environment variable *names*, never
  values, and the config root);
- `receipts/<row-id>/NNN.receipt.json` — one per dispatcher invocation, written by
  the `uv` wrapper: the parsed `MULTI_LLM_RESULT` payload, the run-id-correlated
  worker record, exit code and wall clock;
- `artifacts/` — the plan's output directory, copied out by the entrypoint's
  `finally` step so a failed row still leaves its phase dirs, `state.json` and
  `.status.json` files behind.

## This directory is not provenance

**Nothing in the repo may cite a path under `runs/` as evidence** — not
`hosts/capabilities.yaml`, not a commit message, not a README. It is gitignored,
`make e2e-clean` prunes it, and a fresh clone does not have it. A dangling pointer
into a deleted directory is exactly the failure mode the split exists to prevent,
and a test asserts no tracked file contains a `runs/` path outside the two places
that describe the directory itself.

The durable half is the sibling **`records/`**: a sanitized, schema-pinned,
**tracked** summary the host-side driver derives from these bundles — host id, CLI
version, image digest, row outcomes, the credential census, cost actuals, and the
sha256 of every file here. That is what a capability promotion cites, committed in
the same diff as the `capabilities.yaml` edit it justifies.

If you still have a bundle, `make e2e-verify-record REC=records/<utc>-<host>.json`
re-hashes it and says whether it is the one the promotion cited. If you deleted it,
the record still stands on its own and the tool reports the bundle as *missing*
rather than as *mismatched* — which is the honest answer, since deleting this
directory is a supported thing to do.

## Capped, and safe to delete

The lane prunes itself: `harness.prune_runs` runs as each bundle directory is
created and keeps the newest **8**, protecting the bundle being written.
`MLLM_E2E_RUNS_KEEP` overrides the number and `0` turns the sweep off.
`make e2e-prune KEEP=n` is the same sweep on demand, for a checkout that predates
it. Both are shape-aware: only `<utc>[-<suffix>]/` directories are candidates, so
this README, `lock-update/` and `surface/` stay. `make e2e-clean` is still the
blunt instrument — `rm -rf runs/*/`, which takes those with it — and both leave
`records/` untouched.

**Why a cap and not just an ignore rule.** `runs/` is gitignored, and that was
treated as the whole answer until it wasn't: nothing that copies a *working tree*
consults `.gitignore`. `codex plugin add` against a local-path marketplace and a
local-path plugin install both take the directory as it stands, so an uncapped
`runs/` is a distribution problem before it is a disk one — it reached **1.2 GB
here, in a repo with 18 MB of tracked files**, and the advertised install would
have copied every byte of it. The build context is a separate defence and already
holds (`.dockerignore` re-excludes `runs/**` from inside an allowlisted tree);
the cap is what protects the routes that have no context file to read.

A pruned bundle costs nothing a record needed: `make e2e-verify-record` reports
**MISSING** and exits 0, saying the record stands on its own, rather than reading
the absence as a mismatch.

One consequence worth knowing: **`test_container_argv.py` grades what is in here**.
It grades the bundles *its own invocation* produced — the run roots
`harness.MATRIX` recorded, filtered to the rows whose `row-result.json` says
`lane: provider`, because a discovery-only or host-only bundle dispatches no argv
this table describes and would starve every entry. Reading a bundle back from disk
survives only as the fallback for running that module alone, and only when the
bundle is newer than every input the comparison depends on (`expected_argv.yaml`,
`normalize_argv.py`, the provider adapters, the subagent executor and the write
scheduler); otherwise it skips, naming the command to re-run, rather than passing
on stale evidence. So a clean disarms it until the next lane run, and editing a
`build_command()` — or the table itself — cannot be graded against yesterday's
argv.
