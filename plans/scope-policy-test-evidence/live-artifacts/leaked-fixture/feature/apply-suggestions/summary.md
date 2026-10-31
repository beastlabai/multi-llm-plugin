# Applied Suggestions Report

**Plan**: /tmp/scope-policy-live-20261031/leaked-fixture/feature.md
**Applied to**: /tmp/scope-policy-live-20261031/leaked-fixture/feature.md
**Generated**: 2026-10-31 17:42:27

## Summary
- Batches processed: 1/1
- Suggestions applied: 0
- Suggestions declined by their batch (not applied): 1
- Suggestions skipped: 0
- Decided by the judge agent: 0
- Batching efficiency: 0 subagent call(s) saved (0.0%)

## Applied Changes

### apply-batch-0 — ok (no-op)
**Suggestions**: 1  **Duration**: 16.6s  **Runner**: claude-code:haiku
**Not applied**: 1 declined by the batch (listed under "Declined by their batch (not applied)" below)
**Notes**: Suggestion 1 (1de5f2451724bb54, CSV header row never specified): declined without editing. Scope validation allows naming the header, but this plan has no column set to name, and the only one in the repo is from a different plan. Needs the user to state the report's columns, order and quoting rule, after which the ## Output section can be updated. No other changes were made, and no processes or scratch files were used.
**Reason**: NO-OP: the batch declined every item it held (suggestion 1 (`1de5f2451724bb54`) — leaked-fixture/feature.md defines no report fields, so the header columns, their order and quoting need a plan decision; the only column list in the repo (date,category,amount) belongs to the separate expense-export plans in preserve.md and expand.md and does not apply to this report) and the pre-batch snapshot comparison confirms no byte changes in leaked-fixture/feature.md. Recorded; the loop continues.

## Declined by their batch (not applied)

- **CSV header row is never specified** (`1de5f2451724bb54`, apply-batch-0) — leaked-fixture/feature.md defines no report fields, so the header columns, their order and quoting need a plan decision; the only column list in the repo (date,category,amount) belongs to the separate expense-export plans in preserve.md and expand.md and does not apply to this report.

## Deferred by scope policy (1, not applied)

- **Also export the report as PDF** (`0a7d6dd3784f86f9`) `Scope extension` — Deferred: scope preservation requested: the proposed remedy extends the agreed scope (The plan's only output requirement is CSV export, so adding a PDF exporter extends the agreed scope.)
