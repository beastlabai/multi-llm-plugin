# Wave-Based Parallel Spawning

When there are more than 4 batches, spawn them in waves to control context growth.

## Strategy Selection

- **≤ 4 batches (Strategy A)**: Spawn all batches as parallel Task agents in a single message.
- **> 4 batches (Strategy B)**: Use wave-based spawning (below).

## Wave-Based Spawning (Strategy B)

1. Split batches into waves of up to 4 batches each, except the last wave may include up to 6 batches (absorbing any small remainder to avoid an unnecessary extra round — e.g., 5 batches = one wave of 5, 9 batches = wave of 4 + wave of 5, 8 batches = two waves of 4)
2. For each wave:
   - Spawn all batches in the wave as parallel Task agents (single message with multiple Task calls)
   - Use the same prompt template as single-batch processing, with each batch's `group_indices` and `output_path`
   - Wait for all agents in the wave to complete
3. Proceed to the next wave
4. After ALL waves complete, continue to the next step (reaggregation, etc.)

## Error Handling

- Per-subagent timeout: 60 seconds. If a subagent times out, mark that batch as `validation_failed`.
- If a subagent returns malformed output or fails validation, retry it once (max 1 retry per batch). If still failing, mark it as `validation_failed` and move on.
- If any batch in a wave fails (after retry), log the failed batch index and continue to the next wave -- do NOT abort the workflow.
- After ALL waves complete, report which batches failed (e.g., "Completed {success}/{total} batches. Failed: batch {indices}.").
- Proceed to the next step regardless of partial failures -- `merge_batched_validation_results` already handles missing batch files gracefully, so partial results are merged automatically.

## Important Notes

- Do NOT spawn batches as background tasks. Spawn them as regular parallel Task calls so results return inline.
- Context growth per wave is approximately 30-50 lines (4 Task calls with parameters plus responses), so for 5 waves (~20 batches) expect ~150-250 lines of context growth — still manageable within typical context budgets.

## ETA Reporting

When the calling instruction file wraps the wave loop with `metrics.py start ... finish`, each per-batch `metrics.py record --batch-index {N} --total-batches {total}` prints an `[ETA]` line to stderr. In wave-parallel runs, summing per-subagent durations would overstate wall-clock time by a factor of ~wave-width, so the ETA helper reports both signals (`per-item` and `wall-clock`) and the predicted-remaining segment uses `min(per_item, wall_clock)` — wall-clock dominates here. Pass the **cumulative** `--batch-index` (1..N across all completed waves), not the within-wave index.

## Design Rationale

**Why wave size 4 (with flexible last wave)?** The base wave size of 4 balances parallelism against context growth and typical API concurrency limits. The last wave absorbs remaining batches (up to 6 total) to avoid unnecessary serialization at the boundary — the extra context from 1-2 additional parallel agents is negligible compared to the cost of an entire extra sequential round.

**Trade-off: wave-level synchronization.** Wave-based spawning waits for all subagents in a wave to complete before starting the next wave. If one subagent is slow, the other completed results sit idle until the wave finishes. This is simpler and more predictable than the coordinator approach, but does sacrifice some throughput.
