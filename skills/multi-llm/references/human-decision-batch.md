# Human Decision Batch Mode

When the orchestrator output includes `human_review_config`, use batch mode for efficient review of `needs-human-decision` items.

## Batch Mode Guard (Required)

Before prompting for ANY needs-human-decision item, check:

0. Read `human_review_config.decision_mode` from orchestrator output.
   If it is `"claude_auto_decide"` (set by the `--claude-decide` flag):
   - Do NOT prompt the user at all — skip the batch summary AND any
     per-item AskUserQuestion.
   - Go directly to the **"Let Claude Decide" Mode** section below and
     evaluate every item autonomously.
   - This takes precedence over `batch_enabled` and `mode`.
1. Otherwise, read `human_review_config.batch_enabled` from orchestrator output
2. If `batch_enabled == true`:
   - Do NOT use individual AskUserQuestion for each item
   - Present the batch summary prompt ONCE
   - Record all decisions with the same batch context
3. If `batch_enabled == false` OR `human_review_config` is missing:
   - Fall back to individual AskUserQuestion per item

## Batch Approval Flow

When there are multiple `needs-human-decision` items, present them in batches organized by importance level.

Use AskUserQuestion:
- Question: "There are {N} items requiring human review ({X} HIGH, {Y} MEDIUM, {Z} LOW). How would you like to proceed?"
- Header: "Batch review"
- Options:
  - label: "Review each individually", description: "Traditional one-by-one flow"
  - label: "Let Claude decide", description: "Claude evaluates each item using its judgment"
  - label: "Approve all LOW, review others", description: "Auto-approve LOW importance, ask about rest"
  - label: "Approve all MEDIUM and LOW", description: "Auto-approve MEDIUM/LOW, ask about HIGH only"
  (Note: 4 option max for AskUserQuestion — "Approve all" and "Skip all" available via Other)

## "Let Claude Decide" Mode

Trigger this mode when **any** of the following hold:
- the user selects "Let Claude decide" in the batch-approval prompt — scope: **all** `needs-human-decision` items, **or**
- the user selects "Let Claude decide" for a single item in an individual / legacy per-item prompt — scope: **just that one item**, **or**
- `human_review_config.decision_mode == "claude_auto_decide"` (the user passed
  `--claude-decide` on the command line, so no prompt is shown) — scope: **all** `needs-human-decision` items.

The evaluation is identical in every case; only the *set of items* it runs on differs. The per-item trigger exists so a user who is unsure about one specific item can offload just that decision to Claude — exactly as `--claude-decide` would have handled it — without touching the other items. For each item in scope, evaluate it autonomously and choose one of **three** outcomes per item:

- **Approve (as-is)** — the suggestion is sound; apply it unchanged.
- **Salvage (partial)** — only *part* of the suggestion is valid/worthwhile; rewrite it down to that defensible core and apply the rewritten version. **This is the default whenever an item is partially valid — never skip a whole suggestion just because part of it is wrong.**
- **Skip** — there is nothing worthwhile to keep; drop the item entirely.

### Delegate the judgment to a subagent

Always make the Approve / Salvage / Skip judgment in a **subagent**, never inline in the main conversation — the reasoning is token-heavy and would crowd the main agent's context. The main agent stays responsible for *applying* and *recording* the decisions; the subagent only *judges* and returns its verdict.

- **Per-item trigger (one item):** spawn ONE Task subagent for that single item.
- **Batch trigger (all items), ≤4 items:** spawn ONE Task subagent that judges all of them.
- **Batch trigger (all items), >4 items:** spawn ONE coordinator subagent (`model: haiku`) that fans out per-batch judging subagents internally and aggregates their replies (same pattern as validation batching), to keep the main context clean.

Use the **full model** for the judging subagent(s) — this is complex reasoning; never downgrade the judge to haiku. Only a pure-orchestration coordinator may use haiku.

Give each judging subagent:
- the item(s) details: `title`, `description`, `importance`, `validation_reason`, `confidence`, and `models`, plus the stable `group_id` (= `group_hash`) of each item; and
- an instruction to apply the **Evaluation Criteria** and **Deciding Approve vs. Salvage vs. Skip** rules below (point it at the "Let Claude Decide" Mode section of this reference, `references/human-decision-batch.md`).

Instruct the subagent to respond with **ONLY** a compact JSON array (no prose, no markdown fences), one object per item:

```json
[
  {
    "group_id": "<group_id>",
    "decision": "approve" | "salvage" | "skip",
    "reason": "<one brief sentence>",
    "salvaged_description": "<rewritten description — only when decision == salvage; else null>",
    "dropped": "<what was intentionally left out — only when decision == salvage; else null>"
  }
]
```

### Evaluation Criteria

Read each item's details (title, description, importance, validation_reason, confidence) and decide based on:

1. **Clear improvement vs. subjective preference**: Approve items that address genuine bugs, security issues, or correctness problems. Be cautious with purely stylistic or preference-based items.
2. **Validation reason signals**: If the validation reason indicates genuine ambiguity ("could go either way", "trade-off", "matter of preference"), lean toward skipping. If it indicates implementation uncertainty ("not sure about scope", "might affect other files"), lean toward approving.
3. **Importance as risk signal**: Be more conservative (lean skip) with HIGH importance items — they have bigger impact if wrong. Be more permissive (lean approve) with LOW importance items.
4. **Security items lean approve**: Security-type items (XSS, injection, auth bypass, data exposure) should lean toward approval.
5. **Deletion caution**: For items that involve removing code/content, require stronger confidence before approving. Deletions are harder to reverse.
6. **Model consensus**: If the `models` array shows multiple models flagged this as `needs-human-decision`, be more cautious. If only one model flagged it while others said `valid`, lean toward approving.

### Deciding Approve vs. Salvage vs. Skip

Apply the criteria above to each **distinct claim inside the suggestion**, not just the suggestion as a whole. Many `needs-human-decision` items are a mix — one solid point bundled with an overreach, a speculative extra, or an out-of-scope aside. Salvaging the worthwhile part is the **default** for such items.

1. **Identify the worthwhile core**: Which specific part(s) address a genuine bug, correctness, security, or clarity issue and survive the criteria above?
2. **Whole thing is sound** → **Approve** as-is.
3. **Part is worthwhile, the rest is questionable / speculative / out-of-scope / low-confidence** → **Salvage**: rewrite the description to keep only the worthwhile core and drop the rest. Salvage only ever **narrows** a suggestion — never add anything that was not in the original. The existing criteria still govern the kept part (e.g. deletion caution, security lean-approve).
4. **Nothing survives** — the item is purely subjective, speculative, redundant, or wrong → **Skip**. Salvage only when a concrete, defensible improvement remains; do **not** manufacture value or apply a watered-down change just to avoid skipping. If there is nothing worthwhile, skip it entirely.

A salvaged item is applied like an approval, except the **rewritten** description is what gets applied — never the original.

### Process

1. Spawn the judging subagent(s) as described in **Delegate the judgment to a subagent** for the item(s) in scope (all `needs-human-decision` items for a batch/`--claude-decide` trigger; just the one selected item for a per-item trigger), and collect their Approve / Salvage / Skip verdicts.
2. Present an informational summary (not a prompt — do not ask for confirmation):

```
Claude's decisions for {N} needs-human-decision items:

Approved ({A}):
  - [Title] ({importance}) — {brief reason}

Salvaged ({V}):
  - [Title] ({importance}) — kept: {what was kept}; dropped: {what was dropped}

Skipped ({S}):
  - [Title] ({importance}) — {brief reason nothing was worth keeping}
```

3. Proceed directly:
   - **Approved** items → apply the original suggestion.
   - **Salvaged** items → apply the **rewritten** description (substitute it for the original when building the apply prompt / batch).
   - **Skipped** items → take no action.

### State Recording

Record each decision using the existing `record_human_decision` mechanism. A salvaged item is recorded as an **`approved`** decision (it is an approval of the rewritten suggestion) carrying a salvage marker — this keeps `--resume` and any report consumer treating it as an applied approval, while the `batch_action` / `decision_source` distinguish it.

The markers below are **identical regardless of what triggered the evaluation** — `--claude-decide`, the batch "Let Claude decide" option, or a single-item per-item "Let Claude decide" selection. Recording them the same way means `--resume`, the Markdown apply summary, and the HTML report all surface a Claude-decided / salvaged item the same way, with no per-trigger special-casing.

Approve / skip:

```json
{
  "decision": "approved" or "skipped",
  "reason": "Claude auto-decide: {brief reason}",
  "timestamp": "...",
  "batch_context": {
    "batch_id": "{generated}",
    "batch_action": "claude_auto_decide",
    "decision_source": "claude_auto_decide",
    "importance_at_decision": "{importance}"
  }
}
```

Salvage (note `decision` stays `"approved"`):

```json
{
  "decision": "approved",
  "reason": "Claude auto-decide (salvaged): kept {what}; dropped {what}",
  "timestamp": "...",
  "batch_context": {
    "batch_id": "{generated}",
    "batch_action": "claude_auto_decide_salvage",
    "decision_source": "claude_auto_decide_salvage",
    "importance_at_decision": "{importance}",
    "salvaged_description": "{the rewritten description that was applied}",
    "dropped": "{brief note on what was intentionally left out}"
  }
}
```

## Individual Review Mode

For "Review each individually" mode, group items by importance and present each level with AskUserQuestion.

For each importance level, first display the items as an informational list:

```
== {IMPORTANCE} IMPORTANCE ({count} items) ==

1. [Title]: [Brief reason for needs-human-decision]
   {context line}

2. [Title]: [Brief reason]
   {context line}
```

Then use AskUserQuestion for each importance level:
- Question: "{IMPORTANCE} IMPORTANCE ({count} items): [{comma-separated titles}]. How would you like to handle these?"
- Header: "{IMPORTANCE} items"
- Options:
  - label: "Approve all {IMPORTANCE}", description: "Apply all {count} items at this level"
  - label: "Skip all {IMPORTANCE}", description: "Skip all {count} items at this level"
  - label: "Review individually", description: "Review each item one by one"
  - label: "Next importance level", description: "Skip to the next group"

When the user picks **Review individually**, present each item in turn using the per-item prompt in **Legacy Individual Mode** below — which includes a **Let Claude decide** option so the user can offload any single item they are unsure about to Claude.

## Legacy Individual Mode

For single items, when `batch_enabled == false`, or when the user chose **Review individually** above.

For each item, use AskUserQuestion:
- Question: "How should we handle this item?"
- Header: "Review needed"
- Options:
  - "Apply/Fix this item" -> Process it (apply as-is)
  - "Skip this item" -> Skip and continue
  - "Let Claude decide" -> Offload just this item to Claude: run **"Let Claude Decide" Mode** scoped to this single item (spawn one judging subagent per **Delegate the judgment to a subagent**), then apply the original (approve), apply the rewritten description (salvage), or skip per the subagent's verdict, and continue to the next item
  - "Skip all remaining" -> Exit early

The **Let Claude decide** option lets a user who is unsure about a specific item hand that one decision to Claude; it behaves exactly like `--claude-decide` would for that item (Approve / Salvage / Skip, with salvage preferred over skipping a partially-valid item). Record its outcome with the same `claude_auto_decide` / `claude_auto_decide_salvage` markers shown under **State Recording** so reports and `--resume` treat it identically to a flag-driven auto-decision.
