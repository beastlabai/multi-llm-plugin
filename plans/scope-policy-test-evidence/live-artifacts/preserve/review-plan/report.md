# Plan Review Report: preserve

**Original plan:** preserve.md
**Generated:** 2026-10-31 17:40:56
**Models:** claude-code:haiku ($0.01 est., 3,806 output tokens)
**Groups:** 6 (`G<n>` counts from 1; `group_index` in the JSON files counts from 0)
**Suggestions:** 1 HIGH, 4 MEDIUM, 1 LOW (6 total)
**Validation:** 6 valid, 0 invalid, 0 needs human review, 0 validation failed
**Scope policy:** preserve. Additional scope was not solicited from reviewers; extension counts cover returned findings only.
**Scope tags:** 6 `Within scope`, 0 `Scope extension`, 0 `Scope unclear` (validated, of 6 groups); 0 deferred: scope preservation requested

> **Note:** An interactive HTML report is also available at `report.html`.
> If you export selections from the HTML report (`user_selections.json`),
> those selections will take precedence over checkboxes in this file.
> Scope tags: `Within scope` completes or corrects the agreed outcome; `Scope extension` adds to it; `Scope unclear` could not be settled. Tags describe proposals; they are separate from validity and selection.

## G3 [6e642be4]: Comma-joined rows break when a field holds a comma

- [ ] Skip this group
**Group scope:** `Within scope` (validated) — Quoting and escaping fields to RFC 4180 makes the output the CSV the Goal requires and adds no new behaviour.
**Scope outcome:** Eligible under scope preservation
**Validation:** Valid | **Category:** modification | **Priority:** 2 | **Highest Importance:** HIGH | **Models:** claude-code:haiku

### G3S1 [3a591f36]: Comma-joined rows break when a field holds a comma
- [ ] Skip
**Scope:** `Within scope` (reviewer, provisional) — Makes the output a valid CSV file as the Goal requires; adds no new behaviour.
**Importance:** HIGH | **Type:** modification | **Section:** ## Implementation | **Model:** claude-code:haiku

## Implementation says 'join each row's three fields with commas'. That is not CSV: a category such as 'Food, groceries' or one containing a double quote or newline adds extra columns or splits the row, so the output file is malformed for valid input. The goal is a CSV file, so fields that contain commas, quotes or line breaks must be quoted and escaped (RFC 4180, e.g. Python's csv module). This is a correctness bug in the approach itself, not a style choice.

---

## G1 [747cbd06]: Behaviour on empty or malformed input is unspecified

- [ ] Skip this group
**Group scope:** `Within scope` (validated) — Specifying error handling for the stated JSON-array input clarifies the agreed command requirement without adding features.
**Scope outcome:** Eligible under scope preservation
**Validation:** Valid | **Category:** addition | **Priority:** 2 | **Highest Importance:** MEDIUM | **Models:** claude-code:haiku

### G1S1 [32d82932]: Behaviour on empty or malformed input is unspecified
- [ ] Skip
**Scope:** `Within scope` (reviewer, provisional) — Specifies error handling for the stated input (a JSON array of records with three fields).
**Importance:** MEDIUM | **Type:** addition | **Section:** ## Implementation | **Model:** claude-code:haiku

The plan does not say what happens when the input is an empty array (header only, or an error?), is not valid JSON, is not a top-level array, a record lacks date, category or amount, or has extra fields. An implementer must choose between writing partial output, failing with an error, and skipping records. The plan should state the behaviour for each case, and whether a failed run leaves the output file absent or untouched.

---

## G2 [806a326e]: CLI contract is not defined and 'existing callers' has no…

- [ ] Skip this group
**Group scope:** `Within scope` (validated) — Naming the command and its input and output arguments clarifies the stated contract rather than extending it.
**Scope outcome:** Eligible under scope preservation
**Validation:** Valid | **Category:** clarification | **Priority:** 2 | **Highest Importance:** MEDIUM | **Models:** claude-code:haiku

### G2S1 [4b498a6c]: CLI contract is not defined and 'existing callers' has no referent
- [ ] Skip
**Scope:** `Scope unclear` (reviewer, provisional) — The baseline asks to keep current CLI arguments but does not enumerate them, so the exact contract is not settled.
**Importance:** MEDIUM | **Type:** clarification | **Section:** ## Goal | **Model:** claude-code:haiku

## Goal says 'Keep the current CLI arguments. Existing callers must keep working', but the plan never lists the current arguments, names the command or entry point, or says which file it lives in. It also says the command 'takes input and output file paths' without saying whether those are new arguments or existing ones. Since this is described as a new local command, it is unclear what 'existing callers' refers to. Without the argument names and the command name, an implementer cannot know what must be preserved.

---

## G4 [26282e57]: Date and amount output format is not stated

- [ ] Skip this group
**Group scope:** `Within scope` (validated) — Stating that values pass through as given, or naming a format, clarifies how the three stated fields are written.
**Scope outcome:** Eligible under scope preservation
**Validation:** Valid | **Category:** clarification | **Priority:** 2 | **Highest Importance:** MEDIUM | **Models:** claude-code:haiku

### G4S1 [bec4887d]: Date and amount output format is not stated
- [ ] Skip
**Scope:** `Within scope` (reviewer, provisional) — Clarifies how the three stated fields are written to the CSV the Goal defines.
**Importance:** MEDIUM | **Type:** clarification | **Section:** ## Goal | **Model:** claude-code:haiku

The plan does not say whether dates are written as given in the JSON or reformatted, nor how amounts are rendered (for example 12.5 versus 12.50, or floating-point artefacts such as 0.30000000000000004). Output for these fields is therefore open to interpretation and the check cannot confirm the expected value. The plan should state that values pass through as given, or name the format.

---

## G5 [5a3dca0e]: Happy-path check cannot catch the quoting bug

- [ ] Skip this group
**Group scope:** `Within scope` (validated) — Adding a comma and quote check and quoting so the output is valid CSV completes the agreed CSV requirement, though the quoting rule itself is not spelled out in the baseline.
**Scope outcome:** Eligible under scope preservation
**Validation:** Valid | **Category:** addition | **Priority:** 2 | **Highest Importance:** MEDIUM | **Models:** claude-code:haiku

### G5S1 [706d1a00]: Happy-path check cannot catch the quoting bug
- [ ] Skip
**Scope:** `Within scope` (reviewer, provisional) — A check for the CSV validity the Goal already requires.
**Importance:** MEDIUM | **Type:** addition | **Section:** ## Checks | **Model:** claude-code:haiku

## Checks only asks for 'a happy-path record' with a header and one data row. A record whose fields contain no commas, quotes or newlines passes even if fields are joined naively, so the check cannot fail on the defect above. Add a check with a category containing a comma (and ideally a quote) and confirm it reads back as the same three fields.

---

## G6 [de876c63]: Output encoding and line endings are unspecified

- [ ] Skip this group
**Group scope:** `Within scope` (validated) — Specifying the encoding and line endings of the CSV file the Goal asks for clarifies an agreed requirement and adds no new feature.
**Scope outcome:** Eligible under scope preservation
**Validation:** Valid | **Category:** clarification | **Priority:** 2 | **Highest Importance:** LOW | **Models:** claude-code:haiku

### G6S1 [73ba765f]: Output encoding and line endings are unspecified
- [ ] Skip
**Scope:** `Within scope` (reviewer, provisional) — Specifies the encoding of the output file the Goal asks for.
**Importance:** LOW | **Type:** clarification | **Section:** ## Implementation | **Model:** claude-code:haiku

The input is stated to be UTF-8, but the plan does not say what encoding and line endings the CSV is written in. A category with non-ASCII characters (for example 'Café') could be written in the platform default encoding, which is a problem on some systems. The plan should state the output encoding, normally UTF-8.

---
