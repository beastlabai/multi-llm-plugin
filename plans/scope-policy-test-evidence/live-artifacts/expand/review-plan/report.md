# Plan Review Report: expand

**Original plan:** expand.md
**Generated:** 2026-10-31 17:40:56
**Models:** claude-code:haiku ($0.01 est., 4,295 output tokens)
**Groups:** 7 (`G<n>` counts from 1; `group_index` in the JSON files counts from 0)
**Suggestions:** 1 HIGH, 6 MEDIUM, 1 LOW (8 total)
**Validation:** 7 valid, 0 invalid, 0 needs human review, 0 validation failed
**Scope policy:** expand. Scope tags are informational.
**Scope tags:** 4 `Within scope`, 0 `Scope extension`, 3 `Scope unclear` (validated, of 7 groups)

> **Note:** An interactive HTML report is also available at `report.html`.
> If you export selections from the HTML report (`user_selections.json`),
> those selections will take precedence over checkboxes in this file.
> Scope tags: `Within scope` completes or corrects the agreed outcome; `Scope extension` adds to it; `Scope unclear` could not be settled. Tags describe proposals; they are separate from validity and selection.

## G6 [7b8c477a]: Naive comma join produces invalid CSV for special characters

- [ ] Skip this group
**Group scope:** `Within scope` (validated) — The Goal already requires CSV with exactly three columns per row, so quoting completes that requirement.
**Validation:** Valid | **Category:** modification | **Priority:** 2 | **Highest Importance:** HIGH | **Models:** claude-code:haiku

### G6S1 [a72407f1]: Naive comma join produces invalid CSV for special characters
- [ ] Skip
**Scope:** `Within scope` (reviewer, provisional) — Makes the required CSV output valid; the Goal already asks for CSV with three columns per row.
**Importance:** HIGH | **Type:** modification | **Section:** ## Implementation | **Model:** claude-code:haiku

The Implementation section says to 'join each row's three fields with commas'. Nothing quotes or escapes fields, so a category containing a comma, a double quote or a newline (for example 'Food, drinks') splits into extra columns or extra rows. The output is then not valid CSV and a downstream reader gets the wrong column count. The Goal requires CSV with exactly three columns per row, so the plan needs a quoting rule, such as RFC 4180 quoting of any field containing a comma, quote or newline, with embedded quotes doubled.

---

## G1 [01221d0f]: Goal's 'keep CLI arguments' and 'existing callers' have no…

- [ ] Skip this group
**Group scope:** `Within scope` (validated) — Adds a check and clarifies the entry point for a requirement the Goal already states.
**Validation:** Valid | **Category:** addition | **Priority:** 3 | **Highest Importance:** MEDIUM | **Models:** claude-code:haiku

### G1S1 [2f7a4a19]: Goal's 'keep CLI arguments' and 'existing callers' have no check
- [ ] Skip
**Scope:** `Within scope` (reviewer, provisional) — A check for a requirement the Goal already states.
**Importance:** MEDIUM | **Type:** addition | **Section:** ## Checks | **Model:** claude-code:haiku

The Goal says 'Keep the current CLI arguments. Existing callers must keep working.' Neither the Implementation nor the Checks section says how this is verified, and the Checks section only exercises the new conversion. A regression in argument parsing would pass the plan's checks unnoticed.

### G1S2 [7ab86ddf]: Plan never identifies the existing CLI or whether command is new
- [ ] Skip
**Scope:** `Scope unclear` (reviewer, provisional) — The baseline asks to keep existing arguments but does not say which command or arguments they are.
**Importance:** MEDIUM | **Type:** clarification | **Section:** ## Goal | **Model:** claude-code:haiku

The Goal says 'Implement a local command' and then 'Keep the current CLI arguments. Existing callers must keep working.' The plan does not name the existing CLI entry point, its current arguments, or the file that holds them. It is unclear whether this is a new subcommand alongside the existing CLI or a change to an existing command. Implementers must guess which arguments are 'current' and which callers exist, and the answer determines what must not change.

---

## G2 [2032cbd7]: Destination overwrite and partial-write behaviour is…

- [ ] Skip this group
**Group scope:** `Scope unclear` (validated) — The baseline does not settle overwrite or failure policy, and the fixes differ in scope.
**Validation:** Valid | **Category:** clarification | **Priority:** 2 | **Highest Importance:** MEDIUM | **Models:** claude-code:haiku

### G2S1 [8eae5301]: Destination overwrite and partial-write behaviour is unstated
- [ ] Skip
**Scope:** `Scope unclear` (reviewer, provisional) — The baseline does not address existing output files, so the overwrite policy is not settled.
**Importance:** MEDIUM | **Type:** clarification | **Section:** ## Implementation | **Model:** claude-code:haiku

Implementation says 'Open the destination for writing and write the resulting lines.' It does not say whether an existing output file is overwritten, whether the command fails if the path exists, or whether a failure partway through leaves a truncated file. Opening for writing truncates the destination before the input is validated, so a bad input could destroy an existing output.

---

## G4 [6d295776]: Error handling for bad input is not specified

- [ ] Skip this group
**Group scope:** `Scope unclear` (validated) — The baseline does not state error behaviour, so failing fast versus skipping records is open.
**Validation:** Valid | **Category:** addition | **Priority:** 2 | **Highest Importance:** MEDIUM | **Models:** claude-code:haiku

### G4S1 [4fbe8769]: Error handling for bad input is not specified
- [ ] Skip
**Scope:** `Scope unclear` (reviewer, provisional) — The baseline does not state error behaviour, so the choice between failing fast and skipping records is open.
**Importance:** MEDIUM | **Type:** addition | **Section:** ## Implementation | **Model:** claude-code:haiku

Nothing says what happens on a missing or unreadable input file, invalid JSON, a top-level value that is not an array, a record missing date, category or amount, or a record with extra fields. The Implementation only describes the success path ('Read the JSON, iterate records'). The implementer must choose between failing with a non-zero exit, skipping records, or writing partial output, and each choice changes what existing callers see.

---

## G5 [8c00f53c]: Field formats for date and amount are not specified

- [ ] Skip this group
**Group scope:** `Within scope` (validated) — Specifies value formats for the columns the Goal already defines, which completes the stated requirement.
**Validation:** Valid | **Category:** clarification | **Priority:** 2 | **Highest Importance:** MEDIUM | **Models:** claude-code:haiku

### G5S1 [a78cda4d]: Field formats for date and amount are not specified
- [ ] Skip
**Scope:** `Within scope` (reviewer, provisional) — Specifies the value format of the columns the Goal already defines.
**Importance:** MEDIUM | **Type:** clarification | **Section:** ## Goal | **Model:** claude-code:haiku

The Goal says each record has date, category and amount, and the CSV columns are date,category,amount. The plan does not say whether date is an ISO 8601 string or another format, whether amount is a JSON number or a string, how many decimal places are written, or whether values are validated or passed through. Without this, output such as 12.5 versus 12.50 or 2026-10-31 versus 31/10/2026 is decided by the implementer, and the expected output of the check cannot be stated.

---

## G7 [222eb113]: Only a happy-path check, which cannot catch quoting bugs

- [ ] Skip this group
**Group scope:** `Within scope` (validated) — Adds verification that the CSV output the Goal requires is correct, including column order and header handling.
**Validation:** Valid | **Category:** addition | **Priority:** 2 | **Highest Importance:** MEDIUM | **Models:** claude-code:haiku

### G7S1 [d79710cd]: Only a happy-path check, which cannot catch quoting bugs
- [ ] Skip
**Scope:** `Within scope` (reviewer, provisional) — Adds a check for behaviour the plan's own Goal and Implementation already require.
**Importance:** MEDIUM | **Type:** addition | **Section:** ## Checks | **Model:** claude-code:haiku

The Checks section allows one happy-path record and asserts a header plus one data row. A record with no commas, quotes or newlines passes with the naive join described in Implementation, so the check cannot detect the escaping defect. Nothing checks the 'one header row' rule across several records, or the column order date,category,amount.

---

## G3 [d227bc9c]: Empty array and output line-ending rules are unspecified

- [ ] Skip this group
**Group scope:** `Scope unclear` (validated) — The baseline gives input encoding and columns but not output bytes or the empty case.
**Validation:** Valid | **Category:** clarification | **Priority:** 2 | **Highest Importance:** LOW | **Models:** claude-code:haiku

### G3S1 [091f1841]: Empty array and output line-ending rules are unspecified
- [ ] Skip
**Scope:** `Scope unclear` (reviewer, provisional) — The baseline gives the input encoding and the column layout but not output bytes or the empty case.
**Importance:** LOW | **Type:** clarification | **Section:** ## Goal | **Model:** claude-code:haiku

The plan does not say what an empty JSON array produces: a header only, or an empty file. It also does not say whether lines end with LF or CRLF, whether the file has a trailing newline, or what encoding the output uses, though the input is specified as UTF-8. Each affects byte-level output and any test that compares the file.

---
