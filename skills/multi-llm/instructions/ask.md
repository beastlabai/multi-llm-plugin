# Ask Mode Instructions

Ask each configured LLM a **free-text question** about a plan and aggregate the
raw markdown answers into a single file. This is a lightweight, **read-only**
Q&A side-channel — no JSON schema, no grouping, no validation subagents, no
state-machine phases.

## Usage

```bash
/multi-llm:multi-llm --ask <plan_path> "<free text question>" [--models provider:model ...] [--quick] [--interactive] [--force]
```

- `--force` re-asks **all** models even when prior answer files exist (use after
  editing the plan, or to get a fresh round of opinions).

## Question argument parsing rule (exact)

The question is **a single argument**, conventionally a quoted string, passed
straight through to the orchestrator. Concretely:

- The first positional token after `--ask` is the **plan path**; the next single
  argument is the **question**. The `--models` / `--quick` / `--interactive` /
  `--force` flags are **position-independent** — they may appear before the plan
  path, between plan path and question, or after the question (argparse handles
  ordering on the orchestrator side).
- **The question must be supplied as one (quoted) argument.** This preserves
  spacing, embedded quotes, and any flag-like substring inside the question.
  Because the whole question is one quoted token, text such as
  `"should we keep --quick mode?"` or `"compare --models opus vs sonnet"` is
  treated entirely as question text — the leading `--quick`/`--models` inside the
  quotes are **not** parsed as flags.
- **If multiple bare (unquoted) tokens appear where the question is expected**,
  do **not** silently re-join them (joining loses the user's quoting/spacing and
  is ambiguous against flag-like tokens). Instead **error out with the usage
  hint** and tell the user to quote the question as a single argument.
- **An empty/whitespace-only question is an error** (usage hint), not a silent
  no-op.

Parsing examples (each `<question>` is exactly one quoted argv element):

- `/multi-llm:multi-llm --ask plans/x.md "Is the rollback strategy sufficient?"`
  → plan=`plans/x.md`, question=`Is the rollback strategy sufficient?`
- `/multi-llm:multi-llm --ask plans/x.md "should we keep --quick mode?" --quick`
  → plan=`plans/x.md`, question=`should we keep --quick mode?` (the `--quick`
  **after** the quoted question is the flag; the one **inside** the quotes is
  question text), flags=`--quick`.
- `/multi-llm:multi-llm --quick --ask plans/x.md "Summarize this plan"`
  → flags may precede the plan path; question=`Summarize this plan`.
- `/multi-llm:multi-llm --ask plans/x.md Is the rollback sufficient?`
  → **error** (multi-word question not quoted) with the usage hint to quote it.
- `/multi-llm:multi-llm --ask plans/x.md ""`
  → **error** (empty question) with the usage hint.

This is the **same exact rule** stated in SKILL.md's Mode Detection Logic, so the
two layers cannot diverge.

## Step-by-Step Execution

1. **Validate the plan file exists** and that a **non-empty question** was
   supplied (error with the usage hint otherwise).

2. **Pass the question safely — do not interpolate raw user text into the shell
   command line.** The question is user-controlled free text and may contain
   quotes, `$(...)` command substitutions, backticks, or other shell
   metacharacters; building a `bash` string from it (or from an unquoted
   `$PLAN_PATH`) would break the invocation or execute unintended shell syntax.
   Use a safe argv-construction mechanism, in this order of preference:

   **Preferred — write the question to a temp file and pass `--question-file`.**
   Write the verbatim question bytes to a temp file (no trailing newline — use
   `printf '%s'`), so no question bytes ever transit the shell command line:

   ```bash
   QUESTION_TMPFILE="$(mktemp)"
   printf '%s' "$RAW_QUESTION" > "$QUESTION_TMPFILE"
   uv run --project ${CLAUDE_SKILL_DIR} -- \
     python ${CLAUDE_SKILL_DIR}/ask_orchestrator.py \
     --plan-file "$(realpath "$PLAN_PATH")" --question-file "$QUESTION_TMPFILE" [--models ...]
   ```

   **Alternatively — pass the question via a strictly-quoted environment
   variable** the orchestrator reads:

   ```bash
   ASK_QUESTION="$RAW_QUESTION" uv run --project ${CLAUDE_SKILL_DIR} -- \
     python ${CLAUDE_SKILL_DIR}/ask_orchestrator.py \
     --plan-file "$(realpath "$PLAN_PATH")" --question-env ASK_QUESTION [--models ...]
   ```

   **If passing `--question "<text>"` inline is unavoidable**, the value **must**
   be inside double quotes with every embedded `"`, `` ` ``, `$`, and `\`
   escaped. Prefer the temp-file or env-var mechanisms above precisely so this
   brittle escaping is never required.

   Always quote `"$(realpath "$PLAN_PATH")"` consistently (inner **and** outer
   quotes) regardless of which question-passing mechanism is used. Use
   `--project`, not `--directory` (known path-duplication bug). Exactly one of
   `--question` / `--question-file` / `--question-env` is required (they are
   mutually exclusive).

3. **Run the orchestrator in the FOREGROUND** (Critical Rule 1 — do NOT use
   `run_in_background`; background runs lose stdout, including the final
   `answers.md` path).

   **Bash-tool timeout:** set the Bash `timeout` high enough that it never
   undercuts the per-model provider timeouts. Providers in `providers.yaml`
   currently use `default_timeout` up to **1800s (30 min)** for `claude-code`
   and `codex`. The Bash `timeout` must be **≥ the max provider
   `default_timeout` over the asked models, plus stagger and startup overhead**:
   roughly `max(provider default_timeout) + len(models) * PROVIDER_STAGGER_DELAY`
   plus a safety margin (`PROVIDER_STAGGER_DELAY = 2.0s`). With the current
   1800s ceiling, use a Bash `timeout` of **~2000000 ms (≈33 min)** so a single
   slow model never gets killed before its own per-model timeout fires (which
   would lose the `answers.md` path printed on stdout). Keep this in sync with
   `providers.yaml` — if provider timeouts are raised, raise the Bash timeout
   too. (Alternatively, always pass an explicit `--timeout` to the orchestrator
   that is small enough to guarantee the whole run finishes inside the Bash
   limit.)

4. **No validation/grouping/subagent steps.** On completion, read `answers.md`
   (its path is the final line of stdout) and present a brief summary per model
   plus the output file path, following the skill's standard Summary Format.

## Output Files

```
plans/todo/my-feature/ask/<question-slug>-<hash8>/
├── answers.md            # aggregated markdown answers (regenerated each run)
├── answer_<model>.md     # per-model raw markdown answer
├── log_<model>.txt       # per-model provider debug log
├── error_<model>.log     # per-model error log (only on failure)
└── .status.json          # resume detection + collision/freshness guard
```

## Resume & Freshness

- Re-asking the **byte-for-byte identical** question maps to the same directory
  and resumes: models with a non-empty `answer_<model>.md` are skipped and
  `answers.md` is regenerated from all current answers. Use `--force` to re-ask
  every model.
- On resume, if the plan file changed since the answers were generated, the
  orchestrator prints a stale-plan warning recommending `--force` (it does not
  auto-invalidate answers).

## Exit Status

- **Exit 1** when zero models succeed (all failed); `answers.md` and
  `.status.json` are still written (with a `## Failed Models` section) so the
  failure is diagnosable.
- **Exit 0** on partial success (≥ 1 model answered), with the `## Failed Models`
  section enumerating the failures.
