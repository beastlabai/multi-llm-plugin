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

   **Preferred — write the question to a workspace temp file and pass
   `--question-file`.** Temp artifacts live in the workspace temp dir
   `$PROJECT_ROOT/.multi-llm/tmp/` (never the system temp dir, so Bash, native
   processes, and harness tools resolve the same file on every OS). Resolve the
   project root first and treat failure as a hard prerequisite failure:

   ```bash
   PROJECT_ROOT="$(git rev-parse --show-toplevel)"
   ```

   If the command fails or `$PROJECT_ROOT` is empty, STOP with the error:
   "multi-llm requires running inside a git repository." Never fall back to a
   relative path or `$PWD`.

   Then write the **verbatim question text** (no trailing newline) with the
   **Write tool** (NOT Bash) — no question bytes ever transit the shell command
   line, and the Write tool creates parent directories automatically:

   - file_path: `{PROJECT_ROOT}/.multi-llm/tmp/ask_question_{plan_stem}.txt`
     (absolute path — join the resolved project root with the relative part;
     `{plan_stem}` is the plan filename without extension)
   - content: the raw question text exactly as the user supplied it
   - If `{PROJECT_ROOT}/.multi-llm/tmp/.gitignore` does not exist yet, also
     Write it with content `*` so the temp dir ignores itself. Filenames are
     deterministic per plan: each run overwrites the previous file — there is
     no cleanup step, and the user may delete `.multi-llm/tmp/` at any time
     (the next run recreates it).

   Then launch the orchestrator, passing the same path in Bash:

   ```bash
   QUESTION_FILE="$PROJECT_ROOT/.multi-llm/tmp/ask_question_{plan_stem}.txt"
   PYTHONUNBUFFERED=1 uv run --project "${CLAUDE_SKILL_DIR}" -- \
     python "${CLAUDE_SKILL_DIR}/ask_orchestrator.py" \
     --plan-file "$PLAN_PATH" --question-file "$QUESTION_FILE" [--models ...] \
     > "{plan}/ask/<question-slug>-<hash8>/orchestrator-run.log" 2>&1
   ```

   (Launch DETACHED — see step 3. The `<question-slug>-<hash8>` log dir matches the
   orchestrator's output dir; if you don't know it yet, redirect to the workspace
   temp log instead — a Bash redirect does not create directories, so include the
   `mkdir -p` in the same command block:

   ```bash
   mkdir -p "$PROJECT_ROOT/.multi-llm/tmp"
   ... > "$PROJECT_ROOT/.multi-llm/tmp/ask_orchestrator_{plan_stem}.log" 2>&1
   ```

   then read markers/paths from that log with the Read tool, using the absolute
   path built from the resolved project root.)

   **Alternatively — pass the question via a strictly-quoted environment
   variable** the orchestrator reads:

   ```bash
   ASK_QUESTION="$RAW_QUESTION" uv run --project "${CLAUDE_SKILL_DIR}" -- \
     python "${CLAUDE_SKILL_DIR}/ask_orchestrator.py" \
     --plan-file "$PLAN_PATH" --question-env ASK_QUESTION [--models ...]
   ```

   **If passing `--question "<text>"` inline is unavoidable**, the value **must**
   be inside double quotes with every embedded `"`, `` ` ``, `$`, and `\`
   escaped. Prefer the temp-file or env-var mechanisms above precisely so this
   brittle escaping is never required.

   Always quote `"$PLAN_PATH"`, and pass it **as given** — the orchestrator
   resolves it to an OS-native absolute path itself. Do NOT wrap it in
   `$(realpath ...)`: on Git for Windows, `realpath` emits a POSIX `/c/...`
   path that a native Windows Python process cannot use. Use `--project`, not
   `--directory` (known path-duplication bug). Exactly one of `--question` /
   `--question-file` / `--question-env` is required (they are mutually
   exclusive).

3. **Run the orchestrator DETACHED** (Critical Rule 1). Ask mode fans out across
   many models and routinely runs well past **10 minutes** — and the Claude Code
   Bash tool hard-caps `timeout` at **600000 ms (10 min)**; any larger value is
   silently clamped to 600000. A foreground run would therefore be SIGTERM'd
   mid-flight and lose the in-flight answers. So launch with
   `run_in_background: true`, redirecting stdout+stderr to a log file and setting
   `PYTHONUNBUFFERED=1` (see the command block in step 2):

   ```bash
   ... ask_orchestrator.py --plan-file "$PLAN_PATH" \
     --question-file "$QUESTION_FILE" [--models ...] \
     > "{plan}/ask/<question-slug>-<hash8>/orchestrator-run.log" 2>&1
   ```

   Detached runs are NOT subject to the 10-min Bash cap, so each model's own
   per-provider `default_timeout` (in `providers.yaml`) governs — no Bash-vs-provider
   timeout reconciliation is needed. `PYTHONUNBUFFERED=1` is required so Python
   streams output to the log instead of block-buffering it (block-buffering leaves
   the log empty for minutes when stdout is a non-TTY pipe). Backgrounding WITHOUT
   redirection would lose stdout (including the `answers.md` path) — which is why
   the redirect to a log file is mandatory.

   **Resume**: re-invoking the identical question with `--force` RESUMES (keeps
   already-completed `answer_<model>.md` files, runs only the missing models);
   `--rerun-all` forces a full re-run discarding existing per-model answers.

4. **No validation/grouping/subagent steps.** When the background task completes,
   read the `answers.md` path **from the log file**
   (`{plan}/ask/<question-slug>-<hash8>/orchestrator-run.log`, final line),
   then read `answers.md` and present a brief summary per model plus the output
   file path, following the skill's standard Summary Format.

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
