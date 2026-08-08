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

This is the **same exact rule** every host shim states as its argument
convention, so the two layers cannot diverge — and the dispatcher enforces it
independently: an extra positional, or a question that arrives empty, is exit 3
with the offending token named.

## Step-by-Step Execution

1. **Validate the plan file exists** and that a **non-empty question** was
   supplied (error with the usage hint otherwise).

2. **Hand the question to the dispatcher as one argument — never build a shell
   string out of it.** The question is user-controlled free text and may contain
   quotes, `$(...)` command substitutions, backticks or other shell
   metacharacters. You do **not** have to defend against that yourself: the
   dispatcher takes the plan path and the question as its two positionals and
   normalizes them into the flags the orchestrator parses, writing the question to
   a temp file itself (and deleting it afterwards) when it outgrows the argv
   budget. Copy the user's quoted question through verbatim:

   ```bash
   uv run --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" ask "$PLAN_PATH" "$RAW_QUESTION" [--models ...]
   ```

   Three rules govern that one line:

   - **Quote both positionals.** A plan path or a question containing a space is
     one argv element or the command is wrong.
   - **Pass the plan path as given.** The orchestrator resolves it to an
     OS-native absolute path itself; do NOT wrap it in `$(realpath …)` (on Git
     for Windows `realpath` emits a POSIX `/c/…` path a native Windows Python
     process cannot use).
   - **Never re-type or re-assemble the question.** If the user's question is not
     already a single argument, error out with the usage hint (see the parsing
     rule above) instead of re-joining tokens.

   `--question`, `--question-file` and `--question-env` remain on the
   orchestrator's own parser for a caller that already holds the text in one of
   those forms; the positional above is the documented path and is what the
   dispatcher normalizes into them.

3. **Run detached.** Ask mode fans out across many models and routinely runs far
   longer than a single shell command a host is willing to wait for; a foreground
   run risks being killed mid-flight and losing the in-flight answers. Add
   `--detach` (which implies `--yes`) and poll, or — on a host whose *run
   detached* resolution is its own background mechanism — launch the same command
   through that mechanism with stdout+stderr redirected to a log file in the
   phase dir and `PYTHONUNBUFFERED=1` set. Pick exactly one of the two; never
   both.

   ```bash
   PYTHONUNBUFFERED=1 uv run --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" ask "$PLAN_PATH" "$RAW_QUESTION" \
     [--models ...] --detach
   ```

   A detached run is not subject to your host's shell-command cap, so each
   model's own per-provider `default_timeout` (in `providers.yaml`) governs — no
   cap-vs-provider timeout reconciliation is needed. `PYTHONUNBUFFERED=1` matters
   only on the host-background path: Python block-buffers a non-TTY pipe, which
   leaves the log empty for minutes. A host-background launch **without**
   redirection loses stdout (including the `answers.md` path), which is why the
   redirect is mandatory there; `--detach` writes and reports its own log path, so
   it needs neither.

   ```bash
   uv run --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" status "$PLAN_PATH" --phase ask --view run --wait
   ```

   **Resume**: re-invoking the identical question with `--force` RESUMES (keeps
   already-completed `answer_<model>.md` files, runs only the missing models);
   `--rerun-all` forces a full re-run discarding existing per-model answers.

4. **No validation/grouping/subagent steps.** When the run finishes, read the
   `answers.md` path from the log file
   (`{plan}/ask/<question-slug>-<hash8>/orchestrator-run.log`, final line, or the
   path `status` printed), then read `answers.md` and present a brief summary per
   model plus the output file path, following `references/summary-format.md`.

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

`ask` reaches only three D6 decision sites — the uv preflight (row 1), model
selection (row 2) and partial results (row 3) — and every one of them resolves
without a question, so this mode is unattended on every host:

- **Exit 3** when zero models succeed (all failed): there is nothing to
  aggregate, so this is "needs a human", not a generic error. The retry command
  is named on stdout, and `answers.md` / `.status.json` are still written (with a
  `## Failed Models` section) so the failure is diagnosable.
- **Exit 0** on partial success (≥ 1 model answered): `[PARTIAL_RESULTS: <model>,…]`
  on stdout, `MULTI_LLM_RESULT.state: "partial"`, and the `## Failed Models`
  section enumerating the failures. A partial answer set is usable output, so it
  is deliberately never exit 3 (D6 row 3).
- **Exit 3** when models cannot be resolved (no `--models`, no `--quick`, no
  `defaults.models`) — never a rendered picker (D6 row 2).
