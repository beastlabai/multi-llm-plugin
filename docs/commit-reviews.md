# Commit reviews

Reviewing what was committed rather than a plan: the range you can type, what a
run prints, how the report and the apply fit together, and what the fixes may
touch. Back to the [README](../README.md).

---

`--review-commits`, `--apply-commit-fixes` and `--improve-commits` are the commit counterparts of the three code modes. They review what was committed, not what is on disk, and need no plan: the commit messages, taken together, are the stated intent, and the range's net change (`git diff <base> <head>`) is what every model reviews.

| Command | Does | Counterpart |
| --- | --- | --- |
| `--review-commits [<range>]` | every model reviews the range; the findings are grouped, validated and reported, each naming the commit that introduced it | `--review-code` |
| `--apply-commit-fixes [<range>]` | applies that review's validated fixes as uncommitted edits | `--apply-code-fixes` |
| `--improve-commits [<range>]` | the two above in one unattended command | `--improve-code` |

Nothing commits, amends or rebases: the fixes are uncommitted edits on top of `HEAD`, for you to read and commit.

**The range.** Python resolves it, the same way on every host. Leave it out to review the last commit:

| You type | Reviews |
| --- | --- |
| nothing | the last commit, against its first parent |
| `<rev>`: `abc1234`, `HEAD~2`, a tag, a branch | that one commit |
| `A..B`, `A..`, `..B` | the commits after `A` up to `B`; an omitted side is `HEAD`, and `A` must be an ancestor of `B` |
| `A...B` | the commits on `B` since it forked from `A` |
| `last commit`, `last N`, `last N commits`, `N commits`, `-N` | the `N` newest commits, ending at `HEAD` |
| `today`, `yesterday`, `this week`, `last N days` (or `hours`, `weeks`; `past` works as `last`), `since YYYY-MM-DD` (optionally with `HH:MM`) | the commits in that window, by committer date in your local time zone |
| `since <rev>` | the commits after `<rev>` up to `HEAD` |
| `this branch`, `branch` | the commits since `HEAD` forked from the default branch (`origin/HEAD`, else `main`, else `master`) |

- **`N`** is a positive whole number, or `one` to `twelve`, so `last two days` works. Commits a merge brought in are reviewed as part of that merge commit (first parent only).
- **Anything else is refused** at exit 3 with `[RANGE_UNRECOGNIZED]` and this list of forms; nothing is guessed. A range with no commits in it (an empty window, `A..A`) exits 2 with `[NOTHING_TO_REVIEW]`. A markdown path is never taken as a plan here: to review commits against one, pass it with `--context-plan`.
- **The range goes right after the mode word, before any flag**, and needs no quotes: `--review-commits last 2 commits --quick`. A word after a flag may be that flag's value (`--models` takes several), so range words typed after one are refused with `[RANGE_AFTER_FLAGS]`, naming the command with the range moved, never reviewed as `HEAD`.
- **One range per run.** To cover two, type one `A..B` that spans them, or run once per range. `--files` narrows a review to some of the range's files and never widens it.

**What it prints first.** Before any model runs, every commit mode prints the commits it acts on, one line each, with the range's size:

```text
[COMMIT_RANGE] a1b2c3d..e4f5a6b (3 commits, 14 files, +412 −88)
  e4f5a6b  2026-10-24  Wait for the clock to tick in the shared-ps-table row
  c3d4e5f  2026-10-24  Windows: kill process trees through Job Objects, not…
  b2c3d4e  2026-10-23  docs: slim skill development guide and extract reference…
[COMMIT_RANGE_END]
```

The block shows the 20 newest commits, then where the full list is. A date phrase adds its window, and a range over 50 commits, 200 files or 20000 changed lines adds a `[RANGE_LARGE]` line naming ways to narrow it; the run continues. `[COMMIT_BRIEF] <path>` follows, naming the commit brief. Every follow-up command a run prints names the range pinned as `<base12>..<head12>` (the commit brief's path for a range that starts at the first commit), never the phrase you typed, so following it tomorrow reaches the same review; `--status` takes that pinned range too.

**Reviewing before applying.** `--review-commits` writes the same two reports as `--review-code`, in `plans/commit-reviews/<base>-<head>/code-review/`. Both open with the commit block, and each finding names the commit that introduced it ("Introduced by `e4f5a6b`" and its subject, marked "(model)" when `git blame` could not decide and the model's claim was taken); `report.html` also filters and groups the findings by commit. Tick `Skip` or a `Mark valid` / `Mark invalid` override in `report.md`, or make the same choices in `report.html` and export `user_selections.json` next to it, exactly as after `--review-code`, then run `--apply-commit-fixes`:

- **With no range,** it applies the newest completed commit review when that review ends at `HEAD` or behind it in `HEAD`'s history, else the newest one ending at `HEAD`, and names its pick on a `[COMMIT_REVIEW_SELECTED]` line before any fix is applied. An unfinished review is never picked. With no match, it exits 3 listing up to five completed reviews, each with its relation to `HEAD` and its command.
- **With a range,** it applies that range's review, which must have completed. A phrase is resolved again when the apply runs, so a day later `last 2 days` names other commits; the pinned range the review printed always names the reviewed one.

`--improve-commits` skips this step: it is unattended, and a judge agent decides every `needs-human-decision` item. For commits you do not fully trust, such as a fetched pull request, `--review-commits`, then the report, then `--apply-commit-fixes` is the safer path.

**`--focus` and `--context-plan`.** `--focus "<text>"` steers the review toward a priority; it is not a filter. `--context-plan <path>` hands the review a plan as read-only context, and the commits are checked against it too.

- **Recorded with the commit brief.** A later run of the same range that leaves them out keeps them, so the printed follow-ups, `--resume` and a no-argument `--apply-commit-fixes` all continue the same review. Typing a different value, or `""` to clear one, restarts the review round (`[COMMIT_REVIEW_RESTARTED]`): every model reviews again, and the replaced round, its reports and your marks in them move to `code-review/restarted-<stamp>/`, with a `[COMMIT_DECISIONS_DROPPED]` line counting the decisions not carried over.
- **The context plan is copied when the review starts.** Every reviewer and validator reads that copy, so editing the file later changes nothing in that review: the next `--review-commits` or `--improve-commits` of the range warns with `[CONTEXT_PLAN_CHANGED]` and carries on with the copy. To review against the edited text, type `--context-plan <path>` again.
- **`--apply-commit-fixes` keeps what the review used.** It never restarts a review: a different value typed there prints `[COMMIT_REVIEW_KEPT]`, with the commands that would review with it, and applies the review on record. A fix aimed at the context plan itself is skipped and named, never applied.

**When files changed since the reviewed commit.** The review reads every file at the range's last commit, never from your working tree, so what is on disk does not change what is reviewed. When reviewed files differ on disk from that commit (later commits, uncommitted edits), the commit block adds a line naming them, `[SCOPE_DRIFT] 2 of 14 reviewed files differ on disk from e4f5a6b: utils/process_control.py, tests/test_detach.py`, and the run continues. The apply compares each fix's files with the reviewed commit before anything is written, and again just before each batch: a finding with a file changed since is skipped whole and named in the `[COMMIT_FIXES]` block the apply prints before its first batch, and no batch may write a file that differed from the reviewed commit when the apply started. So after a further commit, `--apply-commit-fixes` still applies every fix to the files that commit left alone.

**Before you run it.** `--review-commits` writes only the commit brief and its folder. The other two edit source files, as `--improve-code` does:

- *What gets written, and how to undo it.* `--apply-commit-fixes` and `--improve-commits` edit source files anywhere in the repository as uncommitted edits, `--improve-commits` unattended. When the run uses `--exec-subagents` (the Codex and OpenCode default), each batch keeps a pre-batch snapshot that `multi_llm.py util restore-batch` restores; nothing backs up the repository as a whole, so git is the way back. Commit or stash first, so `git diff` shows exactly what the run changed and `git restore` undoes it.
- *Re-runs and `--resume`.* `--improve-commits` follows `--improve-code`'s rules ([Workflows in depth](workflows.md#review-and-apply-in-one-command)), and its apply always gets the reviewed range's base. A stopped run names `--improve-commits <base12>..<head12> --resume`, which never re-runs a completed review and keeps every batch already applied; once both steps are done it prints `[ALREADY_COMPLETE]` and exits 2. `--interactive` is refused: run `--review-commits --interactive`, then `--apply-commit-fixes`.
- *What it costs.* As for the code modes: one review call per model in the review's list, then validation, judging and one write dispatch per batch. A broad phrase such as `this branch` adds no calls, but each reviewer has more to read and covers it more thinly, which `[RANGE_LARGE]` warns about.

---

Back to the [README](../README.md) · [Workflows in depth](workflows.md) · [Per-project configuration](configuration.md)
