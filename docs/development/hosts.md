# Host integration and packaging

Read when changing a host shim, capability declaration, installer, generated host document, or package.

Back to the [development guide](../../skills/multi-llm/AGENTS.md). These notes retain detailed contracts and regression history from that guide. Paths beginning with `docs/`, `packaging/`, or `plans/` are repository-relative; other implementation paths are relative to `skills/multi-llm/`. Dated incidents explain the rules; verify version-specific claims against current code.

## Contents

- [Claude Code entry shim](#claude-code-entry-shim)
- [Host capability ownership](#host-capability-ownership)
- [Host-neutral instruction core](#host-neutral-instruction-core)
- [Install and uninstall](#install-and-uninstall)
- [Package distribution](#package-distribution)
- [Detecting the active route](#detecting-the-active-route)
- [Generated host documentation](#generated-host-documentation)
- [OpenCode version updates](#opencode-version-updates)

## Claude Code entry shim

**`SKILL.md` is the Claude Code *shim*, not the contract (D5c).** ~120 lines: the
skill-dir binding, the uv preflight, the flag→mode-word convention, the `util instructions <mode>` read
step, the three baked verb sentences, the one dispatcher command with `--host claude-code`, and
worked examples. Everything else moved — the workflow to `instructions/`, provider
configuration to `docs/providers.md`, the standard report shape to `references/summary-format.md` — and
the argv-scanning shell preamble is **deleted**: it re-implemented `modes.py` plus the
dispatcher's own positional normalization in a shell syntax only one host runs, and the dispatcher
already rejects a bad plan path or an empty question with exit 3 naming the token.

Its three verb sentences are **byte-identical** to `host_capabilities.verb_sentences("claude-code")` — asserted by
`tests/test_shim_correctness.py`, because `installable: false` means nobody renders this file and the
reference host would otherwise be the one host free to drift from the data that governs the others.
That test is parametrized over every shipped shim (line budget against `shim_max_lines`, no
unfilled slot, no deferral to the agent, no env assignment beyond `MULTI_LLM_SKILL_DIR`, no core
paragraph copied in, every named mode resolving through the real parser), so the per-host
`.tmpl` files pick it up with no edit.

## Host capability ownership

**`hosts/capabilities.yaml` is the verb→mechanism map, and it is data (D5b).** One entry per host id,
validated against `schemas/host_capabilities.schema.json` (every field required except a closed set of lint/doc keys
— `shim_max_lines`, `headless_invocation`, `package` — each admissible only
because it governs a lint or a generated document and never a dispatch decision; a defaulted
*capability* is the silent guess the schema exists to remove).
`subagents`/`ask`/`detach` are the three verbs;
`native_tool` is a per-verb map and **the only place a host tool name is allowed to exist**
outside a shim.

Every value carries a `provenance`
(`verified`|`documented`|`unknown`) and **a
`documented`-only value never selects the non-conservative path** — Codex and OpenCode
ship `subagents: executor` and `detach: python`/`required` for exactly that
reason, and a test enforces it. Read through `utils/host_capabilities.py`: `resolve_host_id()` is
`--host` → `$MULTI_LLM_HOST` → `claude-code` (declared, **never
sniffed**), an unknown id is exit 3, `require_executor_runner()` makes `--exec-subagents` exit 3 on a
host whose entry declares no `agent-readonly`/`agent-write` runner, and
`verb_sentence()` renders the one fixed phrasing per enum value that a shim bakes in.

`subagent_executor.effective_ceiling()` takes the host leg of D3i's minimum from the same entry.
`references/host-capabilities.md` is **generated** by `hosts/render_capabilities.py` (`--check` is the CI
staleness guard, byte-compared); `hosts/snippets/*.md` are the host-independent shim partials (uv
preflight, `--detach` poll loop) `install-host` interpolates into any template
that carries their slot — the two shipped shims carry neither, because a 100-line rendered budget
cannot hold uv-preflight's inlined 31-line install message *and* the poll loop, so both name
`hosts/uv-missing.txt` and print it exactly as the reference shim does.

**Nothing here is on an agent read path** — a test asserts no instruction file, reference or shim
names either the YAML or the generated markdown, because an agent reasoning over a capability matrix
mid-run is a worse failure mode than the one the verbs remove. `utils/unattended.INVENTORY` is D5d's
closed per-`ask`-class inventory over the same rows, with one renderer
(`render_option_block`) serving both presentations and refusing the fail-closed rows.

**`next_command` is the one required key that decides no dispatch**
(plans/next-phase-command.md §2.1): a host-level map, keyed by route — `default` (the
route `invocation` documents: the plugin on claude-code, the file shim on codex and
opencode), `package` (required exactly when the host has a `package:`
block, forbidden otherwise) and `skill` (claude-code's plain-skill install under
`install_paths`) — each a template ending in `{args}` with no other brace.

It is required rather than optional because `utils/next_phase.py` reads it at run time to build
output, and a host with none would silently print no next command; it is host-level rather than
inside `package:` because that block is admissible only while nothing at run time reads
it. `host_capabilities.next_command_template(host, route)` reads `default` for a `None` route and
**raises** for a key the host lacks, never falling back (falling back happens once, in
`resolve_route`).

`provenance.next_command` is a map with the same keys, one value per route, and every cell shipped
`documented` until T10, including on hosts whose `invocation` is
`verified`: only a recorded run that typed the printed command into that host and route
and saw it dispatch — named in the promotion comment, with its `RECORDED_SMOKES` row under the
dotted name (`next_command.package`) — makes one `verified`. The ledger tests and
`render_capabilities._verified_cell` flatten a map row into those dotted names, so a nested promotion cannot pass
unrecorded.

Since T10 (2026-10-01), `codex` and `opencode` carry
`next_command: {default: verified, package: verified}`, each promoted on container runs that typed the printed command into a fresh
headless session and saw the dispatcher record the named mode with an unquoted, spaced plan path.
`claude-code`'s `next_command.skill` is `verified` too (runs 20261001T114907Z
on `openai/gpt-6-luna` via OpenRouter and 20261001T121548Z on `sonnet` via the
subscription); its `default` (plugin route) stays `documented`, because the
container lane stages claude-code as a plain skill and cannot install the plugin, so the T11 Stop
hook has no lane coverage either.

The ledger (`RECORDED_SMOKES` in `tests/test_host_capabilities.py`) now includes claude-code; its
`subagents` and `ask` cells are exempted by name
(`PRE_LEDGER_VERIFIED`), because they were verified in 227af2d before the ledger existed and no
recorded run backs them, and the exemption never covers `next_command`. The host lane's
`live_next_command` tier (`make e2e-container-host-next-command`, excluded from `make e2e-container-host`) runs one
`generate-tasks` row per host and route on `plans/next phase.md`; `entrypoint.run_typed_followup` then
hands the printed `user_next_step.command` verbatim to a fresh headless session, stopped once its
receipt shows the named mode.

`TestTheNextCommandLiveCheck` grades criteria 4, 5 and the typed template over `tests/container/next_command_check.py`, and
each record carries `rows[].next_command`, including the model actually driven. codex and
claude-code host rows run on the developer's own subscription whenever the credential file
`creds.yaml` allowlists for them is present (`harness.default_host_creds`; no OpenRouter
variable enters the container), and on the OpenRouter key otherwise; opencode always uses the key,
and `MLLM_E2E_HOST_CREDS=openrouter|fallback` forces one mode on every row (2026-10-02). On the subscription each host
drives its `subscription.model` pin: codex `gpt-6-luna` at `model_reasoning_effort = "high"`,
claude-code `sonnet`. The live provider lane follows the same rule
(`LIVE_SUBSCRIPTIONS`, `MLLM_LIVE_CREDS`).

Model policy for paid test runs on the key (`tests/container/host_matrix.yaml` carries the full comment):
claude-code on `openai/gpt-6-luna`, every other host on `deepseek/deepseek-v4.1-flash`, never Haiku
through OpenRouter. A test holds each template's prefix inside the prose that documents its route
(`default`/`skill` against `invocation`,
`package` against `package.invocation`), which is why claude-code's
`invocation` carries a trailing plain-skill clause.

## Host-neutral instruction core

**The instruction core is host-neutral, and the three verbs are its whole host vocabulary (D5).**
`instructions/*.md` name *spawn subagents* / *ask the user* / *run detached* and no host tool —
no `Task`, no `AskUserQuestion`, no `run_in_background`, no
`${CLAUDE_SKILL_DIR}` — and every command in them is the one invariant prefix
`uv run {UV_RUN_FLAGS} --project "{SKILL_DIR}" -- python "{SKILL_DIR}/multi_llm.py" <mode-or-util>` (two tokens, both substituted at read time — the second from
`unattended.uv_run_args()`, because a hand-typed `uv run --project` in this corpus *is* a dispatch
without `--frozen --no-dev`).

`tests/test_instruction_portability.py` owns both halves: a forbidden-token lint (unambiguous tokens anywhere,
`Task`/`Read`/`Write`/… only in their tool spellings,
per-file allowlist entries asserted to still match) and the per-file `{SKILL_DIR}`
manifest, whose one non-zero `${CLAUDE_SKILL_DIR}` row is `SKILL.md` — the Claude Code
shim, where the host expands the variable at load time. Each verb is resolved to exactly one
concrete mechanism *in the shim*; the matrix is never on an agent read path.

**The way back is `legacy/`**: the frozen pre-refactor tree, served instead of the
rewritten core when `MULTI_LLM_LEGACY_INSTRUCTIONS=1` (read path only, Claude Code only, unmaintained — see
`legacy/README.md`), so the rewrite can be backed out without reverting the dispatcher, the
executor or the unattended work with it. The regression guard is `tests/test_instruction_refactor_guard.py`: a
normalized golden bundle of one fully mocked workflow run over `tests/fixtures/e2e/plans/auth-feature.md` ×
`full_workflow.yaml`, captured on the `phase3-pre-refactor` tag and reproduced with a **zero**
diff outside `tests/golden/expected_changes.yaml`.

**No spelled invocation in the corpus** (plans/next-phase-command.md T6): the lint also forbids, as
a plain substring on every line of `instructions/*.md` and `references/*.md`
(`references/host-capabilities.md` exempt, fenced blocks and examples included), every host's
`next_command` prefix — read through `host_capabilities.next_command_prefix`, never typed, so a new route is
covered with no test edit — and a hit's message names the replacement: end the report with
`user_next_step.command`.

Its exceptions are **exact stripped lines**, not files, each with a rationale and one of three
documentary kinds a shape test checks — `usage` (inside the first fenced block under
the file's `## Usage` heading), `user-input` (the line starts with
`User: `) and `args-example` (an `ask.md` parsing example, or
the `If models provided in args (e.g., ` line) — and there is no fourth kind, so an actionable next-step or
recovery line cannot be kept by calling it documentary.

Every actionable command the cores used to show is rewritten: next-step hints point at
`references/summary-format.md`'s `### Next Steps` rule, the missing-tasks stop and the
`--force` hint name phases, and the Workflow Comparison blocks are numbered lists of
phase names.

## Install and uninstall

**`install-host`/`uninstall-host` are `utils/host_install.py`, and the installed shim
is *output* (D7).** `install-host <host> [--global|--project] [--check] [--all] [--force] [--project-root PATH]` renders `hosts/<host>/shim.md.tmpl` — the three verb slots
from the capability data, `{{uv_missing_file}}`/`{{uv_missing_text}}` and every
`hosts/snippets/*.md` slot, then `{SKILL_DIR}` and the baked `--host <id>` — and
writes it as a **rendered copy, never a symlink or junction** (which is what keeps
`SeCreateSymbolicLinkPrivilege` out of the Windows story): temp file *in the target directory* →
`fsync` → `os.replace`, `newline="\n"` so the bytes are identical
on every platform.

**Verification is part of the write**: after the replace the same `--check` runs, and a
shape failure (rows 1–4) is a *failed install* — previous bytes restored, or nothing left behind —
while a row-5 *environment* miss keeps the byte-correct shim and exits 3, because deleting a correct
file cannot make a missing CLI appear.

`--check` asserts exactly five things, each with its own exit-3 message and the
one-line repair (`install-host <host>`, the same string D1c's moved-skill-dir message and the
host-install docs print): shim **present** (every copy reported in the host's discovery order,
naming the one that loads), **baked prefix current** (resolves, and equals the running skill dir —
D1c), **rendered shape valid** (frontmatter where the format requires one, exactly one dispatcher
block, the uv preflight ahead of the first `uv run`, no surviving placeholder or
unfilled slot, `--yes` baked in, the three verb sentences byte-identical to the data,
the baked `--host` a key in it), **generation current** (`shim_version`, the
shape counter, and the `template_sha256` the header records — a template body that has moved
since the shim was rendered fails here, and it is the only row that catches a prose-only template
edit; `SHIM_VERSION` moves only when a shim's rendered shape changes — a new step or slot, a
changed command form, or `render()` itself — so after a prose-only edit update the
pinned digest in `tests/test_install_host.py` and leave the counter, 4, alone), and **environment**
(`uv` via `unattended.find_uv` — D6c layer 1 — plus four facts about the runner
the *executor* would really dispatch, not the capability entry's default read on its own:
`host_install.configured_runner` resolves it through `get_subagent_runner` with the same host id and project
anchor as a run and reports the layer that supplied it (D3o), then **present** through
`init_config`'s own `get_provider(...).is_available()` detection, **authenticated** and **version**
through the executor's own `preflight_runner` (installed is not authenticated — D3c), and the
host's `nested_exec` legs, `denied` on any of which is a failure; plus Git
Bash on Windows).

Every rendered shim carries a one-line `multi_llm_shim` header (host id,
`shim_version`, template hash, hash of the body as written) in the file's own comment
syntax, placed *after* the frontmatter; it is what makes re-install **idempotent** (byte-identical,
nothing rewritten), what makes a **local edit** exit 3 rather than a silent overwrite
(`--force` keeps it as `<shim>.local.bak`), what makes a target owned by a
*different* host id a reported conflict, and what `uninstall-host <host> [--global|--project|--all] [--dry-run] [--force]` identifies its candidates
by — **never by filename**, never outside that host's own `install_paths`, and never a
compatibility location.

**Nothing is per-host**: the target file is `install_paths[scope][0]` (expanded:
`$HOME`/`~`/`%USERPROFILE%`, `$XDG_CONFIG_HOME` →
`<home>/.config`, a relative entry against the project root) joined with
`basename(shim.installed)`, precedence is project→global and preferred→compat, and the reload line comes
from `reload`. `install-host claude-code` is `--check`-only
(`installable: false`, rows 3–5 only); `link-skill.sh`/`sync-skill.sh` are
Claude-Code-only dev tooling and **not** an install backend. Both modes carry `requires_uv=False`
in `modes.py` — `--check` *is* the uv-bootstrap's install-time layer and
must report a miss, not die on the layer-3 backstop.

`tests/test_install_host.py` drives all of it against a fixture skill dir **containing a space** with
`install_paths` redirected to a temp `$HOME`/`%USERPROFILE%`, and
re-runs `tests/test_shim_correctness.py`'s assertions over the *installed* file so CI's notion of a valid shim
and the installer's cannot diverge. `init` is a thin wrapper forwarding to
`init_config.main` (narrative in `instructions/`).

## Package distribution

**`packaging/opencode/` is a second distribution of the same tree, and it is not a second host.**
It builds `opencode-multi-llm`, an npm package carrying `skills/multi-llm/` beside a v1
OpenCode plugin: the `config` hook injects `/multi-llm` from a pack-time
render of `hosts/opencode/plugin-command.md.tmpl`, and the `tool` hook registers a model-callable
`multi_llm` whose `execute` composes the dispatcher argv, refuses the flags
it owns, runs the uv preflight and supervises a `--detach`ed run to a terminal result.

The point is that **the plugin knows where it is**: `import.meta.dirname` resolves the skill dir at
every startup, so there is no baked absolute prefix to go stale, no `--check` row 2 to
fail and no `install-host --all` owed after an update — and the baked `--host opencode --yes --exec-subagents` stop
being prose a model may improvise past and become code. `build.py` stages by
**allowlist** (parsed from `tests/container/context_manifest.txt`, never re-typed), refuses an unexpected file,
asserts a required inventory, sweeps for credentials and **never invokes npm**;
`packaging/common/payload.py` is that machinery, shared so a second host's packaging plan is a thin driver
over it.

The JS holds no mode word, alias, summary, flag name, exit code or message of its own — everything
the model reads is generated into `dist/cli-surface.json` from
`modes.py`/`cli_common.py`, and `tests/test_opencode_package.py` greps the committed
`src/index.mjs` for any of it. `hosts/opencode/shim.md.tmpl` is untouched and stays the file route's
shim; the two routes are not mutually exclusive, and when both are installed the plugin's
`config` hook overwrites the file entry wholesale — silently, as far as OpenCode is
concerned — which is what `install-host --check`'s route-detection note exists to say.

**The required inventory is the static list plus every `target_module` in
`modes.py`**: `build.py`'s `target_module_paths` maps each dotted target to
`x/y.py` (or `x/y/__init__.py`) and passes them as `payload.stage(required_extra=…)` and to
`--scan-tarball`. Staging takes its candidates from `git ls-files`, so an untracked
runtime module is never staged and `ALLOW_DIRTY=1` cannot see the gap —
`utils/doctor.py` shipped that way once (2026-09-11), the modes table pointing at a module the
tarball did not carry, `ModuleNotFoundError` on `doctor` from the staged tree.

**And the plugin factory consumes its input's `client`**: `execute`
resolves the session's current model from the newest assistant message (`resolveSessionModel`:
`client.session.messages`, `providerID/modelID` plus `variant` as an
`:effort` suffix, a bounded budget, never throws) and exports it as
`env[surface.dispatcher.session_model_env]` for the launcher and every poll (the detached worker inherits it through
`child_env`), echoing `session_model` on the result the way `anchor`
is echoed.

**The `instructions` attachment is gated on the outcome, not only on the mode**
(`rowNeedsInstructions(row, outcome)`: `supports_detach` **and** an unfinished outcome — a non-null
`next`, or a state that is not `done`); reading
`row.supports_detach` alone shipped the whole rendered runbook with every finished run, measured
2026-09-16 at 22 860 of a 25 829-char result, fifteen of its lines `uv run …` commands
the same result's message forbids the model to run.

And `normalizePlan` strips **one** leading `@` — this host's
file-attachment sigil, which is how a user names a file here and which reaches the model verbatim —
so `@plans/x.md` no longer resolves to `<workspace>/@plans/x.md` and dies on "Plan file not
found"; the `plan` property description says so too. **And the mid-run relay reports
itself in numbers**: `relays` (`{attempted, delivered, rejected, unavailable}`) is attached to the tool result
by `execute` — not by the dispatcher — because a silent relay failure was
undiagnosable.

`client.tui.showToast` persists nothing in `opencode.db` and `warn()` writes
to a stderr `opencode.log` does not capture, so "the relay never fired", "every POST was
rejected" and "it fired and the user missed it" left identical evidence: none. Counts only, no
authored text, under the same rule — what it relays is the dispatcher's `progress_lines`,
passed through verbatim. `attempted` is exact when the result is composed and
`unavailable` counts the relays a host carrying no `showToast` route could never
display; `delivered`/`rejected` settle asynchronously — nothing awaits the
SDK call, which is what fire-and-forget costs — so the last relay of a call may land in neither.

Read them as a floor: `attempted - delivered - rejected <= 1`. **Both tool routes lift `user_next_step` and
never follow it** (plans/next-phase-command.md §2.4): `terminalResult` here and the codex MCP
server's `terminal_result` move the redacted result's `user_next_step` to the top level
of the tool result on the exit-0 and exit-2 branches (moved, not copied, since fix round 26: the
section is in the result once) (the completed form included; a poll's exit 2 means the run is still
moving and never carries it), `isMechanical`/`is_mechanical` stay keyed on
`next` alone, and `_tool_description` appends `modes.TOOL_USER_NEXT_STEP_LINE` after the
outcomes line on both hosts, so the model reads before any result that the key is the user's to type
and never a call.

The report ends with `command` **and** each `optional_before` command, each in its
own fenced block, verbatim: every shim says so, as do `modes.TOOL_USER_NEXT_STEP_LINE` and the instructions
that list optional phases (before 2026-10-01 only `references/summary-format.md` did, and T10 measured the
OpenCode file-shim route writing the optional command inline on two models; after the wording fix it
passed, run 20261001T130949Z). Each exposed mode row also carries `optional_why`.

Surface schema version 5 (`dispatcher.route_env` and the per-row `optional_why`); a v5
server refuses a v4 surface, or a v5 one without `route_env`, through the existing
degraded path rather than spawning without `MULTI_LLM_ROUTE`, which would print the file-shim
spelling on a tool route.

## Detecting the active route

**`install-host --check` answers a question it never used to ask: *which route is live?***
`host_install.package_route()` reads the merged host config **first** and stats exactly one computed cache
path second (`<cache>/packages/<normalized spec>/node_modules/<name>` — the innermost component is load-bearing, since the spec
directory is created before resolution is attempted and survives every failure as an empty
directory). Config-first is deliberate: an extracted cache tree is not evidence of an install —
uninstalling leaves it behind forever, every pin ever used stays beside the current one — and since
the whole purpose is to warn about *shadowing*, a false shadowing warning is worse than none.

Row 1 becomes route-aware rather than skipped (its question is "is a route present and loadable?",
not "is there a file?"), so a package user is never told to run `install-host` and create
the duplicate; a declared-but-unresolved route **fails** row 1 with the package route's own repair.
The **note** fires ungated by `path`, so `install-host --all` — the command the
docs prescribe after every plugin update — says so at the moment it recreates the shadowed shim. It
stays a note and never a row: `install()`'s verify path rolls back on a row failure, so a
promoted note would roll back an install on any machine carrying a package route.

## Generated host documentation

**`docs/hosts.md`'s "Hosting multi-llm in other harnesses" section is generated** by
`hosts/render_capabilities.py` between two markers, from the same `hosts/capabilities.yaml` as
`references/host-capabilities.md` — per-host quickstarts, the capability matrix, the install-path matrix and
the after-a-plugin-update line. `python hosts/render_capabilities.py` writes both files; `--check`
byte-compares both and is the CI staleness guard, so a hand-maintained copy of either is a failure.

## OpenCode version updates

**opencode supports its newest release only, and moving it is one command.** `OpenCodeProvider.min_version`
equals `verified_version`, the newest `opencode-ai` on npm, and `engines.opencode`,
the package README, the container's `opencode-ai` pin (`package.json`,
`package-lock.json`, `cli-versions.lock`) and the mock's `--version` name the same
release (`tests/test_opencode_version_pins.py` fails otherwise). Below it the executor refuses (exit 3, naming
`opencode upgrade <pin>`); above it one line warns.

`make opencode-latest-check` exits 1 when npm has a newer release; `make opencode-bump VERSION=<x.y.z>`
(`packaging/opencode/bump.py`) re-captures `tests/fixtures/opencode/capture/<v>/`, `auth_list.<v>.json` and
`debug_config.<v>.json` offline from that release ($0, a localhost stub), deletes the previous
captures and rewrites every pin. Tests reach the captures through `tests/opencode_fixtures.py`, so a red
parser test after a bump is the release's change to review, never a reason to re-pin; then
`make e2e-image` and re-seed the opencode row of `tests/container/cli-surface.yaml`.
