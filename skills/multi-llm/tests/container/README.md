# Containerized harness E2E — real CLIs, stubbed backends, both roles

One developer-invoked target that builds an image with the **real** coding-agent
CLIs installed, redirects their model backends at a local stub, runs the workflow,
and asserts the artifacts.

It exists to catch the two drift classes nothing else in this repo can see:

1. **A provider CLI changed under us** — a renamed flag, a dropped subcommand, a
   restructured JSON event stream. Every other provider test asserts our parser
   against fixtures *we captured by hand*, so a real release can break every user
   while CI stays green.
2. **A host no longer discovers or dispatches what `install-host` wrote** — the
   shim lands in a directory the host stopped scanning, its bash step trips a
   changed sandbox default, its detached run outlives a changed timeout.

Design: `plans/containerized-harness-e2e.md` — a local-only working note, kept out
of the published repo, so a fresh clone will not have it. This file is a *reader* for the
`make` targets, not their origin — **it deliberately restates no `docker` flag.**
Build flags live in the repo-root `Makefile`; run flags live in `posture.yaml`.
A flag that exists only here is a bug in that split.

---

## The two lanes

|  | **Provider lane** | **Host lane** |
| --- | --- | --- |
| Question | does the real CLI still accept our argv and emit a stream we parse? | does the real host still discover, load, run and survive our shim? |
| Backend | local stub, `$0` | real API via one `OPENROUTER_API_KEY` |
| Secrets | **none** | one, dedicated, spend-capped |
| Network | **`--network none`** | bridge (detection, not prevention) — but `--creds none` discovery rows run `--network none` too: no credential, no route |
| Determinism | total | none |
| Assertion | normalized golden bundle, zero diff | receipt + artifact presence |

**The provider lane runs green with zero secrets and zero network at *run* time.**
The build cannot be offline — it pulls a base image and installs CLIs — but every
input is fetched by digest or checksum, and every row then runs under
`--network none` with no credential variable passed into the container at all.
A contributor with no API key can run the half of the suite that gates correctness.

---

## Targets

```
make e2e-container                 pinned build; provider lane
make e2e-container-host            pinned build; host lane (needs OPENROUTER_API_KEY)
make e2e-container-host-discovery  host lane, --creds none tier: no key, no network
make e2e-container-host-invocation host lane, --creds none: type the documented /multi-llm string
make e2e-container-latest          CLIs @latest; provider lane only; credential-free
make e2e-container-latest-host     CLIs @latest; host lane; guarded, opt-in only
make e2e-version-delta             print the pinned → installed versions of the @latest image
make e2e-lock-update               move the pins the @latest image proves moved, then prove it
make e2e-verify-record REC=<path>  re-hash a runs/ bundle against a committed record
make e2e-clean                     prune the ephemeral runs/ bundles
```

All typed at the checkout root, no `-C`: the build context *is* the repo root.

---

## What the provider lane covers, measured

Phase 1 pointed every configured CLI at a local three-surface stub and recorded
what it did on the wire; the four CLIs `harness-expansion.md` added afterwards
were probed the same way. **15 of the 17 configured CLIs are redirectable**
(`redirect.yaml` carries the full row per CLI, including the exact mechanism and
the HTTP footprint):

| CLI | mechanism | surface | turns | one generate call |
| --- | --- | --- | --- | --- |
| `codex` | `$CODEX_HOME/config.toml` provider block | responses | 1 | 0.47 s |
| `claude-code` | `ANTHROPIC_BASE_URL` + a key variable | anthropic | 2 | 1.35 s |
| `gemini` | `GOOGLE_GEMINI_BASE_URL` (the bundled SDK's, not the CLI's) | gemini | 1 | 2.26 s |
| `opencode` | `opencode.json` custom provider block | chat | 2 | 1.25 s |
| `aider` | `OPENAI_API_BASE` + an `openai/*` litellm spec | chat (**non-streaming**) | 1 | 2.87 s |
| `goose` | `GOOSE_PROVIDER` + `OPENAI_BASE_URL` | chat | 3 | 0.16 s |
| `cline` | `~/.cline/.../providers.json` (no env var exists) | chat | 1 | 1.45 s |
| `kilocode` | `~/.config/kilo/kilo.jsonc` provider block | chat | 2 | 4.31 s |
| `pi` | `~/.pi/agent/models.json` provider block | chat | 1 | 0.39 s |
| `grok` | `GROK_MODELS_BASE_URL` + `XAI_API_KEY` | chat | 1 | 0.27 s |
| `muse` | `muse exec --base-url` (**argv only**) | responses | 17 | 0.91 s |
| `qwen-code` | the `OPENAI_BASE_URL`/`OPENAI_API_KEY`/`OPENAI_MODEL` trio | chat | 1 | not timed |
| `copilot` | `COPILOT_PROVIDER_BASE_URL` — the BYOK activation switch | chat | 1 | not timed |
| `kimi` | the `KIMI_MODEL_*` env family | chat | 1 | not timed |
| `vibe` | a seeded `config.toml` `[[providers]]` block (no env var exists) | chat | 1 | not timed |

**The last four rows are later arrivals, and the table says so in one column.**
Their mechanism, surface and turn count are measured exactly like the others,
but none carries a `turn_s` or a `chain_s`, so every *timing* figure in this
file is over the original eleven and says which.

Not one of Phase 1's `documented` guesses turned out to be wrong, and both of its
surprises went our way: `pi` (the one `unknown` row) and `grok` (predicted
unredirectable) both verified.

**"One non-tool message ends the run" is true, but the turn count is not 1.**
`claude-code`, `opencode`, `kilocode`, `goose` **and `grok`** each fire a
**session-title** completion before the real turn, on the *same model*, so a stub
keyed on `(model, surface)` cannot tell the two apart. Four of the five are
tool-free; **grok's carries tools**, so `has_tools` separates four of them and not
the fifth — the discriminator that actually works on all five is the title
generator's own text, which is what
`tests/fixtures/e2e/scenarios/container_full_chain.yaml` anchors on. `muse` is the
outlier at 17 requests: four parallel background "reminder observer" tasks, not
retries, and those need answering too.

A scenario with `on_unmatched: fail` needs a rule for every one of those, **and a
negative lookahead on every product rule**, because a title call *quotes the
prompt it is titling* — goose's carries ours in full. Without the lookahead the
plan-review rule answers goose's title request with a suggestion list.

**Nothing may be anchored with `^` against our own prompt text.** Measured on the
wire: every redirected CLI wraps our prompt inside its own system prompt, so the
text the stub extracts starts with *the CLI's* preamble. A pattern anchored on our
first line matches on the fake-binary lane and matches nothing at all here.

**Uncovered, by name rather than by silence:**

- **`cursor-agent`** — the redirect works perfectly (`CURSOR_API_ENDPOINT` sends
  100% of traffic to a stub), but the backend is Cursor's own Connect-RPC:
  `content-type: application/proto` over `POST /aiserver.v1.<Service>/<Method>`.
  Serving it means reimplementing Cursor's protobuf schema, which is not a stub.
  It stays on the fake-binary lane, where `mock_llm.py` already has an identity
  for it.
- **`agy`** — `CLOUD_CODE_URL` is a working override and the protocol is JSON, but
  agy never reaches a model surface: it fails at Google's *eligibility* gate on
  `:loadCodeAssist`, and **its auth resolves before the first RPC**. Re-measured
  2026-08-19 with `CLOUD_CODE_URL`, `GEMINI_API_KEY`, `GOOGLE_API_KEY` and
  `GOOGLE_GEMINI_BASE_URL` all pointed at a live stub: a byte-identical failure
  envelope and **zero requests in the stub's log**. No environment variable
  selects an API-key path a stub could answer, so this row is permanently
  `lane: fake` — settled, not a gap waiting on effort.

  > **Two earlier claims here were wrong, and `redirect.yaml`'s agy row carries
  > the retraction.** (1) *"Its auth is in the OS keyring, not `$HOME`, so there
  > is nothing to seed"* holds only where **D-Bus is reachable**. Headless —
  > which is every container — agy logs `Failed to load token from keyring,
  > falling back to file` and opens
  > `$HOME/.gemini/antigravity-cli/antigravity-oauth-token`, which is seedable
  > exactly the way cursor-agent's `auth.json` is. It still matters on a laptop:
  > a probe run where D-Bus *is* reachable silently uses the developer's real
  > Google token, so break the bus deliberately
  > (`DBUS_SESSION_BUS_ADDRESS=unix:path=/nonexistent-bus`). (2) *"It is the one
  > provider still owed the five mock parts"* — it is not, and has not been since
  > the identity landed: `mock_llm.py` carries all five (the name in
  > `get_provider_name`'s `known_providers` and in `SkillRunner.PROVIDERS`, an
  > `agy` branch in `parse_provider_args`, `format_agy_output`, a
  > `VERSION_STRINGS` entry, and part 5's *empty* case — `AgyProvider` declares
  > no `auth_check`, so the absence of an `AUTH_PROBE_ARGS` row is the correct
  > answer and `test_fake_lane_roster.py` asserts exactly that).

  agy is also **in the image** (1.1.15) and has a fixture set
  (`tests/fixtures/agy/`), neither of which contradicts the above: it is
  installed for the `cli-surface` help-drift probe, which needs the binary and
  no backend, and the fixture set is **one graded row and eleven written
  reasons** — every unreachable cell recorded as a measurement rather than left
  pending.

### Two modules, two questions

| module | row | question |
| --- | --- | --- |
| `tests/test_container_provider_lane.py` | one phase (`review-plan --quick`) per CLI | does the real binary still accept our argv and emit a stream we parse? |
| `tests/test_container_full_chain.py` | the whole chain per CLI | does every phase's parser, validator, batcher and scheduler still agree with a real CLI, end to end, in one process? |

The chain row is `full --quick --yes --exec-subagents`, then a standalone `ask`
and a `status` read-back. Measured, it is **25 model turns**: 1 review-plan
fan-out, 3 suggestion-validation batches, 4 apply-suggestions batches, 1 task
generation, 1 review-tasks fan-out, 2 task-suggestion-validation batches, 3
apply-task-suggestions batches, 3 implement batches, 1 review-code fan-out, 2
issue-validation batches, 3 apply-code-fixes batches, 1 `ask`. No consolidation,
salvage, revalidation or human-decision turn is reached when every validation
answers `valid`.

Those 25 are the *product's* turns. What a row's stub log actually shows depends
on the CLI's own bookkeeping. Measured across the **eleven Phase-1 chain rows** —
410 model turns in total, **zero** `unmatched` and **zero** `unhandled` records
on every row (`qwen-code`, `copilot`, `kimi` and `vibe` have no recorded chain
turn count, so they are absent here rather than folded into the total):

| turns | rows | why |
| --- | --- | --- |
| 25 | `codex`, `cline`, `gemini`, `pi` | no side turns at all |
| 26 | `aider` | one extra |
| 29 | `kilocode` | a title call on the four turns it runs itself |
| 33 | `goose`, `grok` | ditto, plus grok's `/v1/responses` side call |
| 50 | `claude-code`, `opencode` | both are their *own* write runner, so every one of the 25 execs fires a title call |
| 89 | `muse` | ~2 background reminder observers per exec |

Three things about that row are not optional:

- **`--exec-subagents`, and `--yes` does not imply it.** `full` runs every phase
  in one process, so nothing sits between a phase and the next to execute the
  batches a phase emits. Measured, this lane stops **one phase earlier** than the
  design predicted: not at the first apply phase with
  `[WRITE_PHASE_NOT_EXECUTED]`, but at **review-plan**, which prints
  `[VALIDATION_BATCHES_PENDING]`, exits 0 without recording itself completed, and
  is refused by `_verify` ("exited 0 but state.json records no completion"). Exit
  3, **zero** phases recorded. A host with a native subagent channel gets as far
  as the write guard; the container is the whole invoker, so it does not. The
  negative control in that module asserts the measured behaviour.
- **Both driver keys.** `providers.tmpl.yaml` renders `driver.subagent_runner`
  *and* `driver.subagent_runner_cheap`; an unset cheap key keeps base's
  `claude-code:haiku` and adds a second binary to every row.
- **Twelve of the fifteen CLIs declare no `agent-write` profile** — only
  `claude-code`, `codex` and `opencode` do — so their chain row's executor is a
  second binary: `codex`, named per row in `redirect.yaml`'s
  `chain_write_runner` and carried as that row's one `allowed_helper`. A plain-CI
  test cross-checks the key against the adapters in both directions.

**Every write batch on this lane must record `clean-failure` with zero files
touched.** The stub emits no tool call, so no redirected CLI here can move a
byte; the four write phases complete on the documented no-op and the row asserts
that in both directions — `applied` is a bug in the rig, and a phase resolving to
the "nothing to apply" skip is a vacuous green.

### The golden gate, and how to move it

Those structural assertions all stay green through **output-shape drift**: a CLI
upgrade that changes what we parse out of its stream — half the suggestions, a
lost importance, a group that stopped grouping — still completes every phase,
still records `clean-failure` on every batch, and still moves no bytes. So each
chain row is additionally diffed, at a threshold of **zero**, against a normalized
bundle captured per CLI:

| | |
| --- | --- |
| Baseline | `tests/golden/container/<cli>/bundle.json` (**tracked**) |
| Allowlist | `tests/golden/container/expected_changes.yaml` — ships **empty** |
| Capture | `tests/container/seed_container_golden.py`, on the host |
| Assertion | `tests/test_container_full_chain.py`, which only ever *reads* |
| Machinery | `tests/golden/container_bundle.py` over the shared `bundle_compare.py` |

Compared: the phase JSONs, every `.status.json` step sequence and shape,
`state.json` and the structure of the metrics it carries, the relative-path set
the run created, the derived scalars, the `[MARKER]` sequence, each dispatcher
invocation's `MULTI_LLM_RESULT` line and exit code, and which workspace files
outside the output dir moved. Timestamps, durations, usage, pids and run ids are
folded to type tokens; prompts and any text over 200 chars to a digest (a change
is still a diff, the review is not two kilobytes of prose); `runner_versions` to a
type token, because a moved CLI pin is C9's subject and must not turn all fifteen
baselines red at once.

Re-capture is a **deliberate, reviewed** step — there is no `--update` flag on the
assertion side, because a guard that can rewrite its own baseline is not a guard:

```bash
make e2e-container                     # a green chain run leaves the evidence
uv run --project skills/multi-llm -- python \
  skills/multi-llm/tests/container/seed_container_golden.py --cli codex
git diff skills/multi-llm/tests/golden/container/codex/   # read every line
```

The seeder refuses a row that hit its cap, exited non-zero, ran fewer than its
three invocations or did not record every chain phase completed: a half-run frozen
as "expected" makes every later comparison agree with a broken chain. A CLI that
joins the provider lane without a baseline fails in **plain CI**
(`tests/test_golden_container_bundle.py`), which also asserts the gate is still
wired to the chain module — the failure mode that left this gate documented but
unrun for a release.

### How long it actually takes

**Not "seconds".** A generate call across the verified set is ~11 s of CLI time in
total, but a row is a *chain*, and each phase pays its CLI's cold start again.

**Measured, one green run per CLI** (`chain_s` in `redirect.yaml`, wall clock
including container start and the workspace seed): 20 s (`claude-code`) to 119 s
(`codex`), **17 m 42 s** for the eleven chains that carry a `chain_s`, plus the
no-`--exec-subagents` control. The matrix is **fifteen** chain rows: `qwen-code`,
`copilot`, `kimi` and `vibe` each declare a `chain_row_timeout_s` and ship a
golden bundle but record no `chain_s`, so budget for them and do not read 17 m 42 s
as the whole lane. The spread is not about the CLI under test — it is about
**whose binary the executor dispatches**: a row that is its own `agent-write`
runner (`claude-code` 20 s, `opencode` 51 s) pays one cheap startup per subagent,
while the twelve rows that borrow `codex` pay codex's `exec` startup 21 times.

Two caps keep that bounded rather than open-ended:

- **per row** — `row_timeout_s` for a single-phase row (floor 300 s; `cline`'s is
  600 s because blocked egress costs it ~7.5 s on **every** invocation) and
  `chain_row_timeout_s` for a chain row (floor 600 s), both in that CLI's
  `redirect.yaml` row next to the `chain_s` they were derived from — so a row that
  doubles in runtime shows up as a diff rather than being absorbed.
- **per matrix** — `MLLM_E2E_MATRIX_TIMEOUT_S`, defaulted in the `Makefile` and
  enforced by `harness.py`'s `MatrixBudget`, which every row claims before its
  `docker run` is built. Once the lane's clock is spent the driver starts nothing
  further: it names which rows ran, which failed and which **never started**,
  writes that census to `runs/<utc>/matrix-census.json`, and the invocation exits
  non-zero. It never kills a row already in flight — that is the per-row cap's
  job, and it is the one that leaves readable evidence — so the overshoot is
  bounded by one row's own cap.

**A cap hit is a diagnosis, not a hang.** The driver classifies it from the row's
stub log before reporting:

| Evidence | Reported as | Reading |
| --- | --- | --- |
| turn count climbing at the cap | `cap-hit: tool-loop` | the CLI never accepted a tool-free answer as final — this row belongs on the fake lane, not on a raised cap |
| **zero** requests logged | `cap-hit: no-first-call` | it never reached the stub: a resolver or socket wait under `--network none`, an unredirected surface, or a prompt nobody is answering |
| requests stopped mid-chain | `cap-hit: stalled-after-<turn>` | names the last fixture that answered, which is where triage starts |

A capped row is **never** graded against a golden — a killed CLI has no parseable
output, so the cap can only ever cost coverage, never fabricate it.

---

## What the host lane covers

Four tiers, and only the first two need nothing:

**`--creds none` (free, and it runs today).** `make e2e-container-host-discovery`.
Stages the shim, crosses the `reload` boundary, asks the host what it can see, and
stops. No workflow row, no model call, no credential, `--network none` — and the
tier **asserts it spent nothing**, since "free" is its whole claim. It refuses to
start if any credential variable is present, so it cannot silently become a cheap
credentialed run.

Whether a host can be asked *without a model* is a measured, per-host fact:

| Host | Credential-free listing surface |
| --- | --- |
| `opencode` | **yes** — `opencode debug config` emits resolved configuration as JSON, and an installed command appears under its `command` key by name |
| `codex` | **none** — no listing subcommand exists, and `codex doctor` never names a staged skill |
| `claude-code` | **none** — `claude doctor` explicitly "reads settings files … without a trust prompt" and was the best candidate; it does not list skills either |

The two without a surface are reported `unavailable-without-credential` rather
than passed. **A green `install-host --check` is not discovery** — it reads our own
bytes at a path we chose — so it is never substituted for one.

That verdict is about *this* tier and nothing wider. Since 2026-08-19 the
**credentialed** rows record `listed-via-prompted-turn` for both hosts, from a
`discovery-prompted.json` the agent wrote carrying the argv it actually ran
(`multi_llm.py util instructions ask`) — evidence obtainable only by locating and
reading the staged shim. It costs **$0.00**, because it rides receipts the
`--ask` row already paid for. It is weaker than opencode's `listed` (one command,
not an enumeration) and much stronger than the only thing a credential-free run
can honestly say, which is why both facts are recorded rather than one replacing
the other.

> `opencode debug skill` is the trap, and it looks right: it lists *skills*, our
> shim installs as a *command*, and it exits 0 listing only a built-in. A green
> command that proves nothing.

**`--creds none`, the documented-invocation row (free, and it runs today).**
`make e2e-container-host-invocation`. The other half of the same question, and
the one nothing had ever asked: *does the string the documents tell a reader to
type actually dispatch?* Discovery answers "the host can see the command";
this answers "the published spelling reaches our dispatcher".

They are genuinely different, because the two installable hosts do not install
the same kind of object — codex gets a **skill** (`.codex/skills/multi-llm/SKILL.md`),
opencode gets a **command** (`.opencode/commands/multi-llm.md`) — so each does its
own resolution, namespacing and word-splitting. The credentialed rows do not
settle it either: they prompt with *"Use the multi-llm skill with exactly these
arguments: …"*, which proves the writer and says nothing about `/multi-llm`.

The row installs the shim, points the **host's own model** at the loopback stub,
opens a **fresh session in a pty** (both hosts are `reload: new-session`; one host
process per row satisfies that by construction), types the published string with a
per-run marker, and reads back the prompt the host composed through the stub's
opt-in fact probe — *names of patterns the rig wrote, never the prompt body*. A
host that resolved the command sends its own shim's text; one that did not sends
the characters the user typed. No credential is needed because what is under test
is argument delivery, not model output.

| Host | `/multi-llm …` typed into a fresh session | measured |
| --- | --- | --- |
| `opencode` 1.18.18 | **resolves** — shim body, `{SKILL_DIR}` substituted, `--host opencode --yes --exec-subagents`, and `--review-plan plans/<marker>.md` still adjacent | 2026-08-24 |
| `codex` 0.146.0 | **refused** — `• Unrecognized command '/multi-llm'.` and *zero* requests sent | 2026-08-24 |

Codex's shim is discovered — the skill is advertised to the model by name inside
`<skills_instructions>` with its `SKILL.md` locator — but a Codex *skill* is a
model-invoked instruction source, not a composer command; its slash surface lists
`/model`, `/ide`, `/permissions`, `/keymap`, `/vim`, `/experimental`, `/approve`,
`/memories` and no skill, and the deprecated `install_paths.legacy` prompts
directory does not rescue it either. **That makes it a documentation defect**, and
`test_the_documents_state_the_syntax_each_host_actually_takes` fails until
`README.md` and the sentence `render_capabilities.py` renders into `docs/hosts.md`
say so per host.

> One rig accommodation, recorded rather than buried: `posture.yaml` mounts /tmp
> `noexec`, and opencode's TUI is a bun binary that dlopen()s a native render
> library out of `$TMPDIR`. The row points `$TMPDIR` at the bundle bind — the one
> writable exec-capable path in this posture — and deletes it again afterwards. It
> changes where a temporary file lands, never what the host is asked.

**OpenRouter (default).** One `OPENROUTER_API_KEY`, three hosts. Cents for the 3×3
matrix — **measured, on the rows that have run**: $0.008 (`opencode --ask`),
$0.011 (`codex --ask`), $0.020 (`claude-code --ask`), $0.024 (`opencode
--review-plan`), $0.036 (`codex --review-plan`), $0.239 (`claude-code
--review-plan`). The last figure is the point of the whole cost section: on the
two executor hosts a row really is cheap, and on the reference host every
subagent batch is a real model call that never crosses `/opt/mllm/bin`.

### What a credentialed row actually proved, 2026-08-14

`--ask` passed on all three hosts: an agent read our shim, ran `util instructions
<mode>` out of it, ran the dispatcher, polled `status --view run`, and left a
terminal run state with artifacts on disk. `review-plan` passed on `codex`,
completed-but-off-shim on `opencode`, and stopped at `[VALIDATION_PENDING]` on
`claude-code`. Every cell, and every correction, is in `host_matrix.yaml` beside
the argv it applies to; five of those corrections are worth knowing before you
run this yourself:

- **`codex` refuses `wire_api = "chat"`** as of codex-cli 0.146.0 and exits 1
  with an empty stdout. `responses` is the value that works against OpenRouter.
- **`codex`'s own sandbox is bubblewrap**, and bwrap cannot create a user
  namespace under this lane's posture, so *every* command the agent runs fails
  before it starts. The host-lane launch cell is `--sandbox danger-full-access`:
  the container is already the sandbox, and a second one that cannot initialize
  only kills the row. This says nothing about the `agent-write` profile our own
  adapter builds, which the provider lane still asserts as `workspace-write`.
- **`codex` runs commands through `bash -lc`.** A login shell re-derives `PATH`
  from `/etc/profile`, so the image's `ENV PATH` was invisible to the agent and
  its `command -v uv` preflight correctly reported uv missing. The image now
  ships `/etc/profile.d/00-mllm-path.sh`, asserted at build time.
- **`claude-code` takes its permissions from `--allowedTools`, not from a
  synthesized `$HOME/.claude/settings.json`** — and one token per rule, because a
  comma-joined list silently drops every rule containing a space *and* the ones
  around it. The grant list is read out of `SKILL.md`'s own `allowed-tools`
  frontmatter, never restated here.
- **`opencode` gates the skill directory separately from `bash`.** `bash: allow`
  runs `uv run --project /opt/mllm/skill …` happily and then auto-rejects a
  *read* under the same tree (`permission requested: external_directory`), which
  ends the run with no dispatch. The config now grants that path explicitly.

**Re-run 2026-08-19** (`records/20260819T150929Z-*`, 13 rows, 21:51 wall,
**$0.0726** total): all three `review-plan` rows passed, and a `full` row on
`codex` and on `opencode` each completed **all eight phases** against a real
model in one process. Nothing above was fixed to make that happen — the recorded
cause of the two earlier `review-plan` outcomes is model-following variance on a
long instruction core — so read the greener run as *observed*, not as resolved,
and expect either shape. The costs also moved with the model: `claude-code
--review-plan` was **$0.017** on that run against the $0.239 recorded above, so
treat these figures as the range a real row has produced rather than as a quote.

**Where `MULTI_LLM_RESULT` landed in each host's stream** — the observation the
design wanted before deciding whether the receipt is the primary anchor or the
only one:

| Host | The inner result line, in the host's own event stream |
| --- | --- |
| `codex` | present, complete, **JSON-escaped inside a nested tool result** (`item.completed` → `command_execution` → `aggregated_output`); the agent's final message paraphrased it |
| `opencode` | same, one field along (`tool_use` → `part.state.output`); no `text` event carried it |
| `claude-code` | **absent.** `--output-format json` emits only the final result envelope — no per-turn tool results reach the harness at all. The single occurrence of the string in the whole stream is the agent's prose paraphrase |

So the receipt is not merely the better anchor; on the reference host it is the
**only** one.

One more thing a grader has to know: on a detaching row **one receipt is
structurally `result: null`** and it is not a finding. `spawn_detached` sends the
worker's stdio to `/dev/null` and the worker reopens `orchestrator-run.log`
itself, so the `uv` wrapper that exec'd it captures nothing. The payload is on
the launcher's receipt, the verdict is in the worker's log and `.status.json`.

**Subscription (documented fallback).** Narrow `:ro` mounts copied into a tmpfs
`$HOME`. `~/.claude.json` is **never** mounted — 87 KB of per-project history and
MCP definitions that routinely embed third-party keys — a minimal one is
synthesized instead.

> **Required, not recommended:** the host lane runs on a **dedicated OpenRouter key
> carrying a key-level spend limit.** `MLLM_E2E_BUDGET_USD` is a pre-flight
> *refusal*, not a ceiling: it is computed before a token is spent, so it cannot
> stop one oversized request, a CLI's internal retries, or the reference host's
> native subagent batches from overshooting mid-run. The only limit that actually
> holds is enforced by the party doing the billing.

### The one measured surprise worth knowing

**`opencode auth list` exits 0 when logged out**, printing "0 credentials", which
matches none of the adapter's `auth_error_patterns` — so `classify_auth_probe`
returns `True` and `install-host opencode` reports `auth=ok` on a completely fresh
`$HOME`. `codex` fails closed here (`Not logged in`, exit 1); opencode does not.
A negative auth row therefore cannot expect a named `not_authenticated` refusal on
opencode, and a green codex negative row proves nothing about it.

---

## Two artifact classes, two lifetimes

| | `runs/<utc>/<row-id>/` | `records/<utc>-<host>.json` |
| --- | --- | --- |
| What | the raw bundle: stub log, call log, receipts, exported phase dirs | the sanitized summary the driver derives from it |
| Git | **gitignored** (README kept) | **tracked**, committed with the promotion |
| Lifetime | ephemeral; delete any time | permanent, reviewable in the diff |
| Role | local diagnostics | the cited provenance |

**Nothing in the repo may cite a path under `runs/` as evidence.** A green host-lane
row is the only thing that promotes a `capabilities.yaml` field from `documented`
to `verified`, and evidence a reviewer cannot open at the promotion commit cannot
back a claim that ships to users. `make e2e-verify-record` re-hashes a bundle
against the record that cited it, and reports it as *missing* rather than
*mismatched* once you have pruned it — deleting `runs/` is a supported thing to do.

The harness **prints** the `capabilities.yaml` promotions a run justifies; it never
applies them. Promoting a capability changes which code path ships to users, so it
stays a human commit carrying the record and the edit in the same diff.

### "Green" is two verdicts, and a promotion needs both

The **container** exiting 0 and the **grader** (pytest, asserting over the bundle
that run left behind) passing are different facts, and rows really do produce one
without the other: a `codex-review-plan` row exited 0 having never detached its
worker — no `orchestrator-run.log`, no `.lock`, every run state `None` — and a
`claude-code-ask` row exited 0 with its host at `error_max_turns`. Both would once
have proposed capability edits, because the record was written the moment the row
finished, before any assertion had read it.

So each host row's record is written **twice**. `run_row` writes it as soon as the
row exits, carrying the container's verdict, no promotions at all, and a `reason`
saying the grader has not spoken yet — evidence survives an interrupted session,
but a provisional record can never propose an edit. `GradeLedger` then re-emits it
at session end with both verdicts, which is the only point a promotion can appear.
`promotions()` defaults its grader argument to `ungraded`, so "promote by
omission" is not a code path any caller can reach.

Two consequences worth knowing: a row nothing asserted about is `ungraded`, which
promotes nothing and says so (it is not a pass); and because the re-emission
re-digests the bundle, a grading test that writes into the bundle it grades —
`negative-auth-observed.json` — no longer turns `make e2e-verify-record` into a
false MISMATCH.

---

## Two operational rules, on you rather than on the code

- **The image is never pushed.** It is tagged `mllm-e2e:local` — no registry
  component, so a bare `docker push` fails by construction. `docker save` of a
  host-lane container is equivalent to exporting a credential.
- **The sample workspace has no git remote.** `seed_workspace.py` initializes with
  no `origin`, no credential helper, `GIT_TERMINAL_PROMPT=0`, and pinned author and
  committer identity *and* dates. Nothing inside the container can push anywhere,
  including the `agent-write` rows that are allowed to edit files.

---

## When the drift report is red

`make e2e-container-latest` installs the CLI layer at `@latest` — every other pin
held, so a red run points at the CLI that moved rather than at a base-image change
nobody chose — and prints which CLIs changed version, which options in
`cli-surface.yaml` the new version no longer offers, whose help text changed digest
without losing an option we need, and which parse assertions changed.

**The argv table is not in that list.** Both sides of it originate in this
repository, so it cannot move when upstream does: read it as coverage of our own
`build_command()`. The **surface probe** is what carries the upstream signal.

Its digest half is soft — a changed help text is a `SKIP`, never a failure — so it
is also **written down**: `runs/<utc>/cli-surface-drift.json`, one row per CLI with
the version the contract was captured at, the version the image really installed,
both digests and both help sizes. That file is the report to read when the skip
reasons scrolled past; `-rs` prints the same lines to the console.

A green latest run is followed by `make e2e-lock-update`, and its scope is decided
by **what that image can prove**, not by what would be convenient. It reads the
build's own version census, the `package-lock.json` that `npm install @latest`
resolved (exact versions plus an integrity hash for the whole transitive tree) and,
for the rows the Dockerfile installs as a plain file, the artifact bytes themselves
— then rewrites `cli-versions.lock`, `package.json`, `package-lock.json` and the
Dockerfile's artifact `ARG`s from them. What it cannot substantiate it **refuses by
name, with the command**: `goose`, whose tarball the image unpacks and discards, so
its checksum is not recoverable; `aider`, whose pin is a `--generate-hashes`
requirements file only a resolver can regenerate; and the npm layer as a whole
whenever the new tree adds a package upstream ships without an integrity hash,
since widening that trust boundary is a decision rather than bookkeeping. It then
**rebuilds the pinned image against what it wrote** — `npm ci` against the new
lockfile, every artifact checksum verified before it runs, the CLI rows enforced by
`verify_versions.py` — and grades its own record against that image's census. A
failure anywhere after the write prints the exact `git checkout --` that reverts
it. `cli-surface.yaml` is deliberately left alone: a changed help digest is a
deprecation notice to read, and its re-seed command is printed rather than run.

A red one names the CLI and the row, which is the bug report, pre-written.

`make e2e-container-latest-host` is the only place an unpinned image is allowed
near a key, and it **refuses to start** when a credential is present unless
`MLLM_ALLOW_UNPINNED_HOST_LANE=1` is set deliberately — never from a shell profile.

---

## Running without Docker

Three mechanisms, and the middle one is the easy one to forget:

1. `pyproject.toml` registers the `container` marker and sets
   `addopts = -m 'not container'`, so a bare `pytest` stays green.
2. `.github/workflows/test.yml` says `-m "not live and not container"` **itself** —
   an explicit `-m` on the command line *replaces* `addopts` rather than composing
   with it, so without this a container-marked file would be collected and run on
   every PR, on runners with no daemon.
3. Every container-marked module carries a module-level `skipif` that probes for a
   reachable daemon and skips with a named reason.

Podman is **out of scope for v1** — not blocked, just unverified. Two things would
need work: the `uid=` in the tmpfs `$HOME` mode string resolves to a different uid
inside a rootless user namespace, and the posture test's host-side half reads
`docker inspect`'s `HostConfig` shape.
