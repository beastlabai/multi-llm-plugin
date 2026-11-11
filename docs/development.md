# Development

Running the test suite, the containerized end-to-end harness, and the plugin
validator. Back to the [README](../README.md).

The skill's source and test suite live under `skills/multi-llm/`. Every command
below is typed at the checkout root.

For architecture, shared invariants, and subsystem ownership, start with the
[skill development guide](../skills/multi-llm/AGENTS.md). Its topic index links
to the detailed contracts and regression history under
[`docs/development/`](development/); read the references relevant to your change.

---

## Where things live

The workflow contract lives in the host-neutral instruction core under [`skills/multi-llm/instructions/`](../skills/multi-llm/instructions/); [`skills/multi-llm/SKILL.md`](../skills/multi-llm/SKILL.md) is the Claude Code entry shim over it, [`docs/hosts.md`](hosts.md) and [`skills/multi-llm/references/host-capabilities.md`](../skills/multi-llm/references/host-capabilities.md) are the generated host documents, and [`skills/multi-llm/AGENTS.md`](../skills/multi-llm/AGENTS.md) has the architecture notes.

**Relocatable paths.** The skill references its own bundled files through `${CLAUDE_SKILL_DIR}`, so it works wherever Claude Code installs the plugin; a shim installed into another host by `install-host` bakes the same directory in, which is why a plugin update needs one `install-host --all`. The Codex plugin and the OpenCode npm package carry their own copy of the skill and need no such step.

---

## Test suite

```bash
# The ordinary CI selection, in parallel (about 3 min on 16 cores; N=8 to share the machine)
make test

# The same selection one test at a time, as the gating CI job runs it (about 30 min)
make test-serial

# A bare run (the addopts default below; includes `live` rows)
uv run --project skills/multi-llm -- pytest

# Run a single test file
uv run --project skills/multi-llm -- pytest skills/multi-llm/tests/test_filtering.py -v

# Robustness (error-handling) tests
uv run --project skills/multi-llm -- pytest -m robustness
```

Three marker groups are excluded from a bare run: `pyproject.toml` sets
`addopts = -m 'not container and not package and not codex_jail'`. Tests marked
`container` need a Docker daemon (each container-marked module also carries a
daemon probe that skips with a named reason), `package` rows need a packaging
artifact plain CI does not build, and `codex_jail` drives the developer's own
`codex` binary. Tests marked `live` require provider CLIs and real API access;
they are not in that default, so CI passes `-m "not live and not container and
not package and not codex_jail"` itself.

To find the tests for the module you are editing, run
`make test-for MODULE=utils/foo.py` (or read
[`skills/multi-llm/tests/MODULE_TESTS.md`](../skills/multi-llm/tests/MODULE_TESTS.md)).
It is an advisory import map and never replaces the CI selection; see
[`skills/multi-llm/tests/README.md`](../skills/multi-llm/tests/README.md).

### Keeping the suite cheap

The suite runs on three CI operating systems, and on Windows every process start
and every small file costs several times what it costs on Linux: a test file that
builds its own git repository runs 12 to 25 times slower there than on Ubuntu,
and one that drives the whole implement step against mock CLIs about 8 times.
Thousands of millisecond tests barely register; per-test setup that starts
processes is what fills the Windows leg. So, when adding a test:

- **Prefer a row, or an assertion on a run that already exists.** A regression
  from a live run belongs as a row in a table-driven test of the function that
  produces the text or state, or as one more assertion on a compatible existing
  run. Add a new end-to-end run only for another entry point or state transition.
- **Use an in-process seam for logic.** Keep a real subprocess for what the
  subprocess is: argv, quoting, containment, kill. The write scheduler's
  `respond=` dispatcher and hand-built state files are the usual seams.
- **Give a real wait a test-only knob.** A kill grace, poll budget or probe
  timeout should be an argument the test can shorten, with margins that hold on
  a slow Windows runner under xdist.
- **Reuse a repository template when the shape matches.**
  [`tests/support/git_template.py`](../skills/multi-llm/tests/support/git_template.py)
  builds a repository shape once per worker and copies it; keep explicit
  construction where config, mutable history or containment is the subject.
- **Share a read-only run only among readers.**
  [`tests/support/shared_stop.py`](../skills/multi-llm/tests/support/shared_stop.py)
  shares one check-gate stop among tests that only read it, and checks after
  each that nothing changed it. Anything that resumes, answers or writes builds
  its own.
- **Give a migration guard its exit.** Its docstring names the change that
  deletes it, so it does not outlive the refactor.

### CI trial: parallel suite

`.github/workflows/test.yml` has a second job, `test-parallel`, beside the gating
`test` job: the same suite and the same `-m "not live and not container and not
package and not codex_jail"` selection on ubuntu, macOS and Windows, run with
`pytest -n auto --dist loadfile` (pytest-xdist, from the `dev` dependency group).
`loadfile` keeps each test file on one worker. Locally, four workers finish the
suite in about six minutes.

**Why it matters most on Windows.** The serial Windows leg took about 190 min of
pytest on 2026-11-08, and the suite has about a third more tests since, so the
serial leg may now come close to its 240-minute cap.

The leg is **non-blocking** (`continue-on-error: true`) and its checks are named
`test-parallel (<os>)`, so no required check changes. It is a one-week trial,
because hosted runners are slower than a developer machine and tests that time a
2 to 3 s child can flake there. Compare it with the serial legs on the same
commits: pass rate, flakes (a failure the serial leg does not share) and wall time.

Both jobs upload a JUnit file per OS (artifacts `junit-test-<os>` and
`junit-test-parallel-<os>`, kept 30 days, failed runs included), and both print
their 25 slowest tests. The parallel leg's log cannot be profiled by file, since
its workers interleave, so compare legs by file from the JUnit files:

```bash
gh run download <run-id> --repo beastlabai/multi-llm-plugin --pattern 'junit-*' --dir junit
python skills/multi-llm/tests/tools/junit_times.py junit/junit-test-parallel-windows-latest/junit.xml
```

The sums are per-test time, not wall time; read wall time from the job, and the
worker count from the log's `created: N/N workers` line. A fix for a
parallel-only failure goes at its source (a widened budget, a test that no longer
depends on wall time), never a blanket rerun or `-p no:xdist`.

- **Promote it** when the week is clean: make `test-parallel` the gating job, by
  moving the `-n auto --dist loadfile` flags into `test`'s command and deleting
  the trial job, so the check names `test (<os>)` stay the required ones.
- **Fall back to file shards** if Windows stays too slow or the flakes do not go
  away: split the file list over a matrix instead, each shard still running whole
  files. Bin-pack files by their Windows JUnit seconds, give a file missing from
  the table a fixed weight so it still runs, and keep the required check name
  with a final `test (windows-latest)` job that fails unless every shard passed.
- **Drop it** if neither holds; the serial job is unchanged and nothing depends
  on this one.

---

## Containerized E2E tests (Docker)

The container harness builds an image with the **real** coding-agent CLIs
installed, redirects their model backends at a local stub, runs the workflow and
asserts the artifacts. It catches the two drift classes nothing else here can
see: a provider CLI that changed its flags or event stream under us, and a host
that no longer discovers or dispatches what `install-host` wrote.

It runs through `make` targets at the repo root — the build context *is* the repo
root, so there is no `-C`:

```bash
make e2e-container                 # pinned build; provider lane (no secrets, --network none)
make e2e-container-host-discovery  # host lane, credential-free tier: stage + list, no key, no network
make e2e-container-host            # pinned build; host lane (needs OPENROUTER_API_KEY)
make e2e-container-latest          # CLIs @latest; provider lane only; credential-free by construction
make e2e-container-latest-host     # CLIs @latest; host lane; guarded, opt-in, needs OPENROUTER_API_KEY
make e2e-live-provider [CLI=x]     # credentialed PROVIDER lane; SPENDS CREDITS on openrouter.ai
make e2e-version-delta             # print the pinned → installed versions of the @latest image
make e2e-lock-update               # move the pins the @latest image proves moved, then prove it
make e2e-verify-record REC=<path>  # re-hash a runs/ bundle against its run record
make e2e-clean                     # prune the ephemeral runs/ bundles
```

`make help` prints the same list. A missing Docker binary or unreachable daemon
is reported in one line by the `e2e-preflight` step that every build target
depends on.

Three things worth knowing before running these:

- **Two lanes, two questions.** The *provider lane* asks whether the real CLI
  still accepts our argv and emits a stream we parse — it runs with `--network
  none` and no credential of any kind. The *host lane* asks whether the real host
  still discovers, loads and runs our shim; its full tier needs one
  `OPENROUTER_API_KEY` and spends real credit, which is why
  `make e2e-container-host-discovery` exists as a free, keyless tier that CI and a
  contributor without a key can run.
- **A key here should be a dedicated one with a key-level spend limit.** The
  whole-lane wall-clock cap is `MLLM_E2E_MATRIX_TIMEOUT_S` (default 3600),
  overridable on the command line.
- **The image is never pushed.** It is tagged `mllm-e2e:local` — no registry
  component, so a bare `docker push` fails by construction — and `docker save` of
  a host-lane container is equivalent to exporting a credential.

The harness has its own reference document:
[`skills/multi-llm/tests/container/README.md`](../skills/multi-llm/tests/container/README.md)
covers what each lane measures, the golden gate, the drift report, and the
artifact lifetimes. Run flags live in `posture.yaml`, build flags in the root
`Makefile`; neither is restated in prose.

---

## Generated files

Two documents are rendered from `skills/multi-llm/hosts/capabilities.yaml` and
must never be hand-edited — `skills/multi-llm/references/host-capabilities.md`
and the generated section of [`docs/hosts.md`](hosts.md):

```bash
uv run --project skills/multi-llm -- python skills/multi-llm/hosts/render_capabilities.py          # write both
uv run --project skills/multi-llm -- python skills/multi-llm/hosts/render_capabilities.py --check  # CI staleness guard
```

[`docs/development/markers.md`](development/markers.md), the registry of bracket markers
(`[BATCH_DONE]` and the rest), is rendered from the skill's and `packaging/`'s sources and
the instructions by `skills/multi-llm/tests/tools/render_markers.py` (`--check` for the
staleness guard); `tests/test_markers_registry.py` fails when it is stale, when a marker
is neither handled by an instruction nor listed as not-for-the-host, and when a printed
marker is built from parts the scan cannot list.

---

## Vendored report scripts

The HTML reports (`report.html` in both styles, and `consolidated-report.html`) render
model and plan text as Markdown with [marked](https://github.com/markedjs/marked) and then
sanitise it with [DOMPurify](https://github.com/cure53/DOMPurify). Both are vendored under
`skills/multi-llm/templates/vendor/`, and `utils/html_report_generator.py` inlines them
into every page at the `<!--MLLM:vendor-scripts-->` slot. The rules:

- **No page loads a script from the network.** A report is one file that reads the same
  offline, and opening it tells no CDN anything. `tests/test_html_report_scripts.py`
  fails on any `<script src>`, any module import from a URL, and any unpinned or
  integrity-less script.
- **The files are the npm tarballs' own bytes**, with their licence files beside them.
  `VENDORED_SCRIPTS` pins each file's sha256. The generator inlines a file only when its
  digest matches. When one does not match, or is missing, the page escapes model text
  instead of rendering it.
- **One renderer.** The `BEGIN SAFE MARKDOWN` block lives once, in the
  [shared report code](#shared-report-code) every page inlines, and nothing outside it
  calls marked or DOMPurify. DOMPurify is given a markdown-only
  profile: no `script`, `style` or `img` element, no form control, and no `id`, `class`,
  `style` or `data-*` attribute. That matters because the pages' own export code queries checkboxes by
  attribute. Without both libraries, or when DOMPurify reports `isSupported` false (it
  then returns its input unchanged), the text is escaped.
- **Everything else is escaped, in the same block.** Statuses, importances, model names,
  ids, labels, file paths and references come from model or validator output too. The
  block's `escapeHtml` escapes `& < > " '` and serves text and quoted attributes alike
  (`escapeAttr` is the same function, named for attribute sites); a value inside an
  inline handler's JS string takes `escapeJsAttr`. No class name is built from data:
  `getValidationBadgeClass` and `getImportanceBadgeClass` map onto a fixed list. The
  tests scan every template and the shared script for a raw value in an attribute, load
  hostile metadata into all three pages in headless Chrome, and run the real sanitiser
  there on hostile markdown (`tests/test_report_templates_js.py`). A reviewer's identity is the slot that ran it:
  code-review aggregation stamps `source_model` from the slot, whatever the reply claims.

To update a library, choose a release at least two weeks old. Download its tarball from
`registry.npmjs.org` and check it against the registry's `dist.integrity`. Then copy the
same file (`lib/marked.umd.js` or `dist/purify.min.js`) and its `LICENSE` over the old ones,
and update the version and sha256 in `VENDORED_SCRIPTS`. The current pins are marked
18.1.1 (tarball `sha512-zEGgrtUG8crk+TrMacJSVABU4DYDDkS8uqgU3ohrn/2fW9C2AEfUrO3m6IhpIpQpl7YcVpiXXC9cSMWDANT3dg==`)
and DOMPurify 3.4.17 (`sha512-B/pSlYVHQ58kqISD/yDjuJ1yfzI34m1gF4lAod2bwSRhiN4K5Cl3H8Af7hqB2PmTL911tdEudnnXn5FnhRoz0w==`).
A changed file changes what every package ships, so it needs a release bump.

---

## Shared report code

The three report pages (`report.html` in the flat and PR styles, and
`consolidated-report.html`) share their escaping, markdown, badges, notices, selection
storage and export, card order, commit-review badges and scope tags. That code lives once,
in `skills/multi-llm/templates/shared/report_core.js`, with the styles they share in
`report_core.css`. `utils/html_report_generator.py` inlines both into every page:

- **Where.** The style goes at `<!--MLLM:report-core-style-->`, before the page's own
  `<style>`, so a page's rule wins a tie. The script goes at
  `<!--MLLM:report-core-script-->`, after the vendored libraries and before the page's
  own `<script>`. Each is an element of its own, marked `data-shared`.
- **When.** These slots and the vendored scripts are filled first, then the slots derived
  from the data, then the data itself. Text in the data is never searched for a slot.
- **What may not be in them.** A missing file is a missing template: generation writes
  the error page. A file holding `<script`, `</script`, `</style`, `<!--` or the data
  placeholder is refused.
- **What a page supplies.** The shared script only declares functions and constants;
  nothing in it runs at load. It reads the page's `reportData` and three page hooks:
  `selections` (the fields the page keeps), `getStorageKey()` and `selectionsFile(now)`
  (the name and content its Export saves). That is how one `exportSelections` serves
  report.html, which saves the whole selection as `user_selections.json`, and the
  consolidated page, which saves only its groups' decisions as
  `consolidated_user_selections.json`.
- **What stays per page.** A function a page needs differently stays in its template:
  the filters, the renderers and `init`, the commit-review header and filter, the header
  panel, and each page's `:root` palette and component sizes. No page may declare a
  shared function again, because that would silently replace it for that page.

`tests/test_report_templates_js.py` generates every page kind and runs its code. Under
Node it parses every inline script and checks no page redefines a shared function. It
runs the escapers on hostile text, and a selection through export, reload and the Python
reader the apply phase uses. In headless Chrome it runs the real marked and DOMPurify on
hostile markdown. The Node checks skip without `node`, and the browser checks without
Chrome.

---

## Validating the plugin

```bash
claude plugin validate .   # validates marketplace.json and the referenced plugin
```
