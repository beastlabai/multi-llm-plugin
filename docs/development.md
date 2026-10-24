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

## Test suite

```bash
# Run the test suite (from the repo root)
uv run --project skills/multi-llm -- pytest

# Run a single test file
uv run --project skills/multi-llm -- pytest skills/multi-llm/tests/test_filtering.py -v

# Robustness (error-handling) tests
uv run --project skills/multi-llm -- pytest -m robustness
```

Two marker groups are excluded from a bare run. Tests marked `live` require
provider CLIs and real API access and are skipped in CI. Tests marked `container`
need a Docker daemon: `pyproject.toml` sets `addopts = -m 'not container'`, so a
bare `pytest` stays green without one, and each container-marked module also
carries a daemon probe that skips with a named reason.

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

---

## Validating the plugin

```bash
claude plugin validate .   # validates marketplace.json and the referenced plugin
```
