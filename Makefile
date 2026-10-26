# Developer entry points for the containerized E2E harness.
# Design: plans/containerized-harness-e2e.md (C9, File changes) — a local-only
# working note, gitignored and absent from a fresh clone.
#
# WHY THIS FILE IS AT THE REPO ROOT. The build context *is* the repo root — the
# product under test sits above tests/container/, and `COPY` cannot reach outside
# the context. A makefile beside the Dockerfile would spend every recipe reaching
# back through ../../../.., and the one flag pair that must not be mistyped
# (`-f <dockerfile> <repo-root>`) would be written relative to a directory nobody
# builds from. Every command here and in the README is a bare `make <target>`
# typed at the checkout root, with no -C.
#
# WHAT THIS FILE OWNS: the `docker build` invocation in full, the `docker version`
# pre-flight, and each target's `pytest -m` expression.
#
# WHAT IT DELIBERATELY DOES NOT OWN: `docker run` flags. Those are built per row
# by tests/container/harness.py from tests/container/posture.yaml — --network,
# --user, the --tmpfs mode strings, --cap-drop, --read-only, --pids-limit, the
# single /work/out bind and (host lane only) the one credential variable. That
# split is what makes the posture test able to inspect the run flags from both
# sides; a shell-quoted per-row matrix in a make recipe could not be asserted
# against at all.

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

CONTAINER_DIR := skills/multi-llm/tests/container
DOCKERFILE    := $(CONTAINER_DIR)/Dockerfile
IMAGE         := mllm-e2e:local
IMAGE_LATEST  := mllm-e2e:latest-probe
# No registry component in either tag, so a bare `docker push` fails by
# construction. `docker save` of a host-lane container is equivalent to exporting
# a credential; the README says so in those words.

# An arm64 developer and an amd64 machine would otherwise compare goldens produced
# by different binaries.
PLATFORM ?= linux/amd64

PYTEST := uv run --project skills/multi-llm -- pytest skills/multi-llm/tests

# --- The opencode npm package (packaging/opencode/) -------------------------- #
# A different layer of npm from the container's pinned CLIs above: those are
# `npm ci`-installed INTO the image, this one is a package we publish. They share
# nothing but the word.
OPENCODE_PKG   := packaging/opencode
# THE ONE REGISTRY. Every place npm is consulted on the release path — the
# `npm whoami` check, row 9's absence probe, step 5's registry-install
# probe, both publish paths and the post-publish integrity read
# — passes this explicitly, and the workflow reads it from here
# (`make -s opencode-print-registry`) rather than spelling it again. npm's registry
# is configurable (a user `.npmrc`, `npm_config_registry`, a scoped registry), so
# an inherited setting could otherwise make a gate accept `E404` from a different
# server than the one publication targets. packaging/opencode/registry.py reads
# this line; nothing else may define the URL.
OPENCODE_REGISTRY := https://registry.npmjs.org
# `ALLOW_DIRTY=1` stages from a tree with uncommitted or untracked changes. The
# build refuses one by default, and that default is correct: a build that ships
# local state is a build nobody can reproduce, and npm makes the result immutable
# the moment it is published. This exists so the gate is runnable *during*
# development, and it is deliberately a named opt-in rather than a silent
# fallback — `opencode-publish` re-runs the build with no such flag, so a dirty
# publish is refused however this was invoked.
# Exported rather than passed as a flag: `npm pack` invokes the manifest's
# `prepack` hook with its own fixed command line, so a flag here cannot reach it.
export MULTI_LLM_ALLOW_DIRTY := $(if $(ALLOW_DIRTY),1,)
OPENCODE_DIRTY := $(if $(ALLOW_DIRTY),--allow-dirty,)
OPENCODE_BUILD := uv run --project skills/multi-llm -- python $(OPENCODE_PKG)/build.py $(OPENCODE_DIRTY)
# The build with every spelling of the dirty opt-in removed — `env` rather than
# a bare `VAR=` prefix so it also works as the argv of a gate step, which runs
# without a shell.
OPENCODE_BUILD_CLEAN := env MULTI_LLM_ALLOW_DIRTY= uv run --project skills/multi-llm -- python $(OPENCODE_PKG)/build.py
OPENCODE_STAMP := $(OPENCODE_PKG)/.package-check.stamp
# The tarball `npm pack` writes for this version, `<name>-<version>.tgz`, named
# from the manifest when the recipe runs. Exact, never `ls -t | head -1`: a stale
# tarball of another version sitting beside it must not be the one that is
# scanned, loaded by row 7 and bound by the stamp.
OPENCODE_TARBALL = $(OPENCODE_PKG)/$$(node -p "const m = require('./$(OPENCODE_PKG)/package.json'); m.name + '-' + m.version + '.tgz'")
# The release gate's one executable contract (packaging/opencode/release_gate.py):
# the rows of plans/opencode-plugin-publish.md step 0, the session their evidence
# is written under, row 8's image binding, what --write-stamp requires, and the
# release-gate record CI validates. Its evidence lives under OPENCODE_GATE_DIR
# (gitignored). `step` runs one gate step under a wall-clock bound, in its own
# session with no terminal (so a timeout kills the whole process group, and
# `opencode run` is not wedged the way a plain `timeout N` wedges it), and keeps
# its output in $(OPENCODE_GATE_DIR)/logs/<step>.log.
OPENCODE_GATE := uv run --project skills/multi-llm -- python $(OPENCODE_PKG)/release_gate.py
OPENCODE_GATE_DIR := $(OPENCODE_PKG)/.gate
OPENCODE_GATE_STEP := $(OPENCODE_GATE) step
# The image the full gate builds for row 8, under its own tag: `mllm-e2e:local`
# is whatever the last `make e2e-image` staged, dirty tree included.
OPENCODE_GATE_IMAGE := mllm-e2e:opencode-gate
# Row 8: the package-route scenarios and nothing else. Every key unset, no live or
# credentialed tier selected; MLLM_E2E_RUNS_KEEP=0 keeps the lane from pruning
# the bundles check-row8 is about to verify.
OPENCODE_GATE_PYTEST := env -u OPENROUTER_API_KEY -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN \
    -u OPENAI_API_KEY MLLM_E2E_RUNS_KEEP=0 uv run --project skills/multi-llm -- pytest \
    skills/multi-llm/tests/test_container_opencode_package_gate.py -q -rs \
    -m "container and not live and not live_provider"
# The package rows only. `$(PYTEST)` already names the whole tests directory, so
# appending files to it would collect the entire suite *and* those files — which
# is the `test` job's work, done a second time in a job that exists to grade the
# tarball.
PACKAGE_PYTEST := uv run --project skills/multi-llm -- pytest \
    skills/multi-llm/tests/test_opencode_package.py \
    skills/multi-llm/tests/test_uv_candidates.py

# --- The codex plugin (packaging/codex/) ------------------------------------ #
# Unlike the npm package, the built tree `packaging/codex/multi-llm/` is
# COMMITTED: the Git marketplace route serves the checked-in copy with no build
# on the user's machine. So there are two builds, and which one runs is the
# ordering constraint (plans/codex-native-plugin.md, Phase 1 item 2, A6):
#   codex-regen    the development regeneration. --allow-dirty, writes the
#                  tracked tree, takes no stamp, reaches no publish. It runs
#                  BEFORE the commit, on inputs dirty by construction — the
#                  source edit and the output it implies land in one commit.
#   codex-package  the clean build, AFTER the commit. Refuses dirty inputs
#                  whatever ALLOW_DIRTY says, and the rebuilt tree must equal
#                  the committed one. Both check targets and codex-publish stand
#                  on it.
CODEX_PKG   := packaging/codex
CODEX_BUILD := uv run --project skills/multi-llm -- python $(CODEX_PKG)/build.py
CODEX_STAMP := $(CODEX_PKG)/.package-check.stamp
# The codex route's own rows. Not PACKAGE_PYTEST, which names the opencode suite
# literally and would grade this tree with the other route's tests.
# test_uv_candidates.py is registry-neutral (it greps the whole tracked tree), so
# it is the row that sees a new candidate-list reader added under packaging/codex/.
CODEX_PACKAGE_PYTEST := uv run --project skills/multi-llm -- pytest \
    skills/multi-llm/tests/test_codex_package.py \
    skills/multi-llm/tests/test_codex_mcp_server.py \
    skills/multi-llm/tests/test_uv_candidates.py
# The full gate's container half (A4 rows 1, 6b, 7, 13, 14). release_gate.py owns which
# rows each tier requires (host_matrix.yaml's `gate_tiers`), the codex version
# each tier runs, and the accounting that turns run records into the evidence
# the stamp requires.
CODEX_GATE := uv run --project skills/multi-llm -- python $(CODEX_PKG)/release_gate.py
CODEX_VERSION_DOCKERFILE := skills/multi-llm/tests/container/Dockerfile.codex-version
CODEX_MINIMUM_IMAGE := mllm-e2e:codex-minimum
CODEX_CEILING_IMAGE := mllm-e2e:codex-ceiling
# The zero-quota package rows and nothing else: every key unset, no live or
# credentialed tier selected, the module's own row filter set by the tier.
CODEX_GATE_PYTEST := env -u OPENROUTER_API_KEY -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN \
    -u OPENAI_API_KEY uv run --project skills/multi-llm -- pytest \
    skills/multi-llm/tests/test_container_host_lane.py -q -rs \
    -m "container and not live and not live_provider and not live_native_phase and not live_next_command" \
    -k "codex-package or plugin_floor or gate_tier"

# The whole-lane wall-clock cap (C1). Defaulted here, overridable on the command
# line, and ENFORCED in the host-side driver (tests/container/harness.py's
# MatrixBudget, claimed by run_row before every docker run and by the surface
# probe): once it is spent the driver starts nothing further, names which rows
# ran, failed and never started, and the invocation exits non-zero. Exported
# rather than passed per target because a value only this file knows would be a
# cap in the makefile and no cap in a bare `pytest -m container`.
# The PER-ROW cap is not here: it is per CLI, so it lives in redirect.yaml
# next to the measurement it was derived from.
MLLM_E2E_MATRIX_TIMEOUT_S ?= 3600
export MLLM_E2E_MATRIX_TIMEOUT_S

# The image tags are values, never rule targets: `mllm-e2e:local` contains a colon,
# which make would read as a target/prerequisite separator.
.PHONY: help e2e-preflight e2e-image e2e-image-latest e2e-container e2e-container-host \
        e2e-container-host-phases e2e-container-host-next-command \
        e2e-container-host-discovery e2e-container-host-invocation \
        e2e-version-delta e2e-container-latest \
        e2e-container-latest-host e2e-live-provider e2e-lock-update e2e-verify-record \
        e2e-clean e2e-prune \
        opencode-stage opencode-package opencode-package-check-ci \
        opencode-package-check opencode-publish opencode-bump opencode-latest-check \
        opencode-print-registry opencode-check-bump opencode-registry-check \
        opencode-registry-absent opencode-publish-first \
        codex-regen codex-package codex-package-check-ci codex-package-check \
        codex-gate-reset codex-gate-pinned codex-gate-minimum codex-gate-ceiling \
        codex-gate-derived codex-publish

help:
	@echo "Containerized E2E harness (see $(CONTAINER_DIR)/README.md)"
	@echo
	@echo "  make e2e-container                 pinned build; provider lane (no secrets, --network none)"
	@echo "  make e2e-container-host            pinned build; host lane (codex/claude-code on the subscription; OPENROUTER_API_KEY for opencode)"
	@echo "  make e2e-container-host-phases     host lane, claude-code per-phase tier; EXPENSIVE (~\$$3-5/pass)"
	@echo "  make e2e-container-host-next-command host lane, T10 next-command rows (needs OPENROUTER_API_KEY)"
	@echo "  make e2e-container-host-discovery  host lane, --creds none tier: stage + list, no key, no network"
	@echo "  make e2e-container-host-invocation host lane, --creds none tier: type the documented /multi-llm string"
	@echo "  make e2e-container-latest         CLIs @latest; provider lane only; credential-free by construction"
	@echo "  make e2e-container-latest-host    CLIs @latest; host lane; guarded, opt-in, needs OPENROUTER_API_KEY"
	@echo "  make e2e-live-provider [CLI=x]    credentialed PROVIDER lane; SPENDS CREDITS on openrouter.ai"
	@echo "  make e2e-version-delta            print the pinned → installed versions of the @latest image"
	@echo "  make e2e-lock-update              move the pins the @latest image proves moved, then prove it"
	@echo "  make e2e-verify-record REC=<path> re-hash a runs/ bundle against its run record"
	@echo "  make e2e-prune [KEEP=n]           keep the newest n runs/ bundles, delete the rest"
	@echo "  make e2e-clean                    prune the ephemeral runs/ bundles"
	@echo
	@echo "OpenCode npm package (see $(OPENCODE_PKG)/README.md)"
	@echo
	@echo "  make opencode-stage               stage the plugin payload the image COPYs (build input only)"
	@echo "  make opencode-package             build, pack, then sweep the tarball for secrets"
	@echo "  make opencode-package-check-ci    rows 1-6 and 5b, graded test by test: no Docker, no registry, no opencode binary"
	@echo "  make opencode-package-check       the FULL gate: rows 7-9 too, in one session; stamps and exports its record only if all passed"
	@echo "  make opencode-registry-absent [OUT=f] row 9: the version must be absent from the single registry"
	@echo "  make opencode-publish-first [DRY_RUN=1] the laptop FIRST release: uploads the gated tarball"
	@echo "  make opencode-publish             refuses: names the two guarded publish paths"
	@echo "  make opencode-bump VERSION=x.y.z  move every opencode pin to one release; re-capture its fixtures (\$$0)"
	@echo "                                    (a release older than the pin also needs ALLOW_DOWNGRADE=1)"
	@echo "  make opencode-latest-check        exit 1 when npm has a newer opencode than the pin"
	@echo "  make opencode-check-bump          the shared-version check: HEAD against every codex-v*/opencode-v* tag and npm's versions"
	@echo "  make opencode-registry-check SPEC=n@v TAG=t ENVIRONMENT=ci-runner|container|vm OUT=f [REHEARSAL_REGISTRY=url]"
	@echo "                                    the registry-install probe: the bytes npm serves, installed by OpenCode"
	@echo
	@echo "Codex plugin, checked in at $(CODEX_PKG)/multi-llm (see $(CODEX_PKG)/README.md)"
	@echo
	@echo "  make codex-regen                  regenerate the tracked plugin tree from dirty inputs (before the commit)"
	@echo "  make codex-package                clean rebuild; must equal the committed tree (after the commit)"
	@echo "  make codex-package-check-ci       the A4 rows that need no Docker and no codex binary"
	@echo "  make codex-package-check          the FULL gate; stamps only if every row ran"
	@echo "  make codex-gate-pinned            the gate's container rows on the pinned codex (rows 1, 6b, 7, 13)"
	@echo "  make codex-gate-minimum           the same rows on the declared minimum codex"
	@echo "  make codex-gate-ceiling           the same rows on npm's newest codex (row 14; fetches it)"
	@echo "  make codex-publish                refuses unless the full gate passed on these exact bytes; tags codex-v<version>"

# One line naming what is missing (binary absent vs daemon unreachable), rather
# than a page of CLI errors.
e2e-preflight:
	@command -v docker >/dev/null 2>&1 || { \
	  echo "make: docker is not on PATH — install Docker or run the non-container suite with 'pytest -m \"not container\"'"; exit 1; }
	@docker version >/dev/null 2>&1 || { \
	  echo "make: the docker CLI is present but no daemon is reachable — start Docker and retry"; exit 1; }

# The opencode PLUGIN payload, staged for the image's §10b COPY.
#
# STAGING ONLY — no `npm pack`, no secret sweep, no stamp. Those belong to
# `opencode-package`, which is the publishing path; this is the build-input path,
# and the two are separated so that building a test image can never look like a
# step toward publishing one.
#
# It is a prerequisite of `e2e-image` because `dist/` and `skills/` are
# .gitignore'd build output: a fresh clone has neither, and the Dockerfile COPYs
# both. Without this the first `make e2e-image` after a clone fails inside
# docker on a missing build-context path — which reads as a broken Dockerfile
# rather than as a missing prior step.
#
# `--allow-dirty` BY DEFAULT, through OPENCODE_STAGE_DIRTY: the payload has to
# match the working tree the image is being built from, which is the whole point
# of a test image, and refusing a dirty tree would make the container lane
# unrunnable during development. A plain `make e2e-image` keeps that path.
#
# THE FULL GATE OVERRIDES IT. Row 8 grades the package inside the image, and an
# image staged from a dirty tree is not the candidate, so `opencode-package-check`
# builds its image with `OPENCODE_STAGE_DIRTY=$(OPENCODE_DIRTY)` — empty unless
# ALLOW_DIRTY is set — and the stage below then refuses a dirty tree like any
# clean build. That alone does not bind the image (a clean `git status` cannot see
# ignored files), which is why the gate also reads the payload digest back out of
# the image id (`release_gate.py image`). MULTI_LLM_ALLOW_DIRTY is blanked on the
# line, so the exported opt-in cannot re-enable what the flag left off.
OPENCODE_STAGE_DIRTY ?= --allow-dirty
opencode-stage:
	@MULTI_LLM_ALLOW_DIRTY= uv run --project skills/multi-llm -- python $(OPENCODE_PKG)/build.py $(OPENCODE_STAGE_DIRTY) >/dev/null
	@echo "opencode-stage: payload staged at $(OPENCODE_PKG)/{dist,skills}$(if $(OPENCODE_STAGE_DIRTY),, from a clean tree)"

# --------------------------------------------------------------------------- #
# Pinned build: the gate you run before a release.
# --------------------------------------------------------------------------- #
e2e-image: e2e-preflight opencode-stage
	docker build \
	  --platform $(PLATFORM) \
	  --build-arg MLLM_PIN_MODE=pinned \
	  -f $(DOCKERFILE) \
	  -t $(IMAGE) \
	  .

# `not live_provider` is a SPEND GUARD, not tidiness. The credentialed lane's
# tests carry `container` + `live_provider` and deliberately NOT `live`, so
# `-m "container and not live"` SELECTS them -- and this recipe inherits the
# caller's environment, so a developer who happens to export OPENROUTER_API_KEY
# would spend real credit on the gate whose whole contract is "no key, no
# network, nothing on the host". It skipped in practice only when the key was
# absent, which is luck rather than design. `make e2e-live-provider` stays the
# one thing that runs that lane, and it is the one thing that sources the key.
e2e-container: e2e-image
	$(PYTEST) -m "container and not live and not live_provider"

# `not live_native_phase` is a SPEND GUARD, in the same spirit as the
# `not live_provider` one above. The claude-code per-phase rows are native Task
# dispatches: every turn is real spend that never crosses /opt/mllm/bin and
# never appears in the row's `num_turns`. Measured, they cost ~$0.60 per write
# row against this lane's $0.0080-$0.0094 for an executor host's equivalent —
# an order of magnitude — so folding them in here would make the cheap tier
# ~$3-5 more expensive every pass. They get their own target below.
#
# The OpenRouter key is NOT demanded here since 2026-10-02: codex and claude-code
# rows run on the developer's own subscription when its credential file is
# present (`harness.default_host_creds`), so a subscription-only run needs no key
# and its opencode rows skip into the census as `not-run`-style named skips. The
# lane itself refuses a run where NO row could execute (`_refuse_empty_lane`
# exits non-zero), so dropping the gate cannot turn "nothing ran" green.
# `MLLM_E2E_HOST_CREDS=openrouter` forces every row back onto the key.
e2e-container-host: e2e-image
	@test -n "$${OPENROUTER_API_KEY:-}" || \
	  echo "make: OPENROUTER_API_KEY unset: codex and claude-code run on the subscription when its credential file exists; opencode rows skip (named, in the census)."
	OPENROUTER_API_KEY="$$OPENROUTER_API_KEY" $(PYTEST) -m "container and live and not live_native_phase and not live_next_command"

# The claude-code PER-PHASE tier: one workflow phase per row, seeded by the
# row's `pre_commands` leg so the host turn does exactly one phase natively.
#
# A target of its own, for two independent reasons, and either one would be
# enough:
#
# * **Cost.** `MLLM_E2E_BUDGET_USD` defaults to $2.00 and is the LANE total:
#   `_Spend.check()` refuses to start a row once the running key-usage total
#   reaches it, and since 2026-09-23 a mid-row poller (`tests/container/
#   host_spend.py`) also stops the row that crosses it — or crosses its own
#   `cost_budget_usd × MLLM_E2E_ROW_BUDGET_HEADROOM` — as `budget-exceeded`.
#   Either way every row after the crossing is abandoned rather than trimmed.
#   A ~$3-5 tier crosses $2.00 inside its third row — the failure mode being
#   a HALF-COVERED TIER, not an overspend. $6.00 covers the pass with headroom
#   for one retried row.
#   Raise the dedicated key's own spend limit above that figure before first
#   use. The 13-row provider tier spent $0.1610 of $2.00 on 2026-08-20 and its
#   default is therefore unchanged.
# * **Wall clock.** `MLLM_E2E_MATRIX_TIMEOUT_S` defaults to 3600 s and the
#   existing rows already spend 1968 s of it. Native phase rows measured
#   80-376 s each, so a COMBINED lane would exceed the cap. Split, each tier
#   fits: ~33 min for the provider tier as measured today, ~30-55 min here.
#
# Raise either bound deliberately, on a recorded overrun, not pre-emptively.
e2e-container-host-phases: e2e-image
	@test -n "$${OPENROUTER_API_KEY:-}" || { \
	  echo "make: the host phase tier needs OPENROUTER_API_KEY (a DEDICATED key carrying a key-level spend limit)."; \
	  echo "      It is the EXPENSIVE tier: budget ~\$$3-5 per green pass, and bring-up separately."; exit 1; }
	OPENROUTER_API_KEY="$$OPENROUTER_API_KEY" \
	MLLM_E2E_BUDGET_USD="$${MLLM_E2E_BUDGET_USD:-6.00}" \
	MLLM_E2E_MATRIX_TIMEOUT_S="$${MLLM_E2E_MATRIX_TIMEOUT_S:-3600}" \
	  $(PYTEST) -m "container and live and live_native_phase"

# T10 of plans/next-phase-command.md: the next-command live check. One
# `generate-tasks` row per host and route the lane can run (codex and opencode,
# file and package; claude-code as the plain skill it is staged as), each with
# a short typed follow-up session that hands the printed command to a FRESH
# session. Cheap (measured 2026-10-01: $0.001-0.03 a row on deepseek v4 flash,
# ~$0.13 for claude-code on gpt-6-luna) but paid, so it is its own tier:
# `make e2e-container-host` excludes `live_next_command`. One row at a time is
# this expression plus `-k '[<row id>]'` (brackets included) on the pytest line.
e2e-container-host-next-command: e2e-image
	@test -n "$${OPENROUTER_API_KEY:-}" || { \
	  echo "make: the next-command tier needs OPENROUTER_API_KEY (a DEDICATED key carrying a key-level spend limit)."; exit 1; }
	OPENROUTER_API_KEY="$$OPENROUTER_API_KEY" \
	MLLM_E2E_BUDGET_USD="$${MLLM_E2E_BUDGET_USD:-1.00}" \
	  $(PYTEST) -m "container and live and live_next_command"

# The third, FREE tier of the host lane (C6): stage the shim, cross the reload
# boundary, assert the command is listed and invocable — then stop. No workflow
# row, no model call, no credential of any kind, --network none like the provider
# lane. It buys the single most likely real-world break (the shim landing where
# the host no longer scans) at zero cost and zero secrets, which is why CI and a
# keyless contributor can run it.
# NOT `-m "container and live"`: the discovery tier is defined to need no
# credential, so its module carries `container` alone and a keyless contributor
# (and CI) can run it. Selecting `live` here would have hidden it behind the very
# gate that makes the credentialed rows unrunnable.
e2e-container-host-discovery: e2e-image
	env -u OPENROUTER_API_KEY -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN \
	  $(PYTEST)/test_container_host_lane.py -m "container and not live"

# The FOURTH credential-free host row, and the only one that types anything:
# install the shim, open a FRESH session with the host's own model pointed at the
# loopback stub, and type the exact `/multi-llm …` string README.md and
# docs/hosts.md publish. It answers the acceptance row those documents had no
# evidence for -- *does the published spelling reach our dispatcher on this
# host?* -- and it answers it per installable host, because the two do not
# install the same kind of object.
#
# Same terms as the discovery tier and for the same reasons: no credential, no
# route out, `--network none`. What is under test is ARGUMENT DELIVERY, not model
# output, so a stub is not a compromise here -- the question is answered by the
# bytes the host sends, not by what comes back. It runs in well under a minute.
#
# It is ALSO selected by `make e2e-container` (`container`, not `live`), which is
# deliberate: a documentation claim that has gone false should surface on the
# gate a contributor already runs, not only on a target they have to know about.
e2e-container-host-invocation: e2e-image
	env -u OPENROUTER_API_KEY -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN \
	  $(PYTEST)/test_container_documented_invocation.py -m "container and not live"

# --------------------------------------------------------------------------- #
# Drift probe: CLI layer @latest, every other pin held.
# --------------------------------------------------------------------------- #
e2e-image-latest: e2e-preflight opencode-stage
	docker build \
	  --platform $(PLATFORM) \
	  --build-arg MLLM_PIN_MODE=latest \
	  -f $(DOCKERFILE) \
	  -t $(IMAGE_LATEST) \
	  .

# The version column of C9's drift report, printed from the image itself.
#
# This is the one place the makefile writes `docker run` flags, and the exception
# is deliberate: this is not a lane row. There is no posture to assert from both
# sides, no credential, no /work/out bind — it re-runs the build's own last step
# (verify_versions.py) against the installed binaries and prints, per row,
# `expected=<pin> installed=<what is really there>` plus an explicit
# `(recorded, not enforced) <cli>: pinned X but installed Y` line for every pin
# that moved. `--mode latest` is what inverts the CLI rows from enforced to
# recorded, so a moved CLI is a report rather than a non-zero exit; node, npm, uv
# and both PATH checks stay enforced and would still fail here.
#
# --out and --home are redirected onto a tmpfs /tmp because the container runs as
# 10001, for which /opt/mllm/lock and the image's own /tmp are read-only — this
# step records nothing durable, the build already wrote
# /opt/mllm/lock/cli-versions.installed.json. The tmpfs is the same one
# test_container_cli_surface.py mounts to probe help text out of band.
e2e-version-delta: e2e-preflight
	@docker image inspect $(IMAGE_LATEST) >/dev/null 2>&1 || { \
	  echo "make: $(IMAGE_LATEST) is not built — run 'make e2e-image-latest' first."; \
	  echo "      (the tag carries no registry component, so docker cannot pull it)"; exit 1; }
	@echo "pinned → installed version delta ($(IMAGE_LATEST)):"
	docker run --rm --network none --tmpfs /tmp:rw,mode=1777 \
	  --entrypoint /usr/bin/python3 $(IMAGE_LATEST) \
	  /opt/mllm/rig/verify_versions.py \
	    --lock /opt/mllm/lock/cli-versions.lock \
	    --mode latest \
	    --out /tmp/cli-versions.installed.json \
	    --home /tmp/mllm-version-delta-home

# THE LATEST TARGET NEVER TOUCHES A CREDENTIAL. Installing every CLI @latest means
# the image contains third-party binaries nobody pinned, reviewed or hashed.
# Running "the same suite" under that image would, on any machine that happens to
# export OPENROUTER_API_KEY, select the host lane too and hand those binaries a
# live key plus outbound network — a supply-chain hole that would open silently,
# from a variable already in the developer's shell. So the lane is selected
# explicitly and no credential variable is passed into pytest at all.
#
# THE DRIFT REPORT IS THE OUTPUT OF THE PARTS THAT EXIST, not a pytest flag. This
# recipe used to end in `--drift-report`, which no `pytest_addoption` anywhere in
# this repo registers: pytest exited on `unrecognized arguments` — after the full
# @latest image build — and the target reported nothing at all. Until an
# aggregating report ships, its three inputs are printed by the three things that
# already produce them:
#   * versions — `e2e-version-delta` above;
#   * the surface probe — test_container_cli_surface.py fails hard on an option
#     that vanished and reports a changed help digest as a SKIP, which is why
#     `-rs` is here: without it pytest prints a skip count and swallows the
#     `DRIFT REPORT — …` reason lines. Those lines are the report's *console*
#     half only; the durable half is written per row to
#     $(CONTAINER_DIR)/runs/<utc>/cli-surface-drift.json (CLI, contract version,
#     installed version, both digests), so a run read after the fact — or one
#     whose skip reasons a build log buried — still names what moved;
#   * the parse assertions — the provider lane's own failures.
e2e-container-latest: e2e-image-latest
	@$(MAKE) --no-print-directory e2e-version-delta
	env -u OPENROUTER_API_KEY -u ANTHROPIC_API_KEY -u OPENAI_API_KEY -u GEMINI_API_KEY \
	  MLLM_E2E_IMAGE=$(IMAGE_LATEST) \
	  $(PYTEST) -m "container and not live and not live_provider" -rs

# The one place in this design where an unpinned image is allowed near a key.
# The guard is a target-level refusal, not a README sentence.
#
# THE KEY IS REQUIRED UP FRONT, and that is not a convenience check. This target
# *is* the credentialed host lane under an unpinned image: every test its
# `-m "container and live"` expression selects skips by name when
# OPENROUTER_API_KEY is absent. So a keyless invocation used to skip the opt-in
# guard below (whose condition began `[ -n "$$OPENROUTER_API_KEY" ]`), build the
# whole @latest image, run pytest, skip every row and exit **0** — an explicitly
# invoked "host lane against @latest" reporting success while proving nothing,
# which is the exact outcome C6 calls `not-run`. Refusing before the build also
# means a missing key costs no image.
#
# THE ZERO-EXECUTED-ROW GUARD RUNS AFTER IT, and `--allow-empty-host-lane` is
# deliberately NOT mirrored onto this target: an empty run of *this* one is never
# something to accept. The pytest-side guard reports `not-run` for the keyless
# case, which is already refused above; the check below covers what it cannot see
# — key present, rows selected, and every one of them skipped for some other
# reason. Its evidence is C10's own: a host-lane row that really ran writes
# `records/<utc>-<row>.json` from `harness.run_row`, so "no record newer than the
# stamp" *is* "no row executed", and a green pytest over an empty lane fails here
# rather than being read as a pass.
#
# THE VENDOR KEYS ARE SCRUBBED, NOT REFUSED, and the asymmetry is deliberate.
# ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, OPENAI_API_KEY and GEMINI_API_KEY are
# credentials in exactly the sense this guard cares about, but none of them can
# *become* this lane's credential: `harness.build_run_argv` forwards one explicit
# `-e` list per row — `OPENROUTER_API_KEY` by name, `ANTHROPIC_API_KEY=` as the
# literal empty string, and nothing else — so a vendor key sitting in the
# developer's shell is never handed to a container, pinned or unpinned. Refusing
# on one would make the target unrunnable on every machine that also drives its
# own models, and would buy nothing. So this target does what the sibling
# `e2e-container-latest` does: it unsets them for the run, and the guard above
# names any it found rather than staying silent about them. That is defence in
# depth against the next change to the forwarded list, not a fix for a live leak.
# The scrub is host-side ONLY and cannot disturb the in-container assertion —
# `-e VAR=` is an explicit assignment, so ANTHROPIC_API_KEY still arrives
# present-and-EMPTY on the openrouter path (C6), which is the one spelling that
# keeps Claude Code off the developer's own Anthropic subscription. Unsetting
# OPENROUTER_API_KEY here would instead delete the lane's credential outright,
# since `-e VAR` with no value forwards nothing when the variable is unset — it
# is passed through by name, and it is the only credential this target gives away.
#
# The version delta prints BEFORE the first row, because the list of pins that
# moved is precisely what the opt-in is accepting. It was spelled
# `--print-version-delta` at pytest, which nothing registers — so the target
# exited on `unrecognized arguments` and printed no delta at all, which is the
# one thing it must not do before handing a live key to unreviewed binaries.
e2e-container-latest-host: e2e-preflight
	@test -n "$${OPENROUTER_API_KEY:-}" || { \
	  echo "make: this target IS the credentialed host lane, so it needs OPENROUTER_API_KEY"; \
	  echo "      (a DEDICATED key carrying a key-level spend limit). Without one every row"; \
	  echo "      skips, the lane covers nothing, and an @latest image proves nothing —"; \
	  echo "      refusing here rather than building it first."; \
	  echo "      Credential-free instead: make e2e-container-host-discovery  (host lane, free tier)"; \
	  echo "                               make e2e-container-latest          (provider lane @latest)"; \
	  exit 1; }
	@if [ "$${MLLM_ALLOW_UNPINNED_HOST_LANE:-}" != "1" ]; then \
	  echo "make: refusing to run the host lane against an UNPINNED image with a credential present."; \
	  echo "      found: OPENROUTER_API_KEY"; \
	  for v in ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN OPENAI_API_KEY GEMINI_API_KEY; do \
	    if [ -n "$${!v:-}" ]; then \
	      echo "      also exported: $$v (unset for the run below — no container is ever given it)"; \
	    fi; \
	  done; \
	  echo "      This target installs every CLI @latest — unreviewed, unhashed binaries — and would"; \
	  echo "      hand them a live key plus outbound network. If that is what you want, set"; \
	  echo "      MLLM_ALLOW_UNPINNED_HOST_LANE=1 deliberately; it must never come from a shell profile."; \
	  exit 1; \
	fi
	$(MAKE) e2e-image-latest
	@$(MAKE) --no-print-directory e2e-version-delta
	@mkdir -p $(CONTAINER_DIR)/records
	stamp="$$(mktemp)"; trap 'rm -f "$$stamp"' EXIT; \
	env -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN -u OPENAI_API_KEY -u GEMINI_API_KEY \
	  MLLM_E2E_IMAGE=$(IMAGE_LATEST) OPENROUTER_API_KEY="$$OPENROUTER_API_KEY" \
	  $(PYTEST) -m "container and live" -rs; \
	if [ "$$(find $(CONTAINER_DIR)/records -name '*.json' -newer "$$stamp" | wc -l)" -eq 0 ]; then \
	  echo "make: NOT-RUN — pytest exited green but not one host-lane row executed:"; \
	  echo "      no $(CONTAINER_DIR)/records/<utc>-<row>.json was written by this invocation,"; \
	  echo "      and a row that ran always writes one. A lane that covered nothing is a"; \
	  echo "      distinct outcome from a pass, and it is reported as one."; \
	  echo "      The named skip reasons are in the run above (-rs) and in $(CONTAINER_DIR)/runs/."; \
	  exit 1; \
	fi

# --------------------------------------------------------------------------- #
# The credentialed PROVIDER lane (harness-expansion.md §7.4) — the third lane.
#
# WHY IT IS NOT A TIER OF ANY EXISTING TARGET. `e2e-container` runs the provider
# lane `--network none` against the stub: there is no openrouter.ai to reach.
# `e2e-container-host` proves the HOST's brain is live, and its PATH wrapper does
# the exact opposite for providers — it scrubs every vendor credential and points
# the CLI at the stub, which is the one thing that keeps a live key out of a
# third-party binary's environment on an open network. Neither can answer "does
# this adapter still reach the real endpoint on the user's own key", so that
# question gets its own target, its own marker and its own row list.
#
# THREE PROPERTIES HERE ARE CONTRACT, NOT RECIPE TASTE:
#
#   * IT IS A PREREQUISITE OF NOTHING. No other goal in this file may reach it —
#     skills/multi-llm/tests/test_makefile_lanes.py asserts that by parsing this
#     file, because a credit-spending target reachable from `check` or
#     `e2e-container` is a bill nobody typed.
#   * THE MARKER, NOT THE RECIPE, KEEPS IT OUT OF CI. The module carries
#     `live_provider` in addition to `container`, so pyproject's
#     `addopts = -m 'not container'` already deselects it and the CI expression
#     does too.
#   * IT NEVER BUILDS THE IMAGE. `e2e-image` is deliberately NOT a prerequisite:
#     a stale image must be a visible failure, not a silent rebuild that spends
#     credits on the wrong binary. Build it yourself, then run this.
#
# The key is read HOST-SIDE from ~/.config/multi-llm/e2e.env (0600, outside git)
# and reaches the container only as `-e OPENROUTER_API_KEY` on the docker run. It
# is never baked into the image, never written into redirect.yaml,
# cli-versions.lock, a home_seed/ file or any committed fixture, and the run
# record redacts it out of the CLI's own stdout before anything is written.
#
#   make e2e-live-provider            every landed Wave-1/2 row
#   make e2e-live-provider CLI=copilot  one row
e2e-live-provider:
	set -a; . $${MULTI_LLM_E2E_ENV:-$$HOME/.config/multi-llm/e2e.env}; set +a; \
	uv run --project skills/multi-llm -- pytest \
	  skills/multi-llm/tests/test_container_live_provider.py \
	  -m 'container and live_provider' $(if $(CLI),-k '[$(CLI)]',)
# `-k '[<id>]'` selects the PARAMETRIZE id, not a bare substring, and the
# brackets are load-bearing. A bare `-k pi` also matches the `skipif` marker
# keyword every item on this module carries via `requires_docker()` -- the
# substring is inside "ski-pi-f" -- so `CLI=pi` silently ran the whole ten-row
# lane and spent ten rows' credit instead of one. Measured 2026-08-19.

# --------------------------------------------------------------------------- #
# Provenance and housekeeping.
# --------------------------------------------------------------------------- #

# The pin writer, and the only target here that edits tracked files.
#
# IT MOVES WHAT THE @LATEST IMAGE PROVES, AND NAMES THE REST. The evidence is not
# a changelog and not a reading of the drift report: it is what a real
# `MLLM_PIN_MODE=latest` build carries inside itself — its own version census
# (`/opt/mllm/lock/cli-versions.installed.json`, written by the build's last
# layer), the `package-lock.json` that `npm install @latest` actually resolved,
# and, for the rows the Dockerfile installs as a plain file, the artifact bytes
# themselves. `lock_update.py` (whose docstring carries the full table) rewrites
# the npm layer and the `install: download` rows from those three, and REFUSES —
# by name, with the command — the rows it cannot substantiate: `goose`, whose
# tarball the image unpacks and discards, and `aider`, whose pin is a
# `--generate-hashes` requirements file no image can regenerate. The image-level
# pins (base digest, platform, uv) are held pinned by the latest build on
# purpose, so it can never be the thing that moved them; they change by hand.
#
# This recipe used to pass `--lock-update` to pytest, which nothing registers: it
# collected the whole container suite and then died on `unrecognized arguments`,
# having rewritten nothing. Batch 10 replaced that with an honest refusal; this
# is the writer the refusal was standing in for.
#
# THE REWRITE IS VERIFIED BY REBUILDING THE PINNED IMAGE, not by being printed.
# A pin move is a claim about bytes a future build will fetch, and the only thing
# that settles it is fetching them: the pinned build runs `npm ci` against the new
# lockfile, checks each artifact's sha256 before it executes, and ends in
# verify_versions.py with the CLI rows ENFORCED. `lock_update.py verify` then
# grades the record against that image's census, so every field the record claims
# to have moved is a field the rebuilt image really installed. A failure anywhere
# after the write prints the exact `git checkout --` that reverts it.
#
# `cli-surface.yaml` is deliberately NOT rewritten here: it is a contract, not a
# pin, and its own probe reports a changed help digest as "re-seed once reviewed".
# Re-seeding it as bookkeeping would discard the one upstream-facing signal in the
# design, so the re-seed command is printed against the rebuilt image instead.
LOCK_UPDATE := uv run --project skills/multi-llm -- python $(CONTAINER_DIR)/lock_update.py

e2e-lock-update: e2e-preflight
	@docker image inspect $(IMAGE_LATEST) >/dev/null 2>&1 || { \
	  echo "make: $(IMAGE_LATEST) is not built, so there is no evidence to move a pin with."; \
	  echo "      A pin moves on what a real @latest build installed, never on a version"; \
	  echo "      string someone read somewhere. Run 'make e2e-image-latest' first, or"; \
	  echo "      'make e2e-container-latest' for the whole drift report."; exit 1; }
	@mkdir -p $(CONTAINER_DIR)/runs/lock-update
	ev="$$(mktemp -d)"; \
	rec="$(CONTAINER_DIR)/runs/lock-update/$$(date -u +%Y%m%dT%H%M%SZ).json"; \
	cid="$$(docker create $(IMAGE_LATEST))"; \
	trap 'docker rm -f "$$cid" >/dev/null 2>&1 || true; rm -rf "$$ev"' EXIT; \
	mkdir -p "$$ev/npm" "$$ev/artifacts"; \
	$(LOCK_UPDATE) evidence | while read -r src dst; do \
	  docker cp "$$cid:$$src" "$$ev/$$dst" >/dev/null; \
	done; \
	rc=0; $(LOCK_UPDATE) apply --evidence "$$ev" --image $(IMAGE_LATEST) --record "$$rec" || rc=$$?; \
	if [ "$$rc" -eq 2 ]; then exit 0; fi; \
	if [ "$$rc" -ne 0 ]; then exit "$$rc"; fi; \
	echo "lock-update: rebuilding the PINNED image against the rewritten pins."; \
	echo "             That build is the verification: npm ci against the new lockfile,"; \
	echo "             every artifact checksum checked before it runs, and the CLI rows"; \
	echo "             enforced by verify_versions.py."; \
	if ! $(MAKE) --no-print-directory e2e-image; then \
	  echo "make: the pinned rebuild FAILED against the rewritten pins — they are wrong,"; \
	  echo "      and the record of what was written is $$rec."; \
	  echo "      Revert with the 'git checkout --' line printed above, which names"; \
	  echo "      exactly the files this run rewrote and nothing else."; \
	  exit 1; \
	fi; \
	cid2="$$(docker create $(IMAGE))"; \
	docker cp "$$cid2:/opt/mllm/lock/cli-versions.installed.json" "$$ev/pinned.json" >/dev/null; \
	docker rm -f "$$cid2" >/dev/null; \
	$(LOCK_UPDATE) verify --record "$$rec" --installed "$$ev/pinned.json"; \
	echo; \
	echo "Still by hand, and on purpose — the recorded CLI surface:"; \
	echo "  mkdir -p $(CONTAINER_DIR)/runs/surface && chmod 0777 $(CONTAINER_DIR)/runs/surface"; \
	echo "  # 0777 for the same reason harness.py does it: the bind is a host directory"; \
	echo "  # owned by you, and the container is uid 10001."; \
	echo "  docker run --rm --network none --user 10001:10001 \\"; \
	echo "    --tmpfs /home/e2e:rw,mode=0700,uid=10001,gid=10001 --tmpfs /tmp:rw,mode=1777 \\"; \
	echo "    -v \"\$$PWD/$(CONTAINER_DIR)/runs/surface:/work/out\" --entrypoint /usr/bin/python3 $(IMAGE) \\"; \
	echo "    /opt/mllm/rig/seed_cli_surface.py --out /work/out/cli-surface.json"; \
	echo "  then move the version + help digest of each reviewed row into"; \
	echo "  $(CONTAINER_DIR)/cli-surface.yaml. A changed digest is a deprecation notice to"; \
	echo "  READ, which is why it is not re-seeded as a side effect of moving a pin."

e2e-verify-record:
	@test -n "$(REC)" || { echo "usage: make e2e-verify-record REC=$(CONTAINER_DIR)/records/<utc>-<host>.json"; exit 1; }
	uv run --project skills/multi-llm -- python $(CONTAINER_DIR)/verify_record.py "$(REC)"

# runs/ is the raw evidence bundle: ephemeral, gitignored, safe to delete at any
# time. The run records in records/ beside it are gitignored too (local only),
# except the schema and the codex release network-check records.
#
# Gitignored is not the same as absent, and that distinction is the reason the
# retention target below exists: nothing that copies a WORKING TREE consults
# .gitignore. `codex plugin add` from a local-path marketplace and a local-path
# plugin install both take the directory as it stands, so an uncapped runs/ is a
# distribution problem and not merely a disk one. It reached 1.5 GB here against
# 18 MB of tracked files before anyone measured it.
#
# The lane prunes itself on every row (harness.prune_runs, MLLM_E2E_RUNS_KEEP);
# this target is the same sweep on demand, for a checkout that predates it.
KEEP ?= 8
e2e-prune:
	uv run --project skills/multi-llm -- python -c \
	  'import sys; sys.path.insert(0, "skills/multi-llm"); \
	   from tests.container.harness import prune_runs; \
	   r = prune_runs(keep=$(KEEP)); \
	   print(f"pruned {len(r)} bundle(s), kept the newest $(KEEP) (records/ untouched)")'

e2e-clean:
	rm -rf $(CONTAINER_DIR)/runs/*/
	@echo "pruned $(CONTAINER_DIR)/runs/ (records/ untouched)"


# --------------------------------------------------------------------------- #
# The OpenCode npm package.
#
# WHY THE BUILD AND THE PACK ARE SEPARATE. `build.py` stages deterministic
# artifacts and stops; it never invokes npm. Packing and publishing are owned
# here, so there is exactly one place each happens and a build cannot quietly
# produce a second tarball nobody publishes. The tarball secret sweep is a
# function `build.py` exposes and this file calls AFTER `npm pack`, because what
# npm writes is what ships and a `prepack` hook runs too early to see a tarball.
#
# WHY EVERY RULE BELOW RUNS THE BUILD EXPLICITLY. The manifest's `prepack` hook
# is a backstop for a bare `npm pack` someone types by hand; it does not fire
# under `--ignore-scripts`. The explicit call is what makes the staged tree fresh
# on every path. Running the build twice is harmless — it is deterministic and
# idempotent — and `test_package_is_built_before_every_pack` asserts the ordering
# rather than trusting this comment.
# --------------------------------------------------------------------------- #

opencode-package:
	$(OPENCODE_BUILD)
	cd $(OPENCODE_PKG) && npm pack
	$(OPENCODE_BUILD) --scan-tarball "$(OPENCODE_TARBALL)"

# Rows 1-6 and 5b: everything that needs no Docker, no registry and no `opencode`
# binary. This is what the PR package job runs, and the full gate's first step.
# It writes NO stamp — a CI-tier run can never authorize a publish.
#
# GRADED ROW BY ROW, NOT BY EXIT CODE. Both pytest selections write junit XML, and
# `release_gate.py grade-ci` reads it against the node ids each row declares: a
# row passes only when every one of its tests is present and passed. A skip, a
# deselection (the test is simply absent from the file) and an empty selection
# each fail the row by name, where pytest would have exited 0 or 5. Both
# selections always run and the grader always reads both, so a red run still
# names every row it cost; the recipe then fails if any of the three did. Inside a
# full-gate session for HEAD the grader also writes each row's evidence; anywhere
# else (the PR job's fresh clone) it grades only.
opencode-package-check-ci:
	$(OPENCODE_BUILD)
	cd $(OPENCODE_PKG) && npm pack --dry-run
	@mkdir -p $(OPENCODE_GATE_DIR)/junit
	@rm -f $(OPENCODE_GATE_DIR)/junit/ci-package-rows.xml $(OPENCODE_GATE_DIR)/junit/ci-plain-rows.xml
	rc=0; \
	MLLM_REQUIRE_PACKAGE_ROWS=1 $(PYTEST) -m "package and not container" -q \
	  --junitxml=$(OPENCODE_GATE_DIR)/junit/ci-package-rows.xml || rc=$$?; \
	$(PACKAGE_PYTEST) -m "not live and not container and not package" -q \
	  --junitxml=$(OPENCODE_GATE_DIR)/junit/ci-plain-rows.xml || rc=$$?; \
	$(OPENCODE_GATE) grade-ci --junit $(OPENCODE_GATE_DIR)/junit/ci-package-rows.xml \
	  --junit $(OPENCODE_GATE_DIR)/junit/ci-plain-rows.xml || rc=$$?; \
	exit $$rc
	@echo "opencode-package-check-ci: rows 1-6 and 5b graded green (no stamp written)"

# The full gate: every row of plans/opencode-plugin-publish.md step 0 — the CI
# rows (1-6, 5b), the clean-install probe (7), the package-route container
# scenarios (8) and the registry-absence check (9) — then bump discipline, then
# the stamp, then the release-gate record.
#
# IT STAMPS ONLY WHAT IT RAN, IN THIS RUN. The stamp is what authorizes
# `opencode-publish-first`, and npm makes a published version permanent, so a stamp
# written after a SUBSET of the rows is an authorization nobody earned. Recipe
# order is not what enforces that: `begin` opens a gate SESSION (earlier evidence
# deleted, a fresh id beside the candidate commit), every row's grader writes its
# evidence under that id, and `--write-stamp` refuses unless every row passed in
# this session over these bytes, this commit, the pinned host and the gated
# tarball. A zero pytest exit never satisfies a row on its own.
#
# SERIAL BY CONSTRUCTION, like codex-package-check: recipe lines, never
# prerequisites, so `make -j` cannot reorder the reset after a row, build the
# image before the candidate, or race row 8 against a rebuild. Each step runs
# through `release_gate.py step` with a wall-clock bound, and keeps its output in
# $(OPENCODE_GATE_DIR)/logs/<step>.log for when it fails.
#
#   * ROW 7 is wired: the pinned OpenCode, installed from the container lockfile
#     under a scratch prefix (`bump.py install`; its last stdout line is the
#     binary), loads the gated tarball extracted as a directory in an isolated
#     home and runs init and status through the real tool, in a committed and a
#     no-commit project (`probe.py row7`). `record 7` then re-reads what the probe
#     says it did: the pinned host, the single registry, the gated tarball's
#     integrity and digest, both projects green.
#   * ROW 8 runs on the image the gate builds itself, staged from the clean
#     candidate (OPENCODE_STAGE_DIRTY=$(OPENCODE_DIRTY) — never a literal
#     `--allow-dirty`), under its own tag. `image` resolves it to its immutable id,
#     reads the payload digest out of that id and refuses one built from other
#     bytes — and an inherited MLLM_E2E_IMAGE that names another image, before row
#     8 runs. The gate then sets MLLM_E2E_IMAGE to that id itself, unsets every
#     provider key, and runs the package-route scenario module (init/status,
#     detached launch, supervision, completion, cancellation, each with its own
#     grader). `check-row8` refuses a scenario missing, skipped, ungraded or
#     failing, a record from another image or another run, and a bundle
#     `make e2e-verify-record` does not MATCH.
#   * ROW 9 is the shared entry point (`opencode-registry-absent`, Node built-ins
#     only), decided on npm's error CODE: only E404 is absence; exit 0 is
#     "already published" and anything else is an unknown. The publishing
#     workflow's jobs run the same script: the two tiers still run separately,
#     but they share one implementation rather than two hand-kept copies.
#
# ALLOW_DIRTY=1 IS A REHEARSAL. Every row may run from a dirty tree, and the
# session says so; `--write-stamp` and `export-record` then both refuse it by
# name, so a rehearsal can never authorize a publish or reach CI.
#
# THE STAMP NEVER TRAVELS. It is gitignored and never uploaded, because a stamp
# from another machine attests to that machine's rows. CI consumes no stamps: its
# jobs re-run rows 1-7 and 9, and row 8 — which needs a Docker daemon the runners
# do not have — reaches CI only as the release-gate record exported at the end,
# from the session that authorized the stamp, into the gitignored
# $(OPENCODE_GATE_DIR)/export/ (never records/, where an untracked file would
# make the tree dirty and the stamp refuse). It is copied into
# $(CONTAINER_DIR)/records/ and committed in the default branch's own commit after
# the fast-forward (plan step 2), never in the tag's tree; the tag's commit is
# then checked out again before `make opencode-publish-first`. `export-record`
# prints those lines.
opencode-package-check:
	@$(OPENCODE_GATE) begin $(if $(ALLOW_DIRTY),--allow-dirty,)
	$(OPENCODE_GATE_STEP) ci-rows --timeout 3600 -- $(MAKE) --no-print-directory opencode-package-check-ci
	$(OPENCODE_GATE_STEP) candidate --timeout 900 -- $(MAKE) --no-print-directory opencode-package
	$(OPENCODE_GATE_STEP) host-install --timeout 900 --capture $(OPENCODE_GATE_DIR)/raw/opencode-bin.txt -- \
	  uv run --project skills/multi-llm -- python $(OPENCODE_PKG)/bump.py install --prefix $(abspath $(OPENCODE_GATE_DIR))/host
	$(OPENCODE_GATE_STEP) row7-probe --timeout 1800 -- \
	  uv run --project skills/multi-llm -- python $(OPENCODE_PKG)/probe.py row7 --tarball "$(OPENCODE_TARBALL)" \
	  --opencode "$$(cat $(OPENCODE_GATE_DIR)/raw/opencode-bin.txt)" --registry $(OPENCODE_REGISTRY) \
	  --out $(OPENCODE_GATE_DIR)/raw/row7.json
	$(OPENCODE_GATE_STEP) row7-record --timeout 300 -- $(OPENCODE_GATE) record 7 --from $(OPENCODE_GATE_DIR)/raw/row7.json
	$(OPENCODE_GATE_STEP) image-build --timeout 3600 -- \
	  $(MAKE) --no-print-directory e2e-image IMAGE=$(OPENCODE_GATE_IMAGE) OPENCODE_STAGE_DIRTY=$(OPENCODE_DIRTY)
	$(OPENCODE_GATE_STEP) image-bind --timeout 600 --capture $(OPENCODE_GATE_DIR)/raw/image-id.txt -- \
	  $(OPENCODE_GATE) image --image $(OPENCODE_GATE_IMAGE)
	@date -u +%Y%m%dT%H%M%SZ > $(OPENCODE_GATE_DIR)/raw/row8-since.txt
	$(OPENCODE_GATE_STEP) row8-run --timeout 5400 -- \
	  env MLLM_E2E_IMAGE="$$(cat $(OPENCODE_GATE_DIR)/raw/image-id.txt)" $(OPENCODE_GATE_PYTEST)
	$(OPENCODE_GATE_STEP) row8-check --timeout 900 -- \
	  $(OPENCODE_GATE) check-row8 --since "$$(cat $(OPENCODE_GATE_DIR)/raw/row8-since.txt)"
	$(OPENCODE_GATE_STEP) check-bump --timeout 600 -- $(MAKE) --no-print-directory opencode-check-bump
	$(OPENCODE_GATE_STEP) row9-probe --timeout 300 -- \
	  $(MAKE) --no-print-directory opencode-registry-absent OUT=$(OPENCODE_GATE_DIR)/raw/row9.json
	$(OPENCODE_GATE_STEP) row9-record --timeout 300 -- $(OPENCODE_GATE) record 9 --from $(OPENCODE_GATE_DIR)/raw/row9.json
	$(OPENCODE_GATE_STEP) stamp --timeout 900 -- $(OPENCODE_BUILD_CLEAN) --write-stamp "$$(date -u +%Y-%m-%dT%H:%M:%SZ)"
	$(OPENCODE_GATE_STEP) export --timeout 300 -- $(OPENCODE_GATE) export-record
	@$(OPENCODE_GATE) summary
	@echo "opencode-package-check: full gate green; stamp written to $(OPENCODE_STAMP), release-gate record exported"

# Row 9's one entry point (packaging/opencode/registry_absent.mjs): the version
# in $(OPENCODE_PKG)/package.json must be ABSENT from $(OPENCODE_REGISTRY), and
# only npm's `E404` says so. Exit 0 is "already published" and every other code
# is an unknown; both refuse. The script re-reads OPENCODE_REGISTRY from this
# file and refuses any other `--registry`, so the URL is still written once.
# `OUT=<file>` keeps the row-9 evidence JSON for `release_gate.py record 9`.
# `PACKAGE_JSON=<file>` asks about another manifest's name and version: the
# workflow's canary, `opencode-multi-llm@0.0.0`, which must answer E404 under the
# job's own npm setup. Honoured only from this make command line: an exported
# PACKAGE_JSON would quietly point row 9 at a different version (`record 9`
# would catch it in the full gate, a workflow's row 9 would not), so one from
# anywhere else is refused by name.
opencode-registry-absent:
	@if [ -n "$(PACKAGE_JSON)" ]; then case "$(origin PACKAGE_JSON)" in "command line") ;; *) \
	  echo "make: REFUSING: PACKAGE_JSON=$(PACKAGE_JSON) comes from the $(origin PACKAGE_JSON), not this make command line; row 9 asks about $(OPENCODE_PKG)/package.json unless PACKAGE_JSON=<file> is passed explicitly. Unset it." >&2; \
	  exit 1 ;; esac; fi
	node $(OPENCODE_PKG)/registry_absent.mjs --registry $(OPENCODE_REGISTRY) $(if $(OUT),--out "$(OUT)") \
	  $(if $(and $(filter command line,$(origin PACKAGE_JSON)),$(PACKAGE_JSON)),--package-json "$(PACKAGE_JSON)")

# THE LAPTOP FIRST RELEASE, and only that. npm's trusted publishing still needs
# an existing package (`npm trust`: "The package you're configuring must already
# exist", rechecked 2026-10-25), so the first version is uploaded from the
# maintainer's machine, authenticated interactively and WITHOUT provenance; every
# later version goes through the workflow dispatch, with it. first_release.py
# refuses, naming the condition, unless the tree is clean, the full stamp binds
# the gated tarball, the annotated `opencode-v<version>` tag names HEAD, HEAD is
# on origin/master, every required check run on that SHA succeeded (`test (<os>)`
# per test.yml matrix leg, plus `package`), the shared-version bump check and
# row 9 pass, it IS the first release (npm lists no version of the package at
# all, and HEAD's capabilities name no `public_version`), and the gated tarball
# still recomputes to the stamp. Then it
# uploads THAT FILE with `--ignore-scripts`, never the directory, and reads
# `dist.integrity` back from a fresh cache until it equals the gated one.
# `DRY_RUN=1` runs every preflight and prints the upload without running it.
# No ALLOW_DIRTY here, and the script strips MULTI_LLM_ALLOW_DIRTY from its children.
opencode-publish-first:
	uv run --project skills/multi-llm -- python $(OPENCODE_PKG)/first_release.py $(if $(DRY_RUN),--dry-run)

# NO LONGER A PUBLISH PATH. This target used to run the upload in the package
# directory with `--provenance`, and that shipped the wrong bytes: the
# manifest's `prepack` hook rebuilt dist/ there and npm uploaded whatever that
# rebuild produced, while `--check-stamp` had compared a separate rebuild a
# moment earlier. Nothing compared the uploaded tarball with the gated one. Both
# guarded paths now upload the exact tarball the full gate packed, scanned and
# loaded (`npm pack` once, then the file, with `--ignore-scripts`); the `prepack`
# hook stays only as the backstop for a hand-typed bare `npm pack`. Measured: a
# tarball upload runs no lifecycle scripts, so nothing can repack it.
opencode-publish:
	@echo "make: REFUSING: opencode-publish no longer uploads anything. The two guarded paths:" >&2; \
	echo "      * the FIRST release, from this machine: make opencode-publish-first" >&2; \
	echo "        (DRY_RUN=1 to run its preflights only)" >&2; \
	echo "      * every later release: the publish-opencode-plugin workflow dispatch" >&2; \
	echo "        (mode: publish, tag: opencode-v<version>), with provenance" >&2; \
	exit 1

# THE SUPPORTED OPENCODE IS ITS NEWEST RELEASE, AND ONLY THAT ONE. It is the
# adapter's min_version and verified_version at once, and engines.opencode, the
# package README, the container's opencode-ai pin, lockfile and cli-versions.lock
# row and the mock's --version all name it; tests/test_opencode_version_pins.py
# fails on any one that does not. `opencode-bump` is the one way to move them:
# it re-captures the fixtures from that release offline ($0, a localhost stub)
# and rewrites every pin, writing nothing before its npm steps succeeded
# (packaging/opencode/bump.py). `opencode-latest-check` is the drift probe: exit
# 1 when npm's latest opencode-ai is newer than the pin. A make target, not a
# scheduled workflow, like the container lane's own drift probes.
opencode-bump:
	@test -n "$(VERSION)" || { echo "usage: make opencode-bump VERSION=<x.y.z>   (make opencode-latest-check names the newest)"; exit 1; }
	uv run --project skills/multi-llm -- python $(OPENCODE_PKG)/bump.py set "$(VERSION)" $(if $(ALLOW_DOWNGRADE),--allow-downgrade)

opencode-latest-check:
	uv run --project skills/multi-llm -- python $(OPENCODE_PKG)/bump.py check

# The single registry, for the workflow and anything else outside this file.
opencode-print-registry:
	@echo $(OPENCODE_REGISTRY)

# THE SHARED-VERSION BUMP CHECK, OpenCode's entry point (plans/
# opencode-plugin-publish.md, step 2; Codex's `--check-bump` runs the same one).
# The version is pyproject.toml's, shared by all three distributions, so HEAD's
# version must either be released already WITH THIS SAME shared payload, or be
# newer than every codex-v* and opencode-v* tag and every version npm lists for
# the package (npm's own 0.0.0-stage stub, `"stub": true`, is reported and
# skipped). Reads git and the registry; writes nothing. A shallow or tagless
# clone is refused, never read as "no releases" — run it from a full clone with
# every tag.
#
# Deliberately in NO `tests` workflow leg and not in `opencode-package-check-ci`:
# after V is released, `master` may change shipped bytes while still saying V.
# This check refuses a RELEASE from that state, and must not turn `master` red
# over the legal edit. The full gate and the publish preflights call it.
opencode-check-bump:
	uv run --project skills/multi-llm -- python $(OPENCODE_PKG)/build.py --check-bump --registry $(OPENCODE_REGISTRY)

# THE REGISTRY-INSTALL PROBE, one implementation for two records (plans/
# opencode-plugin-publish.md, step 5, and step 0's retired bootstrap job): what npm SERVES
# for SPEC, installed by the pinned OpenCode from $(OPENCODE_REGISTRY) into a
# fresh home, compared with the candidate rebuilt from TAG's commit before the
# first tool call, then driven through `multi_llm` (packaging/opencode/
# registry_check.py). ENVIRONMENT=ci-runner writes the bootstrap record (seven
# criteria, the workflow run from $$GITHUB_RUN_ID or WORKFLOW_RUN_ID), the
# record of the workflow's bootstrap path, retired after opencode-v0.1.1; it
# covers a first release only, so every later tag refuses it. container
# drives a stock container from this host and adds the warm-offline run on its
# committed image with --network none; vm runs here and waits for the VM's
# interface to be detached. OUT is the record; its supporting output
# goes beside it. REHEARSAL_REGISTRY=http://127.0.0.1:<port> (a
# `registry_stub.py serve` of the gated tarball) makes the run a REHEARSAL: the
# record says so and is never release evidence. It is honoured ONLY from this
# make command line ($(origin) `command line`): an export left over from a
# rehearsal would otherwise turn step 5's real run into another rehearsal, so a
# value from the environment (or anywhere else) is refused by name. OPENCODE,
# PREVIOUS_PIN and REPO pass through. `--frozen --no-dev`: the probe needs only
# the runtime lock.
opencode-registry-check:
	@test -n "$(SPEC)" -a -n "$(TAG)" -a -n "$(ENVIRONMENT)" -a -n "$(OUT)" || { \
	  echo "usage: make opencode-registry-check SPEC=<name@version> TAG=opencode-v<version> ENVIRONMENT=ci-runner|container|vm OUT=<record.json> [REHEARSAL_REGISTRY=http://127.0.0.1:<port>] [WORKFLOW_RUN_ID=n] [OPENCODE=bin] [PREVIOUS_PIN=spec] [REPO=path]"; exit 1; }
	@if [ -n "$(REHEARSAL_REGISTRY)" ]; then case "$(origin REHEARSAL_REGISTRY)" in "command line") ;; *) \
	  echo "make: REFUSING: REHEARSAL_REGISTRY=$(REHEARSAL_REGISTRY) comes from the $(origin REHEARSAL_REGISTRY), not this make command line; a leftover export would turn this run into a rehearsal. Unset it, or pass REHEARSAL_REGISTRY=<url> on the command line for a rehearsal." >&2; \
	  exit 1 ;; esac; fi
	uv run --frozen --no-dev --project skills/multi-llm -- python $(OPENCODE_PKG)/registry_check.py \
	  --spec "$(SPEC)" --registry $(OPENCODE_REGISTRY) --tag "$(TAG)" --environment "$(ENVIRONMENT)" \
	  --out "$(OUT)" $(if $(and $(filter command line,$(origin REHEARSAL_REGISTRY)),$(REHEARSAL_REGISTRY)),--rehearsal-registry "$(REHEARSAL_REGISTRY)") \
	  $(if $(WORKFLOW_RUN_ID),--workflow-run-id "$(WORKFLOW_RUN_ID)") $(if $(OPENCODE),--opencode "$(OPENCODE)") \
	  $(if $(PREVIOUS_PIN),--previous-pin "$(PREVIOUS_PIN)") $(if $(REPO),--repo "$(REPO)")


# --------------------------------------------------------------------------- #
# The codex plugin.
#
# The tree is committed, so there is no pack and no upload: `codex-publish` ends
# at a tag push, and publication proper is the promote job in
# .github/workflows/publish-codex-plugin.yml fast-forwarding `release` — the ref
# the published install pins. The tag publishes nothing.
# --------------------------------------------------------------------------- #

codex-regen:
	$(CODEX_BUILD) --allow-dirty
	@echo "codex-regen: $(CODEX_PKG)/multi-llm rewritten; commit it WITH the source edit that implies it"

# `MULTI_LLM_ALLOW_DIRTY=` on the recipe line, not merely no flag: the variable
# is exported above whenever ALLOW_DIRTY is set, and this is the build the gate
# and the publish stand on, so no spelling of the opt-in may reach it.
codex-package:
	MULTI_LLM_ALLOW_DIRTY= $(CODEX_BUILD) --require-no-diff
	@echo "codex-package: clean rebuild equals the committed tree"

# The rows that need no Docker daemon and no codex binary: the plain rows of the
# codex suite, then every `package`-marked row with skips turned into failures.
# Writes NO stamp. `ALLOW_DIRTY=1` swaps the clean build for the regeneration,
# for a local run mid-edit; CI never sets it.
codex-package-check-ci:
	$(if $(ALLOW_DIRTY),$(MAKE) codex-regen,$(MAKE) codex-package)
	$(CODEX_PACKAGE_PYTEST) -m "not live and not container and not package" -q
	MLLM_REQUIRE_PACKAGE_ROWS=1 $(CODEX_PACKAGE_PYTEST) -m "package and not container" -q
	@echo "codex-package-check-ci: CI tier green (no stamp written)"

# The full A4 gate. IT STAMPS ONLY WHAT IT RAN. The CI tier, then the container
# rows on THREE codex builds — the container's pin, the plugin's declared
# minimum (`hosts.codex.package.min_host_version`) and npm's newest, resolved now
# (row 14) — then bump discipline, then the stamp. Every tier ends in
# `release_gate.py check`, which reads the run records this invocation wrote and
# refuses unless each row the tier requires came back graded `pass` on that
# image, that codex and an installed tree carrying this checkout's digest; and
# `--write-stamp` refuses without all three tiers' evidence over these bytes.
# pytest's exit code alone is not that: a module skipped for want of a daemon
# exits 0 too. Serial by construction — recipe lines, never prerequisites, so
# `make -j` cannot reorder the reset after a tier or race two tiers on the
# shared records directory and docker. The reset begins a gate SESSION and the
# stamp takes only evidence from it: every tier of one invocation, none older.
codex-package-check:
	$(MAKE) --no-print-directory codex-gate-reset
	$(MAKE) --no-print-directory codex-package-check-ci
	$(MAKE) --no-print-directory codex-gate-pinned
	$(MAKE) --no-print-directory codex-gate-minimum
	$(MAKE) --no-print-directory codex-gate-ceiling
	$(CODEX_BUILD) --check-bump
	@$(CODEX_GATE) summary
	$(CODEX_BUILD) --write-stamp "$$(date -u +%Y-%m-%dT%H:%M:%SZ)"
	@echo "codex-package-check: full gate green; stamp written to $(CODEX_STAMP)"

# One invocation's evidence, never an earlier one's: a new session id, and every
# earlier tier's evidence deleted, before any tier runs.
codex-gate-reset:
	@$(CODEX_GATE) begin

# Rows 1, 6b, 7 and 13 on the image `make e2e-image` builds from this tree: every
# package probe row, as the gate has always run them.
codex-gate-pinned: e2e-image
	@$(CODEX_GATE) preflight
	@since="$$(date -u +%Y%m%dT%H%M%SZ)"; \
	  version="$$($(CODEX_GATE) version pinned)" || exit 1; \
	  MLLM_CODEX_GATE_TIER=pinned MLLM_E2E_IMAGE=$(IMAGE) $(CODEX_GATE_PYTEST) && \
	  $(CODEX_GATE) check pinned --image $(IMAGE) --since "$$since" --version "$$version"

# The declared floor, measured rather than assumed: the pinned tier runs on the
# container's pin, which is not the minimum users are told they need. Under this
# tier the lane holds the rows to `min_host_version` instead of host_matrix's
# `plugin_floor` (which tracks the pin and would refuse them).
codex-gate-minimum: e2e-image
	@version="$$($(CODEX_GATE) version minimum)" || exit 1; \
	  $(MAKE) --no-print-directory codex-gate-derived TIER=minimum \
	    CODEX_VERSION="$$version" CODEX_IMAGE=$(CODEX_MINIMUM_IMAGE)

# A4 row 14, the ceiling: npm's newest @openai/codex, asked NOW. A fetch that
# fails stops here naming what it could not reach — never a skip.
codex-gate-ceiling: e2e-image
	@version="$$($(CODEX_GATE) version ceiling)" || exit 1; \
	  echo "codex-gate-ceiling: npm's newest @openai/codex is $$version"; \
	  $(MAKE) --no-print-directory codex-gate-derived TIER=ceiling \
	    CODEX_VERSION="$$version" CODEX_IMAGE=$(CODEX_CEILING_IMAGE)

# The pinned image with only codex moved (Dockerfile.codex-version: built with no
# context, the fetch is this build's one network access), then the tier's rows
# on it. Internal: the two targets above are what to type.
codex-gate-derived:
	@test -n "$(TIER)" && test -n "$(CODEX_VERSION)" && test -n "$(CODEX_IMAGE)" || { \
	  echo "usage: make codex-gate-minimum | codex-gate-ceiling (sets TIER, CODEX_VERSION, CODEX_IMAGE)"; exit 1; }
	@$(CODEX_GATE) preflight
	docker build --platform $(PLATFORM) \
	  --build-arg BASE_IMAGE=$(IMAGE) --build-arg CODEX_VERSION=$(CODEX_VERSION) \
	  -t $(CODEX_IMAGE) - < $(CODEX_VERSION_DOCKERFILE)
	@since="$$(date -u +%Y%m%dT%H%M%SZ)"; \
	  MLLM_CODEX_GATE_TIER=$(TIER) MLLM_E2E_IMAGE=$(CODEX_IMAGE) $(CODEX_GATE_PYTEST) && \
	  $(CODEX_GATE) check $(TIER) --image $(CODEX_IMAGE) --since "$$since" --version $(CODEX_VERSION)

# Developer-facing, never a CI target: CI holds no credential that pushes a tag.
# Refuses before tagging on a missing or stale stamp (rule i) and on a digest
# that moved since the newest codex-v* tag under an unmoved version (A4 row 10).
codex-publish:
	$(CODEX_BUILD) --check-stamp
	$(CODEX_BUILD) --check-bump
	@version="$$($(CODEX_BUILD) --print-version)"; tag="codex-v$$version"; \
	if git rev-parse -q --verify "refs/tags/$$tag" >/dev/null; then \
	  echo "make: REFUSING (row 9): $$tag already exists. Bump skills/multi-llm/pyproject.toml." >&2; \
	  exit 1; \
	fi; \
	git tag -a "$$tag" -m "codex plugin $$version" && \
	git push origin HEAD "$$tag" && \
	echo "codex-publish: pushed $$tag. The tag publishes nothing; the promote job moves 'release'."
