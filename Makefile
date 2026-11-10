# Developer entry points: the containerized E2E harness, the three package gates and
# the release helpers. Run `make help` for the list; every target carries a `## ...`
# description that `help` is generated from. The rules and their reasons live in
# docs/development/release-gates.md; the comments here state only the one rule beside
# each recipe.
#
# This file sits at the repository root because the Docker build context is the root,
# and it never writes `docker run` flags: tests/container/harness.py builds those per
# row from tests/container/posture.yaml (release-gates.md#why-the-makefile-sits-at-the-repository-root).

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

CONTAINER_DIR := skills/multi-llm/tests/container
DOCKERFILE    := $(CONTAINER_DIR)/Dockerfile
# Image tags carry no registry component, so a bare `docker push` fails by construction.
IMAGE         := mllm-e2e:local
IMAGE_LATEST  := mllm-e2e:latest-probe
PLATFORM ?= linux/amd64

PYTEST := uv run --project skills/multi-llm -- pytest skills/multi-llm/tests
# The unpinned host lane's credential guards, run before its image build and after its pytest.
LANE_GUARD := uv run --project skills/multi-llm -- python $(CONTAINER_DIR)/lane_guard.py

# Advisory: the map is a static import scan and never replaces the CI command.
TESTS_FOR := uv run --project skills/multi-llm -- python skills/multi-llm/tests/tools/tests_for.py
TESTS_FOR_TOP := $(if $(TOP),--top $(TOP),)
TESTS_FOR_MARKERS := not live and not container and not package and not codex_jail

# --- The opencode npm package (packaging/opencode/) ----------------------------- #
OPENCODE_PKG   := packaging/opencode
# The one registry: every npm call on the release path passes it explicitly, and
# packaging/opencode/registry.py reads this line. Nothing else may define the URL.
OPENCODE_REGISTRY := https://registry.npmjs.org
# ALLOW_DIRTY=1 is a named opt-in for development runs; opencode-publish never honours it.
# Exported because `npm pack` runs the manifest's `prepack` hook with its own command line.
export MULTI_LLM_ALLOW_DIRTY := $(if $(ALLOW_DIRTY),1,)
OPENCODE_DIRTY := $(if $(ALLOW_DIRTY),--allow-dirty,)
OPENCODE_BUILD := uv run --project skills/multi-llm -- python $(OPENCODE_PKG)/build.py $(OPENCODE_DIRTY)
# The build with every spelling of the dirty opt-in removed; `env` so it also works as a gate step's argv.
OPENCODE_BUILD_CLEAN := env MULTI_LLM_ALLOW_DIRTY= uv run --project skills/multi-llm -- python $(OPENCODE_PKG)/build.py
OPENCODE_STAMP := $(OPENCODE_PKG)/.package-check.stamp
# Exact name from the manifest, never `ls -t | head -1`: a stale tarball must not be scanned or stamped.
OPENCODE_TARBALL = $(OPENCODE_PKG)/$$(node -p "const m = require('./$(OPENCODE_PKG)/package.json'); m.name + '-' + m.version + '.tgz'")
# The gate's contract is packaging/opencode/release_gate.py; evidence lives under OPENCODE_GATE_DIR (gitignored).
OPENCODE_GATE := uv run --project skills/multi-llm -- python $(OPENCODE_PKG)/release_gate.py
OPENCODE_GATE_DIR := $(OPENCODE_PKG)/.gate
OPENCODE_GATE_STEP := $(OPENCODE_GATE) step
# Its own image tag: `mllm-e2e:local` is whatever the last `make e2e-image` staged, dirty tree included.
OPENCODE_GATE_IMAGE := mllm-e2e:opencode-gate
# Row 8: the package-route scenarios only, every key unset, and the lane told not to prune the bundles check-row8 verifies.
OPENCODE_GATE_PYTEST := env -u OPENROUTER_API_KEY -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN \
    -u OPENAI_API_KEY MLLM_E2E_RUNS_KEEP=0 uv run --project skills/multi-llm -- pytest \
    skills/multi-llm/tests/test_container_opencode_package_gate.py -q -rs \
    -m "container and not live and not live_provider"
# The package rows only: `$(PYTEST)` names the whole tests directory, so appending files to it would collect the suite twice.
PACKAGE_PYTEST := uv run --project skills/multi-llm -- pytest \
    skills/multi-llm/tests/test_opencode_package.py \
    skills/multi-llm/tests/test_uv_candidates.py

# Which plain rows the three `-ci` tiers run: `all` (publish workflows, full gates) or
# `graded` (the PR package job, which keeps only the rows a grader reads). Anything else is refused.
PLAIN_ROWS ?= all
ifneq ($(PLAIN_ROWS),all)
ifneq ($(PLAIN_ROWS),graded)
$(error PLAIN_ROWS must be `all` or `graded`, not `$(PLAIN_ROWS)`)
endif
endif
PLAIN_GRADED := $(filter graded,$(PLAIN_ROWS))

# --- The codex plugin (packaging/codex/) ---------------------------------------- #
# The built tree packaging/codex/multi-llm/ is committed. `codex-regen` runs before the commit
# on dirty inputs; `codex-package` is the clean rebuild after it (release-gates.md#the-codex-and-claude-code-plugin-trees).
CODEX_PKG   := packaging/codex
CODEX_BUILD := uv run --project skills/multi-llm -- python $(CODEX_PKG)/build.py
CODEX_STAMP := $(CODEX_PKG)/.package-check.stamp
# The codex route's own rows, not PACKAGE_PYTEST (which names the opencode suite).
CODEX_PACKAGE_PYTEST := uv run --project skills/multi-llm -- pytest \
    skills/multi-llm/tests/test_codex_package.py \
    skills/multi-llm/tests/test_codex_mcp_server.py \
    skills/multi-llm/tests/test_uv_candidates.py
# --- The Claude Code plugin (packaging/claude/) --------------------------------- #
# Committed like the codex tree, with the same two builds.
CLAUDE_PKG   := packaging/claude
CLAUDE_BUILD := uv run --project skills/multi-llm -- python $(CLAUDE_PKG)/build.py
CLAUDE_PACKAGE_PYTEST := uv run --project skills/multi-llm -- pytest \
    skills/multi-llm/tests/test_claude_package.py \
    skills/multi-llm/tests/test_claude_package_seed.py \
    skills/multi-llm/tests/test_claude_release_helpers.py \
    skills/multi-llm/tests/test_bump_version.py \
    skills/multi-llm/tests/test_marketplace_manifest_location.py \
    skills/multi-llm/tests/test_uv_candidates.py
# The full gate's container half; release_gate.py owns which rows each tier requires.
CODEX_GATE := uv run --project skills/multi-llm -- python $(CODEX_PKG)/release_gate.py
# One bounded tier step: its own session with no terminal, SIGTERM then SIGKILL at the bound,
# output kept in $(CODEX_PKG)/.gate/logs/<step>.log.
CODEX_GATE_STEP := $(CODEX_GATE) step
CODEX_VERSION_DOCKERFILE := skills/multi-llm/tests/container/Dockerfile.codex-version
CODEX_MINIMUM_IMAGE := mllm-e2e:codex-minimum
CODEX_CEILING_IMAGE := mllm-e2e:codex-ceiling
# The zero-quota package rows only: every key unset, no live or credentialed tier selected.
CODEX_GATE_PYTEST := env -u OPENROUTER_API_KEY -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN \
    -u OPENAI_API_KEY uv run --project skills/multi-llm -- pytest \
    skills/multi-llm/tests/test_container_host_lane.py -q -rs \
    -m "container and not live and not live_provider and not live_native_phase and not live_next_command" \
    -k "codex-package or plugin_floor or gate_tier"

# Whole-lane wall-clock cap, enforced by the host-side driver (harness.py MatrixBudget).
MLLM_E2E_MATRIX_TIMEOUT_S ?= 3600
export MLLM_E2E_MATRIX_TIMEOUT_S

.PHONY: help e2e-preflight e2e-image e2e-image-latest e2e-container e2e-container-host \
        e2e-container-host-phases e2e-container-host-next-command \
        e2e-container-host-discovery e2e-container-host-invocation \
        e2e-version-delta e2e-container-latest \
        e2e-container-latest-host e2e-live-provider e2e-lock-update e2e-verify-record \
        e2e-clean e2e-prune \
        opencode-stage opencode-package opencode-package-check-ci \
        opencode-package-check opencode-publish opencode-bump opencode-latest-check \
        opencode-print-registry opencode-check-bump opencode-registry-check \
        opencode-registry-absent \
        codex-regen codex-package codex-package-check-ci codex-package-check \
        codex-gate-reset codex-gate-pinned codex-gate-minimum codex-gate-ceiling \
        codex-gate-derived codex-publish \
        claude-regen claude-package claude-check-release claude-check-bump \
        claude-package-check-ci claude-package-container \
        bump-version claude-publish claude-hook-install claude-hook-uninstall \
        test-for tests-map lint refresh-model-catalogs

# `make help` lists every `target: ... ## description` line under its `##@` heading.
# A new target needs a description and a .PHONY entry; tests/test_makefile_help.py checks both.
help: ## list the targets, grouped by section
	@awk 'BEGIN {FS = ":.*## "} /^##@/ {printf "\n%s\n\n", substr($$0, 5); next} /^[a-zA-Z0-9_.-]+:.*## / {printf "  make %-32s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

##@ Tests

test-for: ## list the test files that import MODULE (a path under skills/multi-llm; optional TOP), then run only those at -n 4
	@test -n "$(MODULE)" || { echo "usage: make test-for MODULE=utils/foo.py [TOP=n]" >&2; exit 2; }
	@$(TESTS_FOR) $(TESTS_FOR_TOP) "$(MODULE)"
	@files=$$($(TESTS_FOR) --paths $(TESTS_FOR_TOP) "$(MODULE)"); \
	  if [ -z "$$files" ]; then echo "no test file imports $(MODULE); nothing to run" >&2; exit 0; fi; \
	  uv run --project skills/multi-llm -- pytest $$files -n 4 --dist loadfile \
	    -m "$(TESTS_FOR_MARKERS)" -q -p no:cacheprovider

tests-map: ## regenerate skills/multi-llm/tests/MODULE_TESTS.md
	@$(TESTS_FOR) --write

# The lint gate CI runs on its Linux leg. The config is passed by path because packaging/ sits
# outside the project directory, where ruff would not find it.
lint: ## ruff check of the skill and packaging sources: undefined names, unused imports, bug-class rules
	@uv run --project skills/multi-llm -- ruff check --config skills/multi-llm/pyproject.toml skills/multi-llm packaging

##@ Containerized E2E harness (skills/multi-llm/tests/container/README.md)

# One line naming what is missing, rather than a page of CLI errors.
e2e-preflight: ## check that docker is on PATH and its daemon answers
	@command -v docker >/dev/null 2>&1 || { \
	  echo "make: docker is not on PATH — install Docker or run the non-container suite with 'pytest -m \"not container\"'"; exit 1; }
	@docker version >/dev/null 2>&1 || { \
	  echo "make: the docker CLI is present but no daemon is reachable — start Docker and retry"; exit 1; }

# Staging only: no `npm pack`, no sweep, no stamp, so a test image never looks like a step toward
# publishing. A prerequisite of the image targets because dist/ and skills/ under the package are
# gitignored build output. Dirty by default (a test image matches the working tree); the full
# gate overrides OPENCODE_STAGE_DIRTY (release-gates.md#pin-updates-and-housekeeping).
OPENCODE_STAGE_DIRTY ?= --allow-dirty
opencode-stage: ## stage the opencode plugin payload the image COPYs (a build input, never a publish step)
	@MULTI_LLM_ALLOW_DIRTY= uv run --project skills/multi-llm -- python $(OPENCODE_PKG)/build.py $(OPENCODE_STAGE_DIRTY) >/dev/null
	@echo "opencode-stage: payload staged at $(OPENCODE_PKG)/{dist,skills}$(if $(OPENCODE_STAGE_DIRTY),, from a clean tree)"

e2e-image: e2e-preflight opencode-stage ## build the pinned image
	docker build \
	  --platform $(PLATFORM) \
	  --build-arg MLLM_PIN_MODE=pinned \
	  -f $(DOCKERFILE) \
	  -t $(IMAGE) \
	  .

# `not live_provider` is a spend guard: those tests carry `container` but not `live`, and this
# recipe inherits the caller's environment (release-gates.md#container-lanes-and-spend-guards).
e2e-container: e2e-image ## pinned build; provider lane with no secrets and --network none
	$(PYTEST) -m "container and not live and not live_provider"

# `not live_native_phase` is a spend guard: native Task dispatches bill outside /opt/mllm/bin.
# No key is demanded: codex and claude-code run on the subscription when its credential file exists.
e2e-container-host: e2e-image ## pinned build; host lane (codex and claude-code on the subscription, OPENROUTER_API_KEY for opencode)
	@test -n "$${OPENROUTER_API_KEY:-}" || \
	  echo "make: OPENROUTER_API_KEY unset: codex and claude-code run on the subscription when its credential file exists; opencode rows skip (named, in the census)."
	OPENROUTER_API_KEY=$$OPENROUTER_API_KEY $(PYTEST) -m "container and live and not live_native_phase and not live_next_command"

# One workflow phase per row. A separate target for cost (the default 2.00 budget would leave a
# half-covered tier) and wall clock; raise either bound only on a recorded overrun.
e2e-container-host-phases: e2e-image ## host lane, claude-code per-phase tier; EXPENSIVE (about 3 to 5 USD per pass)
	@test -n "$${OPENROUTER_API_KEY:-}" || { \
	  echo "make: the host phase tier needs OPENROUTER_API_KEY (a DEDICATED key carrying a key-level spend limit)."; \
	  echo "      It is the EXPENSIVE tier: budget ~\$$3-5 per green pass, and bring-up separately."; exit 1; }
	OPENROUTER_API_KEY=$$OPENROUTER_API_KEY \
	MLLM_E2E_BUDGET_USD="$${MLLM_E2E_BUDGET_USD:-6.00}" \
	MLLM_E2E_MATRIX_TIMEOUT_S="$${MLLM_E2E_MATRIX_TIMEOUT_S:-3600}" \
	  $(PYTEST) -m "container and live and live_native_phase"

# The next-command live check: cheap but paid, so its own tier. One row is `-k '[<row id>]'` on this expression.
e2e-container-host-next-command: e2e-image ## host lane, next-command rows (needs OPENROUTER_API_KEY)
	@test -n "$${OPENROUTER_API_KEY:-}" || { \
	  echo "make: the next-command tier needs OPENROUTER_API_KEY (a DEDICATED key carrying a key-level spend limit)."; exit 1; }
	OPENROUTER_API_KEY=$$OPENROUTER_API_KEY \
	MLLM_E2E_BUDGET_USD="$${MLLM_E2E_BUDGET_USD:-1.00}" \
	  $(PYTEST) -m "container and live and live_next_command"

# Free tier: no credential, no route out. Its module carries `container` alone, so `-m "container and live"` would hide it.
e2e-container-host-discovery: e2e-image ## host lane, free tier: stage the shim and list the command (no key, no network)
	env -u OPENROUTER_API_KEY -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN \
	  $(PYTEST)/test_container_host_lane.py -m "container and not live"

# Free tier: types the published `/multi-llm …` string into a fresh session on the loopback stub.
# What is under test is argument delivery, so the stub is no compromise.
e2e-container-host-invocation: e2e-image ## host lane, free tier: type the documented /multi-llm string into a fresh session
	env -u OPENROUTER_API_KEY -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN \
	  $(PYTEST)/test_container_documented_invocation.py -m "container and not live"

##@ Drift probe: the CLI layer at @latest, every other pin held

e2e-image-latest: e2e-preflight opencode-stage ## build the image with every CLI at @latest
	docker build \
	  --platform $(PLATFORM) \
	  --build-arg MLLM_PIN_MODE=latest \
	  -f $(DOCKERFILE) \
	  -t $(IMAGE_LATEST) \
	  .

# The one place this file writes `docker run` flags: it is not a lane row, and records nothing durable.
e2e-version-delta: e2e-preflight ## print the pinned to installed versions of the @latest image
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

# Never touches a credential: unpinned binaries must not meet a key. The drift report is the
# printed version delta, the surface probe (`-rs` keeps its skip reasons) and the parse
# assertions; there is no pytest drift flag (release-gates.md#the-unpinned-image-and-credentials).
e2e-container-latest: e2e-image-latest ## CLIs at @latest; provider lane only, credential-free by construction
	@$(MAKE) --no-print-directory e2e-version-delta
	env -u OPENROUTER_API_KEY -u ANTHROPIC_API_KEY -u OPENAI_API_KEY -u GEMINI_API_KEY \
	  MLLM_E2E_IMAGE=$(IMAGE_LATEST) \
	  $(PYTEST) -m "container and not live and not live_provider" -rs

# The one place an unpinned image is allowed near a key, so the guard is a target-level refusal:
# the key is required up front (a keyless run would build the image and skip every row), vendor
# keys are scrubbed rather than refused, the version delta prints before the first row, and a
# lane where no row executed fails after pytest. A missing key costs no image.
e2e-container-latest-host: e2e-preflight ## CLIs at @latest; host lane; guarded, opt-in, needs OPENROUTER_API_KEY
	@$(LANE_GUARD) unpinned-host
	$(MAKE) e2e-image-latest
	@$(MAKE) --no-print-directory e2e-version-delta
	@mkdir -p $(CONTAINER_DIR)/records
	stamp="$$(mktemp)"; trap 'rm -f "$$stamp"' EXIT; \
	OPENROUTER_API_KEY=$$OPENROUTER_API_KEY \
	env -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN -u OPENAI_API_KEY -u GEMINI_API_KEY \
	  MLLM_E2E_IMAGE=$(IMAGE_LATEST) \
	  $(PYTEST) -m "container and live" -rs; \
	$(LANE_GUARD) rows-ran --records $(CONTAINER_DIR)/records --since "$$stamp"

##@ The credentialed provider lane

# Contract: a prerequisite of nothing (tests/test_makefile_lanes.py asserts it by parsing this file),
# kept out of CI by its marker, and it never builds the image, so a stale image is a visible failure.
# The key is read host-side from ~/.config/multi-llm/e2e.env and reaches the container only by name.
# `-k '[<id>]'` selects the parametrize id; a bare `-k pi` also matches a skipif keyword.
e2e-live-provider: ## credentialed PROVIDER lane (CLI selects one row); SPENDS CREDITS on openrouter.ai
	set -a; . $${MULTI_LLM_E2E_ENV:-$$HOME/.config/multi-llm/e2e.env}; set +a; \
	uv run --project skills/multi-llm -- pytest \
	  skills/multi-llm/tests/test_container_live_provider.py \
	  -m 'container and live_provider' $(if $(CLI),-k '[$(CLI)]',)

##@ Pins, provenance and housekeeping

# The pin writer, and the only target here that edits tracked files. It moves what the @latest
# image proves, refuses the rows it cannot substantiate, and verifies the rewrite by rebuilding the
# pinned image (release-gates.md#pin-updates-and-housekeeping).
LOCK_UPDATE := uv run --project skills/multi-llm -- python $(CONTAINER_DIR)/lock_update.py

e2e-lock-update: e2e-preflight ## move the pins the @latest image proves moved, then prove it by rebuilding the pinned image
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

e2e-verify-record: ## REC names a runs/ bundle: re-hash it against its run record
	@test -n "$(REC)" || { echo "usage: make e2e-verify-record REC=$(CONTAINER_DIR)/records/<utc>-<host>.json"; exit 1; }
	uv run --project skills/multi-llm -- python $(CONTAINER_DIR)/verify_record.py "$(REC)"

# runs/ is ephemeral raw evidence, but gitignored is not absent: anything that copies a working
# tree ignores .gitignore, so an uncapped runs/ is a distribution problem. The lane prunes
# itself per row; this is the same sweep on demand (records/ is untouched).
KEEP ?= 8
e2e-prune: ## keep the newest KEEP runs/ bundles (default 8) and delete the rest
	uv run --project skills/multi-llm -- python -c \
	  'import sys; sys.path.insert(0, "skills/multi-llm"); \
	   from tests.container.harness import prune_runs; \
	   r = prune_runs(keep=$(KEEP)); \
	   print(f"pruned {len(r)} bundle(s), kept the newest $(KEEP) (records/ untouched)")'

e2e-clean: ## prune every ephemeral runs/ bundle
	rm -rf $(CONTAINER_DIR)/runs/*/
	@echo "pruned $(CONTAINER_DIR)/runs/ (records/ untouched)"

##@ OpenCode npm package (packaging/opencode/README.md)

# Build and pack are separate: build.py never invokes npm, and the tarball secret sweep runs after
# `npm pack` because what npm writes is what ships. Every recipe runs the build explicitly, since the
# manifest's `prepack` hook does not fire under `--ignore-scripts`; test_package_is_built_before_every_pack
# asserts the order (release-gates.md#the-opencode-package-gate).
opencode-package: ## build, pack, then sweep the tarball for secrets
	$(OPENCODE_BUILD)
	cd $(OPENCODE_PKG) && npm pack
	$(OPENCODE_BUILD) --scan-tarball "$(OPENCODE_TARBALL)"

# Rows 1-6 and 5b: no Docker, no registry, no opencode binary. Writes no stamp. Graded row by row
# from junit XML (`release_gate.py grade-ci`), never by exit code; both selections always run.
# With PLAIN_ROWS=graded the plain selection is exactly the nodes the rows name.
opencode-package-check-ci: ## rows 1-6 and 5b, graded test by test: no Docker, no registry, no opencode binary
	$(OPENCODE_BUILD)
	cd $(OPENCODE_PKG) && npm pack --dry-run
	@mkdir -p $(OPENCODE_GATE_DIR)/junit
	@rm -f $(OPENCODE_GATE_DIR)/junit/ci-package-rows.xml $(OPENCODE_GATE_DIR)/junit/ci-plain-rows.xml
	rc=0; \
	$(if $(PLAIN_GRADED),plain="$$($(OPENCODE_GATE) ci-nodes)"; test -n "$$plain";) \
	MLLM_REQUIRE_PACKAGE_ROWS=1 $(PYTEST) -m "package and not container" -q \
	  --junitxml=$(OPENCODE_GATE_DIR)/junit/ci-package-rows.xml || rc=$$?; \
	$(if $(PLAIN_GRADED),uv run --project skills/multi-llm -- pytest $$plain,$(PACKAGE_PYTEST)) \
	  -m "not live and not container and not package" -q \
	  --junitxml=$(OPENCODE_GATE_DIR)/junit/ci-plain-rows.xml || rc=$$?; \
	$(OPENCODE_GATE) grade-ci --junit $(OPENCODE_GATE_DIR)/junit/ci-package-rows.xml \
	  --junit $(OPENCODE_GATE_DIR)/junit/ci-plain-rows.xml || rc=$$?; \
	exit $$rc
	@echo "opencode-package-check-ci: rows 1-6 and 5b graded green (no stamp written)"

# The full gate: rows 1-6 and 5b, the clean-install probe (7), the package-route container
# scenarios (8) and the registry-absence check (9), then bump discipline, the stamp and the record.
# It stamps only what it ran, in this run: `begin` opens a session and `--write-stamp` refuses
# unless every row passed in it over these bytes. Steps are recipe lines, never prerequisites, so
# `make -j` cannot reorder them; each runs under a wall-clock bound. ALLOW_DIRTY=1 makes a
# rehearsal that the stamp and the exported record both refuse. The stamp never travels
# (release-gates.md#the-opencode-package-gate).
opencode-package-check: ## the FULL gate: rows 7-9 too, in one session; stamps and exports its record only if all passed
	@$(OPENCODE_GATE) begin $(if $(ALLOW_DIRTY),--allow-dirty,)
	$(OPENCODE_GATE_STEP) ci-rows --timeout 3600 -- $(MAKE) --no-print-directory opencode-package-check-ci PLAIN_ROWS=all
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

# Row 9's one entry point (packaging/opencode/registry_absent.mjs): only npm's `E404` means the
# version is absent. The publishing workflow's jobs run the same script: the two tiers still run
# separately, but they share one implementation rather than two hand-kept copies. `PACKAGE_JSON`
# is honoured only from this make command line; one from the environment is refused by name.
opencode-registry-absent: ## row 9: the version must be absent from the single registry (OUT keeps the evidence)
	@if [ -n "$(PACKAGE_JSON)" ]; then case "$(origin PACKAGE_JSON)" in "command line") ;; *) \
	  echo "make: REFUSING: PACKAGE_JSON=$(PACKAGE_JSON) comes from the $(origin PACKAGE_JSON), not this make command line; row 9 asks about $(OPENCODE_PKG)/package.json unless PACKAGE_JSON=<file> is passed explicitly. Unset it." >&2; \
	  exit 1 ;; esac; fi
	node $(OPENCODE_PKG)/registry_absent.mjs --registry $(OPENCODE_REGISTRY) $(if $(OUT),--out "$(OUT)") \
	  $(if $(and $(filter command line,$(origin PACKAGE_JSON)),$(PACKAGE_JSON)),--package-json "$(PACKAGE_JSON)")

# No longer a publish path: the guarded path uploads the exact tarball its gate packed, scanned
# and loaded, from the publish workflow. docs/development/hosts.md says where 0.1.1's record lives.
opencode-publish: ## refuses: names the guarded publish path
	@echo "make: REFUSING: opencode-publish no longer uploads anything. The guarded path:" >&2; \
	echo "      the publish-opencode-plugin workflow dispatch" >&2; \
	echo "      (mode: publish, tag: opencode-v<version>), with provenance" >&2; \
	exit 1

# The supported OpenCode is its newest release, and only that one: bump moves every pin together
# (tests/test_opencode_version_pins.py fails on a stray one).
opencode-bump: ## VERSION names the release: move every opencode pin to it and re-capture its fixtures offline (ALLOW_DOWNGRADE for an older one)
	@test -n "$(VERSION)" || { echo "usage: make opencode-bump VERSION=<x.y.z>   (make opencode-latest-check names the newest)"; exit 1; }
	uv run --project skills/multi-llm -- python $(OPENCODE_PKG)/bump.py set "$(VERSION)" $(if $(ALLOW_DOWNGRADE),--allow-downgrade)

opencode-latest-check: ## exit 1 when npm has a newer opencode than the pin
	uv run --project skills/multi-llm -- python $(OPENCODE_PKG)/bump.py check

# The single registry, for the workflow and anything else outside this file.
opencode-print-registry: ## print the one registry URL
	@echo $(OPENCODE_REGISTRY)

# The shared-version bump check (Codex's `--check-bump` runs the same one): reads git and the
# registry, writes nothing, refuses a shallow or tagless clone. In no `tests` workflow leg on purpose:
# after V is released master may change shipped bytes while still saying V.
opencode-check-bump: ## the shared-version check: HEAD against every claude-v*/codex-v*/opencode-v* tag and npm's versions
	uv run --project skills/multi-llm -- python $(OPENCODE_PKG)/build.py --check-bump --registry $(OPENCODE_REGISTRY)

# The registry-install probe: what npm serves for SPEC, installed by the pinned OpenCode into a
# fresh home. REHEARSAL_REGISTRY is honoured only from the make command line (a leftover export
# would turn a real run into a rehearsal). `--frozen --no-dev`: the probe needs only the runtime lock.
opencode-registry-check: ## SPEC TAG ENVIRONMENT OUT name the probe (usage line on error): install what npm serves with the pinned OpenCode
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

##@ Codex plugin, checked in at packaging/codex/multi-llm (packaging/codex/README.md)

# No pack and no upload: the tree is committed and `codex-publish` ends at a tag push; the promote
# job in .github/workflows/publish-codex-plugin.yml moves `release`.
codex-regen: ## regenerate the tracked plugin tree from dirty inputs (before the commit)
	$(CODEX_BUILD) --allow-dirty
	@echo "codex-regen: $(CODEX_PKG)/multi-llm rewritten; commit it WITH the source edit that implies it"

# `MULTI_LLM_ALLOW_DIRTY=` on the recipe line: no spelling of the opt-in may reach the clean build.
codex-package: ## clean rebuild; must equal the committed tree (after the commit)
	MULTI_LLM_ALLOW_DIRTY= $(CODEX_BUILD) --require-no-diff
	@echo "codex-package: clean rebuild equals the committed tree"

# Writes no stamp. ALLOW_DIRTY=1 swaps the clean build for the regeneration (a local run mid-edit;
# CI never sets it). PLAIN_ROWS=graded leaves the plain rows to the `test` job.
codex-package-check-ci: ## the rows that need no Docker and no codex binary; no stamp
	$(if $(ALLOW_DIRTY),$(MAKE) codex-regen,$(MAKE) codex-package)
	$(if $(PLAIN_GRADED),@echo "codex-package-check-ci: plain rows left to the test job (PLAIN_ROWS=graded)",$(CODEX_PACKAGE_PYTEST) -m "not live and not container and not package" -q)
	MLLM_REQUIRE_PACKAGE_ROWS=1 $(CODEX_PACKAGE_PYTEST) -m "package and not container" -q
	@echo "codex-package-check-ci: CI tier green (no stamp written)"

# The full gate stamps only what it ran: the CI tier, container rows on three codex builds (the pin,
# the declared minimum, npm's newest), bump discipline, then the stamp. Each tier ends in
# `release_gate.py check`, and the tiers are serial recipe lines under a gate session
# (release-gates.md#the-codex-and-claude-code-plugin-trees).
codex-package-check: ## the FULL gate; stamps only if every row ran
	$(MAKE) --no-print-directory codex-gate-reset
	$(MAKE) --no-print-directory codex-package-check-ci PLAIN_ROWS=all
	$(MAKE) --no-print-directory codex-gate-pinned
	$(MAKE) --no-print-directory codex-gate-minimum
	$(MAKE) --no-print-directory codex-gate-ceiling
	$(CODEX_BUILD) --check-bump
	@$(CODEX_GATE) summary
	$(CODEX_BUILD) --write-stamp "$$(date -u +%Y-%m-%dT%H:%M:%SZ)"
	@echo "codex-package-check: full gate green; stamp written to $(CODEX_STAMP)"

# A new session id, and every earlier tier's evidence deleted, before any tier runs.
codex-gate-reset: ## begin a gate session (internal: codex-package-check runs it)
	@$(CODEX_GATE) begin

# Rows 1, 6b, 7 and 13 on the image `make e2e-image` builds; the image is a bounded step so a hung build ends at its bound.
codex-gate-pinned: ## the gate's container rows on the pinned codex (rows 1, 6b, 7, 13)
	$(CODEX_GATE_STEP) pinned-image --timeout 3600 -- $(MAKE) --no-print-directory e2e-image
	@$(CODEX_GATE) preflight
	@since="$$(date -u +%Y%m%dT%H%M%SZ)"; \
	  version="$$($(CODEX_GATE) version pinned)" || exit 1; \
	  MLLM_CODEX_GATE_TIER=pinned MLLM_E2E_IMAGE=$(IMAGE) \
	    $(CODEX_GATE_STEP) pinned-rows --timeout 5400 -- $(CODEX_GATE_PYTEST) && \
	  $(CODEX_GATE) check pinned --image $(IMAGE) --since "$$since" --version "$$version"

# The declared floor, measured rather than assumed: the pinned tier runs on the container's pin, not the minimum users are told they need.
codex-gate-minimum: ## the same rows on the declared minimum codex
	$(CODEX_GATE_STEP) minimum-base-image --timeout 3600 -- $(MAKE) --no-print-directory e2e-image
	@version="$$($(CODEX_GATE) version minimum)" || exit 1; \
	  $(MAKE) --no-print-directory codex-gate-derived TIER=minimum \
	    CODEX_VERSION="$$version" CODEX_IMAGE=$(CODEX_MINIMUM_IMAGE)

# Row 14: npm's newest @openai/codex, asked now. A failed fetch stops here naming what it could not reach, never a skip.
codex-gate-ceiling: ## the same rows on npm's newest codex (row 14; fetches it)
	$(CODEX_GATE_STEP) ceiling-base-image --timeout 3600 -- $(MAKE) --no-print-directory e2e-image
	@version="$$($(CODEX_GATE) version ceiling)" || exit 1; \
	  echo "codex-gate-ceiling: npm's newest @openai/codex is $$version"; \
	  $(MAKE) --no-print-directory codex-gate-derived TIER=ceiling \
	    CODEX_VERSION="$$version" CODEX_IMAGE=$(CODEX_CEILING_IMAGE)

# The pinned image with only codex moved (Dockerfile.codex-version, built with no context), then the tier's rows on it.
codex-gate-derived: ## build a tier's codex image and run its rows (internal: the two targets above set its variables)
	@test -n "$(TIER)" && test -n "$(CODEX_VERSION)" && test -n "$(CODEX_IMAGE)" || { \
	  echo "usage: make codex-gate-minimum | codex-gate-ceiling (sets TIER, CODEX_VERSION, CODEX_IMAGE)"; exit 1; }
	@$(CODEX_GATE) preflight
	$(CODEX_GATE_STEP) $(TIER)-image --timeout 1800 --stdin $(CODEX_VERSION_DOCKERFILE) -- \
	  docker build --platform $(PLATFORM) \
	  --build-arg BASE_IMAGE=$(IMAGE) --build-arg CODEX_VERSION=$(CODEX_VERSION) \
	  -t $(CODEX_IMAGE) -
	@since="$$(date -u +%Y%m%dT%H%M%SZ)"; \
	  MLLM_CODEX_GATE_TIER=$(TIER) MLLM_E2E_IMAGE=$(CODEX_IMAGE) \
	    $(CODEX_GATE_STEP) $(TIER)-rows --timeout 5400 -- $(CODEX_GATE_PYTEST) && \
	  $(CODEX_GATE) check $(TIER) --image $(CODEX_IMAGE) --since "$$since" --version $(CODEX_VERSION)

# Developer-facing, never CI: CI holds no credential that pushes a tag. Refuses on a missing or
# stale stamp (rule i) and on a digest that moved since the newest codex-v* tag under an unmoved version (A4 row 10).
codex-publish: ## refuses unless the full gate passed on these exact bytes; tags codex-v<version>
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

##@ Claude Code plugin, checked in at packaging/claude/multi-llm (packaging/claude/README.md)

# A commit to master that moves this tree IS a release, so it carries a new version.
claude-regen: ## regenerate the tracked plugin tree from dirty inputs (before the commit)
	$(CLAUDE_BUILD) --allow-dirty
	@echo "claude-regen: $(CLAUDE_PKG)/multi-llm rewritten; commit it WITH the source edit that implies it"

# `MULTI_LLM_ALLOW_DIRTY=` on the recipe line for the reason codex-package gives.
claude-package: ## clean rebuild; must equal the committed tree (after the commit)
	MULTI_LLM_ALLOW_DIRTY= $(CLAUDE_BUILD) --require-no-diff
	@echo "claude-package: clean rebuild equals the committed tree"

# The moved-tree rule, reading git only. CLAUDE_RELEASE_BASE is read from the environment through
# `$(value …)` so a branch name holding make or shell syntax never becomes recipe text.
claude-check-release: ## the tree moved since the last claude-v* tag only under a newer version
	$(CLAUDE_BUILD) --check-release $(if $(value CLAUDE_RELEASE_BASE),--base "$$CLAUDE_RELEASE_BASE",)

claude-check-bump: ## claude-check-release plus the shared-version check over every family and npm
	$(CLAUDE_BUILD) --check-bump

# ALLOW_DIRTY=1 swaps the clean build for the regeneration (a local run mid-edit; CI never sets it).
claude-package-check-ci: ## the rows that need no Docker and no claude binary
	$(if $(ALLOW_DIRTY),$(MAKE) claude-regen,$(MAKE) claude-package)
	$(MAKE) --no-print-directory claude-check-release
	$(if $(PLAIN_GRADED),@echo "claude-package-check-ci: plain rows left to the test job (PLAIN_ROWS=graded)",$(CLAUDE_PACKAGE_PYTEST) -m "not live and not container and not package" -q)
	MLLM_REQUIRE_PACKAGE_ROWS=1 $(CLAUDE_PACKAGE_PYTEST) -m "package and not container" -q
	@echo "claude-package-check-ci: CI tier green"

# The plugin as a user gets it, in the image, with the host's model on the loopback stub: no key,
# no route out. Run before a release, after claude-package-check-ci; `make e2e-container` runs it too.
claude-package-container: e2e-image ## in Docker, no key, no network: the sparse install, session, hooks, dispatch and upgrade
	env -u OPENROUTER_API_KEY -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN \
	  MLLM_E2E_IMAGE=$(IMAGE) \
	  $(PYTEST)/test_container_claude_package.py -q -rs -m "container and not live"

##@ Releasing the Claude Code plugin (a push to master that moves the tree IS the release)

# See release-gates.md#release-helpers. `bump-version` moves the shared version in its owner and
# every mirror (not the OpenCode CLI pins); the pre-push hook is local and never committed;
# `claude-publish` never forces and does not run the Docker check.
CLAUDE_RELEASE := uv run --quiet --project skills/multi-llm -- python $(CLAUDE_PKG)/release.py

bump-version: ## VERSION names the release (DRY_RUN shows only): set the shared version in its owner and every mirror, then regenerate
	@test -n "$(VERSION)" || { echo "usage: make bump-version VERSION=<x.y.z> [DRY_RUN=1]"; exit 1; }
	uv run --quiet --project skills/multi-llm -- python packaging/common/bump_version.py "$(VERSION)" $(if $(DRY_RUN),--dry-run)

claude-hook-install: ## local pre-push hook: a push of master or a claude-v* tag must pass the release checks
	$(CLAUDE_RELEASE) install-hook

claude-hook-uninstall: ## remove that hook (and only that hook)
	$(CLAUDE_RELEASE) uninstall-hook

claude-publish: ## from master: fetch, every release check, print the push and tag it would make (PUSH pushes master and the tag, never forced)
	$(CLAUDE_RELEASE) publish $(if $(PUSH),--push)

##@ Model catalog snapshots (run by a person, logged in; never CI)

refresh-model-catalogs: ## re-capture the catalogs tests/test_model_catalogs.py checks (SOURCES limits the run)
	cd skills/multi-llm && uv run --project . -- python -m tests.model_catalogs $(SOURCES)
