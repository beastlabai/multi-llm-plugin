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
        e2e-container-host-discovery e2e-version-delta e2e-container-latest \
        e2e-container-latest-host e2e-lock-update e2e-verify-record e2e-clean

help:
	@echo "Containerized E2E harness (see $(CONTAINER_DIR)/README.md)"
	@echo
	@echo "  make e2e-container                 pinned build; provider lane (no secrets, --network none)"
	@echo "  make e2e-container-host            pinned build; host lane (needs OPENROUTER_API_KEY)"
	@echo "  make e2e-container-host-discovery  host lane, --creds none tier: stage + list, no key, no network"
	@echo "  make e2e-container-latest         CLIs @latest; provider lane only; credential-free by construction"
	@echo "  make e2e-container-latest-host    CLIs @latest; host lane; guarded, opt-in, needs OPENROUTER_API_KEY"
	@echo "  make e2e-version-delta            print the pinned → installed versions of the @latest image"
	@echo "  make e2e-lock-update              move the pins the @latest image proves moved, then prove it"
	@echo "  make e2e-verify-record REC=<path> re-hash a runs/ bundle against a committed record"
	@echo "  make e2e-clean                    prune the ephemeral runs/ bundles"

# One line naming what is missing (binary absent vs daemon unreachable), rather
# than a page of CLI errors.
e2e-preflight:
	@command -v docker >/dev/null 2>&1 || { \
	  echo "make: docker is not on PATH — install Docker or run the non-container suite with 'pytest -m \"not container\"'"; exit 1; }
	@docker version >/dev/null 2>&1 || { \
	  echo "make: the docker CLI is present but no daemon is reachable — start Docker and retry"; exit 1; }

# --------------------------------------------------------------------------- #
# Pinned build: the gate you run before a release.
# --------------------------------------------------------------------------- #
e2e-image: e2e-preflight
	docker build \
	  --platform $(PLATFORM) \
	  --build-arg MLLM_PIN_MODE=pinned \
	  -f $(DOCKERFILE) \
	  -t $(IMAGE) \
	  .

e2e-container: e2e-image
	$(PYTEST) -m "container and not live"

e2e-container-host: e2e-image
	@test -n "$${OPENROUTER_API_KEY:-}" || { \
	  echo "make: the host lane needs OPENROUTER_API_KEY (a DEDICATED key carrying a key-level spend limit)."; \
	  echo "      For the credential-free tier run: make e2e-container-host-discovery"; exit 1; }
	OPENROUTER_API_KEY="$$OPENROUTER_API_KEY" $(PYTEST) -m "container and live"

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

# --------------------------------------------------------------------------- #
# Drift probe: CLI layer @latest, every other pin held.
# --------------------------------------------------------------------------- #
e2e-image-latest: e2e-preflight
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
	  $(PYTEST) -m "container and not live" -rs

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
# time. The durable provenance lives in the tracked records/ beside it.
e2e-clean:
	rm -rf $(CONTAINER_DIR)/runs/*/
	@echo "pruned $(CONTAINER_DIR)/runs/ (records/ untouched)"
