# The containerized E2E harness image (Phase 2 of plans/containerized-harness-e2e.md,
# a local-only working note, gitignored and absent from a fresh clone).
#
# BUILT FROM THE REPO ROOT, ALWAYS:
#
#     docker build --platform linux/amd64 --build-arg MLLM_PIN_MODE=pinned \
#         -f skills/multi-llm/tests/container/Dockerfile -t mllm-e2e:local .
#
# which is what the repo-root `Makefile` writes, and the only supported
# invocation. `COPY` cannot reach above the context and the product under test
# sits above tests/container/, so a build run from this directory cannot see the
# skill at all. Every COPY path below is therefore repo-root-relative, and every
# one of them is drawn from tests/container/context_manifest.txt -- there is no
# `COPY .` here and there must never be one (C6b rule 2); the repo root holds a
# developer's .env, a dogfooded .multi-llm/ and whatever scratch credential file
# is lying around, and the .dockerignore that keeps those out is written
# allow-nothing-by-default. tests/test_build_context.py asserts both directions.
#
# ------------------------------------------------------------------------------
# WHAT "PINNED" MEANS HERE (C9)
# ------------------------------------------------------------------------------
# The CLI version is the LEAST of what moves between two builds of this file, so
# every input is pinned at the depth that closes its own drift:
#
#   base image     FROM ...@sha256:...  -- never a tag. `node:22` silently becomes
#                  a new Node minor on a new Debian.
#   platform       the --platform the Makefile passes. An arm64 developer and an
#                  amd64 machine otherwise compare goldens from different binaries.
#   node + npm     whatever the base digest ships, ASSERTED in the final layer
#                  rather than assumed -- a base bump moves the runtime under
#                  every CLI at once.
#   uv             version + the release artifact's sha256, verified BEFORE it
#                  executes. `curl … | sh` would make the installer a live remote
#                  input running as root.
#   npm CLIs       exact version + registry integrity, `npm ci` against the
#                  committed package-lock.json. A bare `@x.y.z` re-resolves the
#                  whole transitive tree per build and trusts whatever tarball the
#                  registry serves today.
#   PyPI CLIs      a --generate-hashes lock installed with --require-hashes.
#   downloads      URL + sha256, verified before use.
#   the skill env  `uv sync --locked` against the committed uv.lock.
#
# ...and then the build CHECKS WHAT IT ACTUALLY INSTALLED. The last layer runs
# verify_versions.py, which diffs every installed version against
# cli-versions.lock and fails the build on any mismatch -- including for a CLI
# whose installer resolved a version we did not name. A pin is a request; that
# step is the evidence.
#
# MLLM_PIN_MODE=latest inverts exactly ONE row: the CLI layer resolves @latest and
# the version rows become recorded-not-enforced. Base digest, platform, uv and
# every artifact checksum stay pinned, which is what makes a red drift run point
# at the CLI that moved rather than at a base-image change nobody chose.
#
# ------------------------------------------------------------------------------
# LAYER ORDER IS A DESIGN DECISION
# ------------------------------------------------------------------------------
# The skill layer is built LAST, after every CLI layer, so editing an
# orchestrator rebuilds in seconds instead of reinstalling ten CLIs. Only the
# version-verify step comes after it, because it has to assert over the finished
# image.
#
# ------------------------------------------------------------------------------
# NO `# syntax=` DIRECTIVE
# ------------------------------------------------------------------------------
# Deliberate. A syntax line pulls a frontend image from a registry at build time,
# by a tag, on every build -- an unpinned remote input at the very top of a file
# whose entire premise is that there are none. Nothing below needs a feature the
# built-in frontend lacks.

FROM docker.io/library/node:22-bookworm-slim@sha256:d649c27dae7ba0137b3cef5dd75baa422c08dc3d9e3fc0c23dfb172dc3cc6436

# `-o pipefail` matters: several steps below are `curl … | sha256sum -c -` style
# pipelines, where the default shell would report the exit status of the last
# stage only and a failed download would look like a passing checksum.
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

# `pinned` (the gate) or `latest` (the drift probe). Declared before the first
# layer that reads it so a mode change invalidates only the CLI layer.
ARG MLLM_PIN_MODE=pinned

# Every value here is cross-checked against cli-versions.lock by
# tests/test_container_dockerfile_pins.py, so the two cannot drift apart.
ARG UV_VERSION=0.11.28
ARG UV_SHA256=e490a6464492183c5d4534a5527fb4440f7f2bb2f228162ad7e4afe076dc0224
ARG GOOSE_VERSION=1.51.0
ARG GOOSE_URL=https://github.com/aaif-goose/goose/releases/download/v1.51.0/goose-x86_64-unknown-linux-gnu.tar.bz2
ARG GOOSE_SHA256=de5bf71c6bf470fbb55276de871bbbd65c844aa066d3b09b1552da83f8e367e2
ARG GROK_VERSION=1.0.34
ARG GROK_URL=https://storage.googleapis.com/grok-build-public-artifacts/cli/grok-1.0.34-linux-x86_64
ARG GROK_SHA256=be5905e107d2b8b5f3c142d21ecfe4c8fd32a913d2fd551b788707930c4dc80d
ARG MUSE_VERSION=1.3.0-R3401.1
ARG MUSE_URL=https://lookaside.facebook.com/lookaside/muse/download/?channel=muse&version=1.3.0-R3401.1&file=muse-x86-linux
ARG MUSE_SHA256=71b089d055dfe6e4562092bc484896b61bd96fd6ef9fef9da54a14aa174e2a33
# The documented install is `curl https://cursor.com/install | bash`, and that
# script is SERVER-RENDERED: it bakes the current version into its body as a
# literal DOWNLOAD_URL, so the URL is stable while what it installs is not.
# Piping it into a shell would pin nothing, so the tarball is named by its bytes
# here instead. The sha256 is ours -- Cursor publishes none.
ARG CURSOR_AGENT_VERSION=2026.09.18-9a7762b
ARG CURSOR_AGENT_URL=https://downloads.cursor.com/lab/2026.09.18-9a7762b/linux/x64/agent-cli-package.tar.gz
ARG CURSOR_AGENT_SHA256=b1308f5a2fc05458b9d8966752986bb23a971bbcc67c842c1df94c4b8132bad9
# Google's Antigravity CLI. The sha256 is UPSTREAM'S OWN -- the GitHub release
# API publishes a `digest` per asset -- unlike cursor-agent's, which is ours
# because Cursor publishes none. The build checks it anyway: a published digest
# is a claim until sha256sum --check --strict agrees with it.
#
# agy RELEASES DAILY (1.1.15 on 2026-08-19, 1.2.7 on 2026-09-19), so this is a
# pin to re-confirm whenever this file is touched.
ARG AGY_VERSION=1.2.7
ARG AGY_URL=https://github.com/google-antigravity/antigravity-cli/releases/download/1.2.7/agy_cli_linux_x64.tar.gz
ARG AGY_SHA256=e410dd56d8c213ef12643d3ff5eaaab57a17e05bbf72e9415322f23879fc4a18
# oh-my-pi. A DOWNLOAD, never the npm layer: the npm package
# `@oh-my-pi/pi-coding-agent` needs Bun >= 1.3.14 and does not run under node at
# all, so `npm ci` cannot install it into this image. What ships instead is the
# GitHub release asset -- a Bun single-file ELF, 252,405,216 bytes, linked
# against libc/libm/libdl/libpthread only, which is why bookworm-slim's glibc
# 2.36 runs it with nothing added.
#
# Like agy's and muse's and unlike cursor-agent's, this sha256 is UPSTREAM'S
# OWN: the release publishes a SHA256SUMS.txt naming it against
# `omp-linux-x64`. Confirmed on 2026-09-17 to agree with sha256sum over the
# fetched file -- which is the point of checking it below anyway.
ARG OMP_VERSION=18.2.6
ARG OMP_URL=https://github.com/can1357/oh-my-pi/releases/download/v18.2.6/omp-linux-x64
ARG OMP_SHA256=0f38598c91e823d8cce07f151ec3999d51f213fb2cc3e07d89f1af8eef9247a2
# vibe's interpreter, NOT the product's. `mistral-vibe` declares
# `requires_python >=3.12` and bookworm-slim ships 3.11.2 -- and this file sets
# `UV_PYTHON_DOWNLOADS=never`, so `uv venv --python 3.12` cannot fetch one
# (which is the point: an interpreter downloaded at build time with no hash
# would be exactly the unpinned remote input C9 forbids). So it is pinned and
# checksummed here like uv, goose, grok and muse.
#
# The `install_only_stripped` build is deliberate: the unstripped tarball
# unpacks to 333 MB against this one's 99 MB, and nothing in this image debugs
# CPython. Measured: the venv builds and `vibe --version` answers on both.
ARG PYTHON312_VERSION=3.12.11
ARG PYTHON312_URL=https://github.com/astral-sh/python-build-standalone/releases/download/20250818/cpython-3.12.11%2B20250818-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz
ARG PYTHON312_SHA256=b5a4f189f25cbacba0f76c9bd6f3ea8c35d2064068aa74ccbb6863068caababd

LABEL org.opencontainers.image.title="multi-llm containerized E2E harness"
LABEL org.opencontainers.image.description="Real coding-agent CLIs, stubbed backends, both lanes."
LABEL org.opencontainers.image.source="https://github.com/beastlabai/multi-llm-plugin"

# --------------------------------------------------------------------------- #
# 1. Identity, before anything writes a file it will later have to own.
#
# A FIXED NUMERIC NON-ROOT IDENTITY, baked here and re-stated by `docker run
# --user` and by the tmpfs uid=/gid= (posture.yaml). Those are three separate
# failures -- an image whose final USER is root, a run that forgets --user, a
# tmpfs mounted for the wrong owner -- so each is asserted rather than assumed to
# follow from the others.
#
# Non-root here is not hygiene: several agent CLIs refuse to start or change
# sandbox behaviour under uid 0, and on the host lane a root process holds a live
# key on an open network. A real /etc/passwd entry and an OWNED $HOME are part of
# the requirement, not decoration -- `--user $(id -u)` on a bare image produces a
# process with neither, which is precisely the shape those CLIs mishandle, and
# `gemini` crashes outright when $HOME does not exist.
# --------------------------------------------------------------------------- #
RUN groupadd --gid 10001 e2e \
 && useradd --uid 10001 --gid 10001 --create-home --home-dir /home/e2e \
            --shell /bin/bash --comment "containerized e2e harness" e2e \
 && mkdir -p /opt/mllm/bin /opt/mllm/real /opt/mllm/rig /opt/mllm/skill \
             /opt/mllm/lock /opt/mllm/npm /opt/mllm/venvs \
             /work/homes /work/out /work/sample-project \
 && chown -R 10001:10001 /home/e2e /work \
 && chmod 0700 /home/e2e

# --------------------------------------------------------------------------- #
# 2. The two things the base digest does not ship.
#
# `python3` runs the whole rig (entrypoint, stub, wrapper -- all stdlib) and
# backs both uv environments; `git` is what seed_workspace.py initializes the
# sample project with and what the product's own write-safety pin
# (`git rev-parse --show-toplevel`) calls. bzip2 unpacks the goose release.
#
# These come from the base image's own signed, checksummed apt indexes, and their
# versions are recorded in cli-versions.lock as `enforce: false`: a Debian
# security rebuild of git is not a signal about a coding-agent CLI, and failing a
# build on one would train people to ignore that file.
# --------------------------------------------------------------------------- #
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      bzip2 \
      ca-certificates \
      curl \
      git \
      python3 \
 && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------- #
# 3. uv -- by URL and sha256, verified BEFORE it executes.
#
# The upstream install path is `curl -LsSf https://astral.sh/uv/install.sh | sh`,
# which is a live remote input piped into a root shell on every build. This is
# the same bytes, named.
# --------------------------------------------------------------------------- #
RUN test "$(uname -m)" = "x86_64" \
      || { echo "this image is pinned to linux/amd64 (cli-versions.lock: platform); build with --platform linux/amd64" >&2; exit 1; } \
 && curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 \
         -o /tmp/uv.tar.gz \
         "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz" \
 && echo "${UV_SHA256}  /tmp/uv.tar.gz" | sha256sum --check --strict - \
 && tar -xzf /tmp/uv.tar.gz -C /tmp \
 && install -m 0755 /tmp/uv-x86_64-unknown-linux-gnu/uv  /opt/mllm/real/uv \
 && install -m 0755 /tmp/uv-x86_64-unknown-linux-gnu/uvx /opt/mllm/real/uvx \
 && rm -rf /tmp/uv.tar.gz /tmp/uv-x86_64-unknown-linux-gnu

# Never UV_OFFLINE here: the BUILD is the one point in the pipeline where the
# network exists. `harness.py` passes UV_OFFLINE=1 into every `docker run` from
# posture.yaml, which is where offline belongs.
ENV UV_PROJECT_ENVIRONMENT=/opt/mllm/skill/.venv \
    UV_CACHE_DIR=/tmp/uv-cache \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy

# --------------------------------------------------------------------------- #
# 4. PyYAML for the SYSTEM python3 -- the rig's own interpreter.
#
# wrapper.py reads redirect.yaml to decide how to redirect each CLI, and its
# `import yaml` failure path returns an empty row: the redirection would then
# silently no-op and every provider row would run unredirected. That is a
# dependency, not a nicety, so it is installed with the same --require-hashes
# discipline as everything else rather than left to whatever the base ships.
# --------------------------------------------------------------------------- #
COPY skills/multi-llm/tests/container/rig-requirements.txt /opt/mllm/lock/rig-requirements.txt
RUN /opt/mllm/real/uv pip install \
      --python /usr/bin/python3 --break-system-packages \
      --require-hashes -r /opt/mllm/lock/rig-requirements.txt \
 && /usr/bin/python3 -c "import yaml; print('rig pyyaml', yaml.__version__)"

# --------------------------------------------------------------------------- #
# 5. The npm CLI layer -- codex, claude-code, gemini, opencode, cline, kilocode, pi.
#
# `npm ci` against a COMMITTED package-lock.json, never `npm i -g pkg@1.2.3`:
# the bare form re-resolves the entire transitive tree on every build and accepts
# whatever tarball the registry serves for that version today, so a re-published
# or mutated tarball behind an unchanged version passes unnoticed. The lockfile
# carries an integrity hash per package, transitive deps included.
#
# THIS IS THE ONE ROW `MLLM_PIN_MODE=latest` INVERTS. Everything else in this
# file -- base digest, platform, uv, every artifact checksum -- stays pinned in
# both modes, so a red drift run points at the CLI that moved.
# --------------------------------------------------------------------------- #
# COPILOT NEEDS TWO EXTRA STEPS IN THIS LAYER, AND BOTH MUST BE IN *THIS* RUN.
#
# 1. PRUNE THE MUSL PAYLOAD. `npm ci` resolves optional platform packages by
#    `os`/`cpu`, which cannot express glibc vs musl, so it installs BOTH
#    @github/copilot-linux-x64 (378 MB, the one this Debian image runs) and
#    @github/copilot-linuxmusl-x64 (371 MB, which it can never run). That is
#    §7's prune-ladder step 1, and it has to happen in the same RUN: deleting in
#    a later layer leaves the bytes in this one and the image never shrinks.
#    (The probe saw only one payload because `npm install -g` resolves for the
#    running platform; a lockfile install does not.)
#
# 2. PRE-WARM THE PACKAGE CACHE, without which the copilot row cannot start at
#    all. The `copilot` npm entry point is a LOADER: on first run it
#    self-extracts the 209 MB platform package into $XDG_CACHE_HOME and executes
#    from there. At RUN time the wrapper points XDG_CACHE_HOME at the
#    per-invocation root, which is a --tmpfs mount, and Docker's tmpfs default is
#    NOEXEC -- so copilot dies with `failed to map segment from shared object`, a
#    message that reads like a corrupt install and sends you into the adapter.
#    COPILOT_HOME does NOT relocate this cache (measured); only XDG_CACHE_HOME
#    does. Extracting once here, world-readable under /opt, also removes the
#    209 MB extraction from every spawn, and needs NO posture.yaml change: the
#    probe ran exactly this as UID 10001 under --read-only with the unmodified
#    tmpfs mode strings. The copilot row's `env_extra` pins XDG_CACHE_HOME here.
#    Set inline rather than as an image-wide ENV: every other CLI wants its own
#    per-invocation cache.
COPY skills/multi-llm/tests/container/package.json      /opt/mllm/npm/package.json
COPY skills/multi-llm/tests/container/package-lock.json /opt/mllm/npm/package-lock.json
COPY skills/multi-llm/tests/container/link_cli_bins.mjs /opt/mllm/npm/link_cli_bins.mjs
RUN cd /opt/mllm/npm \
 && if [ "${MLLM_PIN_MODE}" = "latest" ]; then \
        echo "MLLM_PIN_MODE=latest: resolving the CLI layer at @latest (drift probe)"; \
        specs=$(node -e 'const d=require("./package.json").dependencies;process.stdout.write(Object.keys(d).map(k=>k+"@latest").join(" "))'); \
        rm -f package-lock.json; \
        npm install --no-audit --no-fund ${specs}; \
    else \
        npm ci --no-audit --no-fund; \
    fi \
 && node /opt/mllm/npm/link_cli_bins.mjs /opt/mllm/npm /opt/mllm/real \
        codex claude gemini opencode cline kilocode kimi pi qwen copilot \
 && rm -rf /opt/mllm/npm/node_modules/@github/copilot-linuxmusl-x64 \
 && mkdir -p /opt/mllm/copilot-cache \
 && XDG_CACHE_HOME=/opt/mllm/copilot-cache \
    COPILOT_OFFLINE=true COPILOT_AUTO_UPDATE=false \
    /opt/mllm/real/copilot --version \
 && chmod -R a+rX /opt/mllm/copilot-cache \
 && test -d /opt/mllm/copilot-cache/copilot/pkg/linux-x64 \
 && npm cache clean --force

# --------------------------------------------------------------------------- #
# 6. aider -- PyPI, in its own virtualenv, every transitive wheel hash-pinned.
#
# `uv pip compile --generate-hashes` + `--require-hashes` is the PyPI twin of
# `npm ci` against a lockfile: the version is named, and so is every byte behind
# it. Its own venv rather than the skill's, because aider pins a large dependency
# tree of its own and the product's environment must stay exactly what uv.lock
# says it is.
# --------------------------------------------------------------------------- #
COPY skills/multi-llm/tests/container/aider-requirements.txt /opt/mllm/lock/aider-requirements.txt
RUN /opt/mllm/real/uv venv --python /usr/bin/python3 /opt/mllm/venvs/aider \
 && if [ "${MLLM_PIN_MODE}" = "latest" ]; then \
        /opt/mllm/real/uv pip install --python /opt/mllm/venvs/aider/bin/python aider-chat; \
    else \
        /opt/mllm/real/uv pip install --python /opt/mllm/venvs/aider/bin/python \
            --require-hashes -r /opt/mllm/lock/aider-requirements.txt; \
    fi \
 && ln -sf /opt/mllm/venvs/aider/bin/aider /opt/mllm/real/aider

# --------------------------------------------------------------------------- #
# 6b. vibe -- PyPI too, but on an interpreter this image does not otherwise have.
#
# `mistral-vibe` declares `requires_python >=3.12`; the base ships 3.11.2. uv
# could fetch a standalone build, except `UV_PYTHON_DOWNLOADS=never` is set above
# and an unpinned interpreter would be the exact unpinned remote input C9 exists
# to forbid. So the interpreter is a hashed artifact like uv/goose/grok/muse, and
# the venv is built against it.
#
# OWNERSHIP IS PART OF THE FIX, not tidiness. uv's own managed-interpreter
# location is under the BUILD user's data dir and is unreachable to uid 10001, so
# the interpreter, the venv and the symlink all live under /opt/mllm with
# world-readable/executable modes -- the runtime user must be able to exec every
# one of them.
#
# The wheel also installs `vibe-acp` and `vibe-app-server`; neither is wanted, so
# only `vibe` is linked into /opt/mllm/real.
# --------------------------------------------------------------------------- #
COPY skills/multi-llm/tests/container/vibe-requirements.txt /opt/mllm/lock/vibe-requirements.txt
RUN curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 \
         -o /tmp/python312.tar.gz "${PYTHON312_URL}" \
 && if [ "${MLLM_PIN_MODE}" = "pinned" ]; then \
        echo "${PYTHON312_SHA256}  /tmp/python312.tar.gz" | sha256sum --check --strict -; \
    else \
        echo "unpinned python312 artifact sha256: $(sha256sum /tmp/python312.tar.gz)"; \
    fi \
 && mkdir -p /opt/mllm/real/python312 \
 && tar -xzf /tmp/python312.tar.gz -C /opt/mllm/real/python312 --strip-components=1 \
 && rm -f /tmp/python312.tar.gz \
 && /opt/mllm/real/python312/bin/python3.12 --version \
 && /opt/mllm/real/uv venv --python /opt/mllm/real/python312/bin/python3.12 \
        /opt/mllm/venvs/vibe \
 && if [ "${MLLM_PIN_MODE}" = "latest" ]; then \
        /opt/mllm/real/uv pip install --python /opt/mllm/venvs/vibe/bin/python mistral-vibe; \
    else \
        /opt/mllm/real/uv pip install --python /opt/mllm/venvs/vibe/bin/python \
            --require-hashes -r /opt/mllm/lock/vibe-requirements.txt; \
    fi \
 && ln -sf /opt/mllm/venvs/vibe/bin/vibe /opt/mllm/real/vibe \
 && chmod -R a+rX /opt/mllm/real/python312 /opt/mllm/venvs/vibe

# --------------------------------------------------------------------------- #
# 7. The three CLIs that ship as a release artifact rather than a package.
#
# goose  -- a GitHub release tarball.
# grok   -- a single static binary from the public artifact bucket that
#           `https://x.ai/cli/install.sh` itself downloads from. The install
#           script is not piped into a shell; the bytes it would have fetched are
#           named and hashed instead.
# muse   -- Meta-proprietary, and the one row that needed a ruling rather than a
#           URL. The `muse` a developer has on PATH is a bash LAUNCHER that
#           contacts api.meta.ai on every invocation and updates itself
#           synchronously and fatally, which would defeat every pin in this file
#           and hang under --network none. But the release channel, the manifest
#           and the artifact are all served UNAUTHENTICATED, and Meta publishes a
#           sha256 in the manifest -- so muse is installable from a public source
#           after all. What ships is the artifact the launcher would have
#           downloaded, under the name `muse`, with the launcher nowhere in the
#           image. That is what makes `muse --version` answer with no network.
#
# In `latest` mode these resolve their channel's current version and their
# checksum is recorded rather than pre-verified, because a checksum you compute
# after downloading is not a pin -- which is exactly the property that build is
# probing for.
# --------------------------------------------------------------------------- #
RUN if [ "${MLLM_PIN_MODE}" = "latest" ]; then \
        goose_url="https://github.com/aaif-goose/goose/releases/download/stable/goose-x86_64-unknown-linux-gnu.tar.bz2"; \
    else \
        goose_url="${GOOSE_URL}"; \
    fi \
 && curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 \
         -o /tmp/goose.tar.bz2 "${goose_url}" \
 && if [ "${MLLM_PIN_MODE}" = "pinned" ]; then \
        echo "${GOOSE_SHA256}  /tmp/goose.tar.bz2" | sha256sum --check --strict -; \
    else \
        echo "unpinned goose artifact sha256: $(sha256sum /tmp/goose.tar.bz2)"; \
    fi \
 && tar -xjf /tmp/goose.tar.bz2 -C /tmp \
 && install -m 0755 /tmp/goose /opt/mllm/real/goose \
 && rm -f /tmp/goose.tar.bz2 /tmp/goose

RUN if [ "${MLLM_PIN_MODE}" = "latest" ]; then \
        grok_version="$(curl --fail --silent --show-error --location --proto '=https' https://storage.googleapis.com/grok-build-public-artifacts/cli/stable | tr -d '[:space:]')"; \
        grok_url="https://storage.googleapis.com/grok-build-public-artifacts/cli/grok-${grok_version}-linux-x86_64"; \
    else \
        grok_url="${GROK_URL}"; \
    fi \
 && curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 \
         -o /tmp/grok "${grok_url}" \
 && if [ "${MLLM_PIN_MODE}" = "pinned" ]; then \
        echo "${GROK_SHA256}  /tmp/grok" | sha256sum --check --strict -; \
    else \
        echo "unpinned grok artifact sha256: $(sha256sum /tmp/grok)"; \
    fi \
 && install -m 0755 /tmp/grok /opt/mllm/real/grok \
 && rm -f /tmp/grok

RUN if [ "${MLLM_PIN_MODE}" = "latest" ]; then \
        muse_version="$(curl --fail --silent --show-error --location --proto '=https' -A 'muse-code/launcher-2' https://api.meta.ai/muse-code/channels/muse-stable | python3 -c 'import json,sys; print(json.load(sys.stdin)["version"])')"; \
        muse_url="https://lookaside.facebook.com/lookaside/muse/download/?channel=muse&version=${muse_version}&file=muse-x86-linux"; \
    else \
        muse_url="${MUSE_URL}"; \
    fi \
 && curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 \
         -A 'muse-code/launcher-2' -o /tmp/muse "${muse_url}" \
 && if [ "${MLLM_PIN_MODE}" = "pinned" ]; then \
        echo "${MUSE_SHA256}  /tmp/muse" | sha256sum --check --strict -; \
    else \
        echo "unpinned muse artifact sha256: $(sha256sum /tmp/muse)"; \
    fi \
 && install -m 0755 /tmp/muse /opt/mllm/real/muse \
 && rm -f /tmp/muse

# cursor-agent. The one CLI in this image that NO provider row drives: it speaks
# binary protobuf over Connect-RPC to its own endpoint, so there is no stub to
# write and no base-URL variable to move. It is here for the two consumers that
# need the genuine binary and no backend that answers -- the credentialed live
# lane (live_provider.py, which execs /opt/mllm/real/<binary> directly) and the
# cli-surface help-drift probe (seed_cli_surface.py, likewise). cli-versions.lock
# carries the full reasoning above its `clis:` block.
#
# COST, because it is the largest single thing in this image: +245 MiB unpacked
# (254,358,760 bytes measured) from an 84 MB tarball -- and 129 MB of that is a
# SECOND bundled node, on top of the base image's own. Cursor ships the bundle
# self-contained and offers no variant that reuses the system runtime, so the
# duplicate interpreter is the price of the row, not an oversight to trim later.
#
# The tarball has a single root (`dist-package/`), so --strip-components=1 lands
# the bundle flat. `cursor-agent` inside it is a bash launcher that resolves its
# own directory through realpath and execs `$SCRIPT_DIR/node index.js`, which is
# why a symlink into /opt/mllm/real works here exactly as it does for vibe.
#
# No NODE_COMPILE_CACHE ENV is set, deliberately. The launcher defaults it to
# ${XDG_CACHE_HOME:-$HOME/.cache}/cursor-compile-cache, and every lane's $HOME is
# already a writable tmpfs (posture.yaml: /work/homes, /home/e2e and /tmp are all
# tmpfs; the wrapper points $HOME at /tmp/mllm-inv/<cli>-<exec>). Pinning it
# would only ADD risk: an unwritable cache directory is tolerated (measured --
# exit 0, correct version), but a NODE_COMPILE_CACHE naming an uncreatable parent
# was measured HANGING the CLI indefinitely. Leaving the launcher's own default
# in place is the safe branch.
#
# THE LATEST-MODE DISCOVERY MATCHES THE VERSION SEGMENT, NEVER THE WHOLE URL.
# MEASURED 2026-09-19: the server-rendered installer stopped baking the platform
# into DOWNLOAD_URL and now emits it as SHELL PLACEHOLDERS the script expands at
# run time --
#
#     DOWNLOAD_URL="https://downloads.cursor.com/lab/2026.09.18-9a7762b/${OS}/${ARCH}/agent-cli-package.tar.gz"
#
# -- so the previous pattern, which required a literal `linux/x64`, matched
# nothing and the drift probe died on `the installer's shape changed` with the
# whole @latest build behind it. The version is the only field we need from that
# page; `linux/x64` is OUR platform choice (this image is pinned to linux/amd64
# and says so three ways), so it is spelled here rather than looked for there.
# The pattern accepts both shapes: it stops at the segment after `/lab/`.
RUN if [ "${MLLM_PIN_MODE}" = "latest" ]; then \
        cursor_ver="$(curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 https://cursor.com/install | grep -oE 'downloads\.cursor\.com/lab/[^/"'"'"']+/' | head -n 1 | sed -e 's#.*/lab/##' -e 's#/$##')"; \
        [ -n "${cursor_ver}" ] || { echo "cursor-agent: no lab version found in https://cursor.com/install -- the installer's shape changed" >&2; exit 1; }; \
        cursor_url="https://downloads.cursor.com/lab/${cursor_ver}/linux/x64/agent-cli-package.tar.gz"; \
        echo "cursor-agent: @latest resolved to ${cursor_ver}"; \
    else \
        cursor_url="${CURSOR_AGENT_URL}"; \
    fi \
 && curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 \
         -o /tmp/cursor-agent.tar.gz "${cursor_url}" \
 && if [ "${MLLM_PIN_MODE}" = "pinned" ]; then \
        echo "${CURSOR_AGENT_SHA256}  /tmp/cursor-agent.tar.gz" | sha256sum --check --strict -; \
    else \
        echo "unpinned cursor-agent artifact sha256: $(sha256sum /tmp/cursor-agent.tar.gz)"; \
    fi \
 && mkdir -p /opt/mllm/real/cursor-agent-bundle \
 && tar -xzf /tmp/cursor-agent.tar.gz -C /opt/mllm/real/cursor-agent-bundle --strip-components=1 \
 && ln -sf /opt/mllm/real/cursor-agent-bundle/cursor-agent /opt/mllm/real/cursor-agent \
 && chmod -R a+rX /opt/mllm/real/cursor-agent-bundle \
 && rm -f /tmp/cursor-agent.tar.gz \
 && /opt/mllm/real/cursor-agent --version

# agy -- Google's Antigravity CLI, and the SECOND row in this image that no
# provider row drives. cursor-agent's reason is "no stub can be written"; agy's
# is stronger and settled: its auth resolves BEFORE the first RPC, and
# GEMINI_API_KEY + GOOGLE_API_KEY + GOOGLE_GEMINI_BASE_URL set together were
# measured producing a byte-identical auth failure with ZERO requests reaching a
# stub. So there is no environment variable that selects an API-key path for a
# stub to answer, and there never will be one to find. It is here for the ONE
# consumer that needs the genuine binary and no backend at all: the cli-surface
# help-drift probe (seed_cli_surface.py), which execs /opt/mllm/real/agy
# directly, offline, as uid 10001, and diffs the flags our adapter emits against
# the flags this version still documents.
#
# COST: +196 MiB unpacked (205,537,320 bytes) from a 55 MB tarball -- second
# only to cursor-agent's 245 MiB. Nothing else in the image gets larger for it:
# a dynamically linked glibc binary, no bundled node, no second interpreter.
#
# The goose pattern, not cursor-agent's: ONE tarball member, named `antigravity`
# and sitting at the ROOT with no directory around it, so there is no
# --strip-components here and `install` renames it to the name it answers to.
#
# AGY_CLI_DISABLE_AUTO_UPDATE=true ON THE SMOKE CALL IS LOAD-BEARING, and `true`
# rather than `1` is the whole of it. `=1` was MEASURED spawning a background
# updater that rewrote the binary 1.1.13 -> 1.1.15 inside a single run; only the
# string `true` disables it. This build layer has a network, so an unguarded
# `agy --version` here would replace the exact bytes sha256sum just verified and
# the pin would be decorative. The image-wide ENV in step 8 repeats it for
# everything downstream, and redirect.yaml's `env_extra` repeats it again for
# anything dispatched through the wrapper -- which rebuilds its child's
# environment from scratch and would otherwise drop it.
RUN if [ "${MLLM_PIN_MODE}" = "latest" ]; then \
        agy_version="$(curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 https://api.github.com/repos/google-antigravity/antigravity-cli/releases/latest | python3 -c 'import json,sys; print(json.load(sys.stdin)["tag_name"].lstrip("v"))')"; \
        agy_url="https://github.com/google-antigravity/antigravity-cli/releases/download/${agy_version}/agy_cli_linux_x64.tar.gz"; \
    else \
        agy_url="${AGY_URL}"; \
    fi \
 && curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 \
         -o /tmp/agy.tar.gz "${agy_url}" \
 && if [ "${MLLM_PIN_MODE}" = "pinned" ]; then \
        echo "${AGY_SHA256}  /tmp/agy.tar.gz" | sha256sum --check --strict -; \
    else \
        echo "unpinned agy artifact sha256: $(sha256sum /tmp/agy.tar.gz)"; \
    fi \
 && tar -xzf /tmp/agy.tar.gz -C /tmp antigravity \
 && install -m 0755 /tmp/antigravity /opt/mllm/real/agy \
 && rm -f /tmp/agy.tar.gz /tmp/antigravity \
 && AGY_CLI_DISABLE_AUTO_UPDATE=true /opt/mllm/real/agy --version

# --------------------------------------------------------------------------- #
# oh-my-pi (`omp`): the download, and the 346 MB of native addons it would
# otherwise extract into every single $HOME at run time.
#
# THE FILE LANDS UNCHANGED (`install -m 0755`, grok's shape rather than agy's
# tarball) and that is load-bearing for `lock_update.py`: `plan_artifacts`
# recovers a moved row's sha256 by hashing /opt/mllm/real/<binary> out of an
# @latest image, which is only the artifact's own digest while the install copies
# the fetched bytes verbatim. `retarget_url` then moves the pin because the
# version appears EXACTLY ONCE in the URL (`/v18.2.2/`).
#
# THE NATIVES. MEASURED (2026-09-17, 18.2.2): the Bun binary embeds a gzipped
# archive of its native addons and extracts 346 MB -- two files,
# pi_natives.linux-x64-{modern,baseline}.node at ~181 MB each -- into
# $HOME/.omp/natives/<version>/ on the first run that loads them. `--version` and
# `--help` do NOT trigger it (50 ms, nothing written) and neither does a bare
# `omp config get`, which prints its usage and stops before the loader; `omp
# config list` DOES, in ~3 s, with no network, no model and no credential -- which
# is what makes it the warm-up here instead of opencode's deliberately-failing
# `run`. The EXIT CODE is not the assertion (that is opencode's rule too): what
# this layer needs is the extracted tree, so the command may fail and the `test`
# below is what fails the build.
#
# Read off the binary (packages/natives/native/loader-state.js, in the bundle):
# the extractor skips any addon already present at the target path WITH THE
# EXPECTED SIZE, and the loader's own candidate list starts at that same
# directory -- so a $HOME/.omp/natives pointing at a populated tree costs two
# stat() calls and no extraction, whether or not the tree is writable (a
# read-only target fails the extractor's utimes() and the loader falls through to
# the candidate, which is the SAME file). That is why the bytes live here, once,
# under the read-only root: `redirect.yaml`'s `seed_links` is what points each
# per-invocation $HOME at them (wrapper.py builds its child's environment and its
# config root from scratch, so neither an image ENV nor the seed tree reaches it
# by itself). MEASURED IN THIS IMAGE, uid 10001, --read-only, --network none:
# 132 ms and 200 KB of $HOME through the link, against 2,729 ms and 346 MB
# without it; and on the host, through the same link, a real agent run against a
# stub in 0.69 s with 253 KB of $HOME. The megabytes would be tmpfs, per
# concurrent exec, since `wrapper.seed_invocation_root` copies the whole per-CLI
# home into /tmp/mllm-inv on EVERY exec.
#
# THE VERSION IS READ BACK OFF THE EXTRACTED TREE rather than from ${OMP_VERSION}:
# under MLLM_PIN_MODE=latest the ARG is precisely the thing that is stale.
# --------------------------------------------------------------------------- #
RUN if [ "${MLLM_PIN_MODE}" = "latest" ]; then \
        omp_version="$(curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 https://api.github.com/repos/can1357/oh-my-pi/releases/latest | python3 -c 'import json,sys; print(json.load(sys.stdin)["tag_name"].lstrip("v"))')"; \
        omp_url="https://github.com/can1357/oh-my-pi/releases/download/v${omp_version}/omp-linux-x64"; \
    else \
        omp_url="${OMP_URL}"; \
    fi \
 && curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 \
         -o /tmp/omp "${omp_url}" \
 && if [ "${MLLM_PIN_MODE}" = "pinned" ]; then \
        echo "${OMP_SHA256}  /tmp/omp" | sha256sum --check --strict -; \
    else \
        echo "unpinned omp artifact sha256: $(sha256sum /tmp/omp)"; \
    fi \
 && install -m 0755 /tmp/omp /opt/mllm/real/omp \
 && rm -f /tmp/omp \
 && /opt/mllm/real/omp --version \
 && warm=/tmp/omp-warm \
 && mkdir -p "$warm" \
 && ( HOME="$warm" /opt/mllm/real/omp config list >/tmp/omp-warm.log 2>&1 || true ) \
 && test -d "$warm/.omp/natives" \
      || { echo "omp extracted no natives tree: every per-invocation HOME would pay the 346 MB extraction, and /work/homes and /tmp are tmpfs" >&2; \
           tail -20 /tmp/omp-warm.log >&2; exit 1; } \
 && mkdir -p /opt/mllm/omp-natives \
 && cp -a "$warm/.omp/natives/." /opt/mllm/omp-natives/ \
 && chmod -R a=rX /opt/mllm/omp-natives \
 && ls /opt/mllm/omp-natives/*/pi_natives.linux-x64-*.node >/dev/null \
      || { echo "the extracted tree holds no pi_natives addon: the loader would fall through to an extraction" >&2; exit 1; } \
 && du -sh /opt/mllm/omp-natives \
 && rm -rf "$warm" /tmp/omp-warm.log

# --------------------------------------------------------------------------- #
# 8. The rig, and the wrapper tree it is the source of (C2).
#
#     /opt/mllm/real/<cli>   the genuine binary, untouched
#     /opt/mllm/bin/<cli>    wrapper.py under that name, FIRST on PATH
#     /opt/mllm/bin/uv       the same file again -- this one writes the receipts
#
# SYMLINKS, not copies: wrapper.py resolves which CLI it is from
# `Path(sys.argv[0]).name`, and argv[0] is the path the caller used, so a symlink
# keeps one source of truth and one identity mechanism. A hard copy per CLI would
# be eleven files to keep in step.
#
# EVERY cli-versions.lock row is linked here, including `cursor-agent`, which has
# no stub to be redirected to. That is not an oversight in either direction:
# verify_versions.py's check_path_shape walks the lock's `clis` block and fails
# the build on any binary that does not resolve to /opt/mllm/bin, so omitting it
# would be a red build. And wrapping it is harmless: cursor DOES have a
# redirect.yaml row (`lane: fake`, plus the live-lane opt-in), but that row
# carries no `base_url_vars`, no `credential_vars`, no `env_extra` and no
# `config_roots` -- there is nothing in it for the wrapper to apply -- so what
# the wrapper contributes is exactly the env scrub and the call log, which is the
# right default for a CLI with no backend to point anywhere. Its two real
# consumers (live_provider.py, seed_cli_surface.py) bypass /opt/mllm/bin entirely
# and exec /opt/mllm/real/<binary>, so neither is affected by this loop at all.
#
# The .yaml sources are copied as a group rather than named one by one so that
# redirect.yaml -- which Phase 3 writes beside posture.yaml -- lands without a
# Dockerfile edit. Every file the glob can reach is inside context_manifest.txt.
# --------------------------------------------------------------------------- #
COPY skills/multi-llm/tests/container/entrypoint.py     /opt/mllm/rig/entrypoint.py
COPY skills/multi-llm/tests/container/stub_server.py    /opt/mllm/rig/stub_server.py
COPY skills/multi-llm/tests/container/wrapper.py        /opt/mllm/rig/wrapper.py
COPY skills/multi-llm/tests/container/seed_workspace.py /opt/mllm/rig/seed_workspace.py
COPY skills/multi-llm/tests/container/seed_cli_surface.py /opt/mllm/rig/seed_cli_surface.py
# The credentialed provider lane (harness-expansion.md §7.4). It is NOT reached
# through entrypoint.py: the lane bypasses the PATH wrapper on purpose -- the
# wrapper's whole job is to scrub the vendor credential and point the CLI at the
# stub -- so its runner is invoked directly, the way `make e2e-version-delta`
# already invokes verify_versions.py with `--entrypoint /usr/bin/python3`.
COPY skills/multi-llm/tests/container/live_provider.py  /opt/mllm/rig/live_provider.py
COPY skills/multi-llm/tests/container/egress_resolver.py /opt/mllm/rig/egress_resolver.py
COPY skills/multi-llm/tests/container/*.yaml            /opt/mllm/rig/
COPY skills/multi-llm/tests/container/home_seed         /opt/mllm/rig/home_seed

RUN chmod 0755 /opt/mllm/rig/*.py \
 && for b in codex claude gemini opencode cline kilocode kimi pi qwen copilot aider vibe goose grok muse cursor-agent agy omp uv; do \
        ln -sf /opt/mllm/rig/wrapper.py "/opt/mllm/bin/$b"; \
    done \
 && /usr/bin/python3 -m compileall -q /opt/mllm/rig >/dev/null

# /opt/mllm/bin FIRST, for the whole image: C2's redirection is "the wrapper is
# first on PATH", and the final layer asserts that every CLI really resolves
# there rather than to /opt/mllm/real.
ENV PATH=/opt/mllm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# ...and again for a LOGIN shell, which is not the same statement (measured,
# host lane, 2026-08-14). `codex exec` runs every command it is asked to run as
# `/bin/bash -lc '<command>'`, and Debian's /etc/profile *assigns* PATH rather
# than extending it — so the ENV above is invisible to every command the agent
# executes. The row that found this read our shim correctly, ran its `command -v
# uv` preflight, got nothing, checked all seven documented fallback locations,
# got nothing, and did exactly what the shim says: printed hosts/uv-missing.txt
# and stopped. A perfect instruction-following run that proved nothing about the
# dispatcher, because the dispatcher was unreachable.
#
# `00-` so it is sourced before anything else in profile.d, and a prepend rather
# than an assignment so the wrapper stays FIRST without discarding what
# /etc/profile just built.
RUN printf '%s\n' 'PATH=/opt/mllm/bin:$PATH' 'export PATH' \
      > /etc/profile.d/00-mllm-path.sh \
 && chmod 0644 /etc/profile.d/00-mllm-path.sh \
 && test "$(/bin/bash -lc 'command -v uv')" = /opt/mllm/bin/uv \
      || { echo "a login shell still cannot see the wrapper's uv" >&2; exit 1; }

ENV MLLM_SKILL_DIR=/opt/mllm/skill \
    MLLM_RIG_DIR=/opt/mllm/rig \
    MLLM_REAL_DIR=/opt/mllm/real \
    MLLM_BIN_DIR=/opt/mllm/bin \
    MLLM_HOMES_ROOT=/work/homes \
    MLLM_REDIRECT_YAML=/opt/mllm/rig/redirect.yaml \
    MLLM_WRAPPER=/opt/mllm/rig/wrapper.py

# Self-updating binaries defeat every pin above, so each one's opt-out is set
# image-wide. cline was measured rewriting its own global node_modules from
# 3.0.39 to 3.0.55 *while a probe was running*; the read-only root is the second
# layer under this, not the first.
#
# OPENCODE_DISABLE_MODELS_FETCH pairs with the pre-baked catalog in step 9.
#
# AGY_CLI_DISABLE_AUTO_UPDATE IS `true`, NOT `1`, and that is measured rather
# than stylistic: with `=1` agy spawned a background updater and rewrote itself
# 1.1.13 -> 1.1.15 inside a single run. Normalising it to `1` for consistency
# with the lines around it would silently unpin a CLI that releases DAILY.
#
# NOTE for Phase 3: wrapper.py builds its child's environment from scratch and
# carries over only PATH/TERM/LANG/LC_ALL/TZ and MLLM_*, so these names do NOT
# reach a CLI dispatched *through* the wrapper. They cover the entrypoint, the
# build, and any direct invocation; the per-CLI copies belong in redirect.yaml.
ENV CLINE_NO_AUTO_UPDATE=1 \
    KIMI_CODE_NO_AUTO_UPDATE=1 \
    KIMI_DISABLE_TELEMETRY=1 \
    QWEN_TELEMETRY_ENABLED=false \
    QWEN_USAGE_STATISTICS_ENABLED=false \
    QWEN_CODE_SKIP_UPDATE_CHECK_ONCE=1 \
    GROK_AUTO_UPDATE=0 \
    AGY_CLI_DISABLE_AUTO_UPDATE=true \
    OPENCODE_DISABLE_MODELS_FETCH=1 \
    DO_NOT_TRACK=1 \
    NO_UPDATE_NOTIFIER=1 \
    NPM_CONFIG_UPDATE_NOTIFIER=false

# --------------------------------------------------------------------------- #
# 9. opencode's cold start, moved into the build -- where the network is.
#
# MEASURED (Phase 1): on a fresh $XDG_CONFIG_HOME, opencode npm-installs ~63 MB
# into $XDG_CONFIG_HOME/opencode/node_modules and fetches ~3.6 MB of models.json
# from models.dev before it will answer anything. Under --network none that is
# not a slow row, it is a HARD failure that never reaches the stub at all -- and
# it presents as a redirection bug rather than as a missing install step.
#
# So it is resolved here and folded into home_seed/opencode, which the entrypoint
# copies to /work/homes/opencode and the wrapper then seeds each per-invocation
# root from. The build is the one point in the pipeline with an index to reach.
#
# THE TRIGGER IS `opencode run`, NOT `opencode models`. Measured in this image:
# `models` fetches the catalog and stops, leaving node_modules absent; the
# provider SDK install only happens once a `provider` block names an `npm`
# package and a run needs it. So the warm-up writes a throwaway config declaring
# the same `@ai-sdk/openai-compatible` package the redirect matrix uses and runs
# one prompt. That run is EXPECTED to fail -- there is no stub listening at build
# time -- and it fails at the connect step, which is after the install. The
# assertion is the node_modules tree, never the exit code.
#
# The throwaway opencode.json is deleted before the tree is folded in: the base
# URL belongs to redirect.yaml, which the wrapper renders per exec, and a stale
# config baked into the image would shadow it. package.json / package-lock.json
# stay, because they are what stops opencode re-resolving the tree on first use.
# --------------------------------------------------------------------------- #
RUN warm=/tmp/opencode-warm \
 && conf="$warm/.config/opencode" \
 && mkdir -p "$conf" "$warm/.cache" "$warm/.local/share" \
 && printf '%s\n' \
      '{"provider":{"mllmstub":{"npm":"@ai-sdk/openai-compatible","name":"mllm stub",' \
      '"options":{"baseURL":"http://127.0.0.1:8099/v1","apiKey":"mllm-build-warmup"},' \
      '"models":{"stub-model":{"name":"stub-model"}}}},"model":"mllmstub/stub-model"}' \
      > "$conf/opencode.json" \
 && ( cd /tmp && HOME="$warm" \
        XDG_CONFIG_HOME="$warm/.config" XDG_CACHE_HOME="$warm/.cache" \
        XDG_DATA_HOME="$warm/.local/share" XDG_STATE_HOME="$warm/.local/state" \
        OPENCODE_DISABLE_MODELS_FETCH= \
        /opt/mllm/real/opencode run -- "warm up the provider tree" \
        >/tmp/opencode-warm.log 2>&1 || true ) \
 && { test -d "$conf/node_modules" && test -s "$warm/.cache/opencode/models.json"; } \
      || { echo "opencode warm-up baked nothing: node_modules and/or models.json are missing, so every --network none row would die on the cold start and look like a redirection bug" >&2; \
           tail -40 /tmp/opencode-warm.log >&2; exit 1; } \
 && rm -f "$conf/opencode.json" \
 && seed=/opt/mllm/rig/home_seed/opencode \
 && mkdir -p "$seed/.config/opencode" "$seed/.cache/opencode" \
 && cp -a "$conf/."                 "$seed/.config/opencode/" \
 && cp -a "$warm/.cache/opencode/." "$seed/.cache/opencode/" \
 && test -d "$seed/.config/opencode/node_modules" \
      || { echo "the baked tree landed in the wrong place" >&2; exit 1; } \
 && du -sh "$seed" \
 && rm -rf "$warm" /tmp/opencode-warm.log

# --------------------------------------------------------------------------- #
# 10. The skill layer -- LAST, and by named paths only.
#
# Last so that editing an orchestrator rebuilds in seconds rather than
# reinstalling every CLI above it. Named paths only: every source here appears in
# tests/container/context_manifest.txt, and tests/test_build_context.py fails the
# build-context guard if a COPY reaches one byte wider.
# --------------------------------------------------------------------------- #
COPY skills/multi-llm/*.py            /opt/mllm/skill/
COPY skills/multi-llm/providers.yaml  /opt/mllm/skill/providers.yaml
COPY skills/multi-llm/pyproject.toml  /opt/mllm/skill/pyproject.toml
COPY skills/multi-llm/uv.lock         /opt/mllm/skill/uv.lock
COPY skills/multi-llm/SKILL.md        /opt/mllm/skill/SKILL.md
COPY skills/multi-llm/utils           /opt/mllm/skill/utils
COPY skills/multi-llm/instructions    /opt/mllm/skill/instructions
COPY skills/multi-llm/references      /opt/mllm/skill/references
COPY skills/multi-llm/prompts         /opt/mllm/skill/prompts
COPY skills/multi-llm/schemas         /opt/mllm/skill/schemas
COPY skills/multi-llm/templates       /opt/mllm/skill/templates
COPY skills/multi-llm/assets/icons    /opt/mllm/skill/assets/icons
# omp's --config overlay, by its own named path: `assets` is never copied
# whole. OmpProvider.build_command renders this file's ABSOLUTE path on every
# dispatch, and a missing overlay is a launch failure by design -- exit 1 in
# 0.13 s with `error: Config overlay not found: <path>` and zero requests -- so
# without this line every container-lane omp row dies before it reaches the
# stub, whatever the build context carries.
COPY skills/multi-llm/assets/omp      /opt/mllm/skill/assets/omp
COPY skills/multi-llm/hosts           /opt/mllm/skill/hosts
# The stub server resolves a scenario through the SHARED matcher rather than a
# second implementation of it -- `sys.path.insert(0, "/opt/mllm/skill")` and then
# `from tests.scenario_matcher import ...` -- so these two files, and only these
# two, cross from the test tree into the image. The rest of the suite stays out:
# the driver runs on the host and nothing inside the image imports pytest.
COPY skills/multi-llm/tests/__init__.py         /opt/mllm/skill/tests/__init__.py
COPY skills/multi-llm/tests/scenario_matcher.py /opt/mllm/skill/tests/scenario_matcher.py

# The edit-capable fake, and it crosses for ONE row: C4's option (a), the
# applied-bytes row (`test_container_full_chain.py`). Every redirected CLI on
# this lane answers from a canned fixture and therefore emits no edit tool call,
# so no chain row can move a byte -- the `applied` branch of the write scheduler
# is unreachable from the stub by construction, and option (c) (a stub that
# emits protocol-specific tool calls) is rejected as simulating an agent. This
# binary is the only thing in the image that can make a real byte change under
# the executor, `--mock-runner` in entrypoint.py is the only thing that puts it
# on PATH, and it shadows exactly one binary name on exactly that row. It reads
# scenario_matcher.py out of the directory above it, which is why the two travel
# together.
COPY skills/multi-llm/tests/mocks               /opt/mllm/skill/tests/mocks

# The e2e fixture tree crosses too, and it is rig INPUT rather than test code:
# `seed_workspace.py` copies `plans/auth-feature.md` into the sample project (the
# plan reuses that exact file so the container lane and the instruction-refactor
# guard cannot drift), and the stub resolves a scenario's `fixture:` paths
# relative to the scenario file, so the responses have to sit beside the
# scenarios. Still no pytest, and still no test module: the driver runs on the
# host. `tests/fixtures/**` is already inside context_manifest.txt, so the
# build-context guard covers this COPY unchanged.
COPY skills/multi-llm/tests/fixtures/e2e        /opt/mllm/skill/tests/fixtures/e2e

# THE ONE POINT IN THE PIPELINE WHERE THE NETWORK EXISTS. At run time the root
# filesystem is read-only AND the provider lane has --network none, so a `uv run`
# that decides it needs to sync does not even get as far as complaining about the
# filesystem: it has no index to reach either, and the row dies in resolution
# with a stack that looks nothing like the bug it is.
#
# `--locked`, not `--frozen`: --frozen uses the lockfile without checking it, so
# a uv.lock out of step with pyproject.toml would sail through here and fail four
# phases later inside a container nobody can attach to. --locked is --frozen plus
# the staleness assertion, which is the half that has to be a BUILD failure.
#
# compileall because the read-only root means __pycache__ can never be written at
# run time; Python swallows that failure and re-parses every module on every one
# of the eleven phases instead.
RUN /opt/mllm/real/uv sync --project /opt/mllm/skill --locked --no-dev \
      --python /usr/bin/python3 \
 && /opt/mllm/skill/.venv/bin/python -c "import yaml; print('skill env ok', yaml.__version__)" \
 && /usr/bin/python3 -m compileall -q /opt/mllm/skill >/dev/null \
 && rm -rf /tmp/uv-cache

# --------------------------------------------------------------------------- #
# 10b. The opencode PLUGIN route -- the host lane's SECOND way in.
#
# `install-host opencode` writes a COMMAND file ($HOME/.config/opencode/commands/
# multi-llm.md), and that is the only route any opencode row has ever graded. The
# npm package `opencode-multi-llm` is the other one, and it is the route
# hosts/capabilities.yaml calls recommended: it registers the same `/multi-llm`
# command AND a model-callable `multi_llm` tool, out of a plugin module rather
# than out of a file the host happens to scan. The two resolve differently -- the
# plugin overwrites a file shim's command entry wholesale and in memory -- so a
# lane that grades only the shim grades only half of what ships.
#
# WHY A DIRECTORY AND NOT THE .tgz. MEASURED 2026-09-19 against opencode 1.18.31,
# four spec forms, each on a clean $HOME, verdict read from `opencode debug
# config | jq '.command["multi-llm"] != null'`:
#
#     <dir>                  -> LOADED
#     file://<dir>           -> LOADED
#     <dir>/<pkg>.tgz        -> NOT loaded (the command map came back empty)
#     file://<dir>/<pkg>.tgz -> NOT loaded
#
# opencode never opens the archive. A `.tgz` spec resolves to the archive's
# CONTAINING DIRECTORY, which is why pointing at a tarball appears to work when
# that directory is itself an unpacked package -- confirmed by a fifth run with
# the tarball copied INSIDE the package dir, which loaded. So the image ships the
# unpacked tree and names it directly: no tarball, and no dependence on a
# behaviour upstream never documented.
#
# THE PAYLOAD CARRIES ITS OWN SKILL COPY AND SHARES §10's VIRTUALENV, which is
# not the arrangement this layer was first written for and is worth the six lines
# it takes to say why. The plugin resolves SKILL_DIR as
# <package>/skills/multi-llm -- never /opt/mllm/skill -- so the obvious move is a
# second `uv sync` under the payload. MEASURED: that sync creates nothing.
# `ENV UV_PROJECT_ENVIRONMENT=/opt/mllm/skill/.venv` is set image-wide above, so
# `uv sync --project <payload>` resolves the payload's lock INTO THE SKILL'S VENV,
# reports `Checked 1 package` and leaves `<payload>/.venv` absent -- which is
# exactly what the build then died asserting.
#
# It is also what happens at RUN time, and there it is correct: the plugin
# dispatches `uv run --frozen --no-dev --project <payload>/skills/multi-llm`, that
# same ENV redirects it to the same venv, and the payload's uv.lock is a copy of
# the skill's, so the environment genuinely satisfies it. One interpreter, one
# pyyaml, no second 500 MB tree.
#
# So this layer asserts the real dispatch instead of manufacturing a venv nothing
# would read. Running the payload's own `multi_llm.py --help` through the exact
# `uv run` the plugin composes proves two things a file check cannot: that the
# environment resolves offline, and that `build.py`'s ALLOWLIST STAGING carried
# every module the dispatcher imports -- the failure mode that shipped a modes
# table pointing at a module the tarball did not contain.
#
# The sources are BUILD OUTPUT: `packaging/opencode/build.py` stages dist/ and
# skills/, both .gitignore'd. `make e2e-image` stages them before it builds, and
# context_manifest.txt marks them `# optional` so the clean-checkout build-context
# guard still passes.
# --------------------------------------------------------------------------- #
COPY packaging/opencode/package.json /opt/mllm/opencode-plugin/package.json
COPY packaging/opencode/dist         /opt/mllm/opencode-plugin/dist
COPY packaging/opencode/skills       /opt/mllm/opencode-plugin/skills
RUN test -f /opt/mllm/opencode-plugin/skills/multi-llm/uv.lock \
 && /opt/mllm/real/uv run --frozen --no-dev \
      --project /opt/mllm/opencode-plugin/skills/multi-llm \
      -- python /opt/mllm/opencode-plugin/skills/multi-llm/multi_llm.py --help >/dev/null \
 && echo "plugin payload dispatches (uv run --frozen --no-dev, shared venv)" \
 && node -e 'const m=require("/opt/mllm/opencode-plugin/package.json"); if(m.name!=="opencode-multi-llm"){console.error("staged package is "+m.name);process.exit(1)}; console.log("plugin package "+m.name+"@"+m.version+" main="+m.main)' \
 && for f in dist/index.mjs dist/command.md dist/cli-surface.json; do \
        test -s "/opt/mllm/opencode-plugin/$f" \
          || { echo "the staged plugin is missing $f -- run 'make opencode-package' before building this image" >&2; exit 1; }; \
    done \
 && /usr/bin/python3 -m compileall -q /opt/mllm/opencode-plugin/skills >/dev/null \
 && chmod -R a+rX /opt/mllm/opencode-plugin \
 && du -sh /opt/mllm/opencode-plugin \
 && rm -rf /tmp/uv-cache

# --------------------------------------------------------------------------- #
# 11. The verification layer: a pin is a request, this is the evidence (C9).
#
# Runs every CLI's version command plus node/npm/uv, diffs the result against
# cli-versions.lock, and FAILS THE BUILD on any mismatch -- including for a CLI
# whose installer resolved a version we did not name. It also asserts C2's
# load-bearing invariant separately: every CLI resolves inside /opt/mllm/bin and
# every one of those entries really is wrapper.py.
#
# Under MLLM_PIN_MODE=latest the CLI rows are recorded rather than enforced --
# that build asked for a moving CLI layer on purpose -- while node, npm, uv and
# both PATH checks stay enforced in both modes.
# --------------------------------------------------------------------------- #
COPY skills/multi-llm/tests/container/cli-versions.lock  /opt/mllm/lock/cli-versions.lock
COPY skills/multi-llm/tests/container/verify_versions.py /opt/mllm/rig/verify_versions.py
RUN /usr/bin/python3 /opt/mllm/rig/verify_versions.py \
        --lock /opt/mllm/lock/cli-versions.lock \
        --mode "${MLLM_PIN_MODE}" \
        --out /opt/mllm/lock/cli-versions.installed.json \
        --home /tmp/mllm-verify-home \
 && rm -rf /tmp/mllm-verify-home /tmp/mllm-inv \
 && chmod -R a+rX /opt/mllm/bin /opt/mllm/lock /opt/mllm/rig /opt/mllm/skill
# Scoped to the four directories the harness owns, deliberately: `chmod -R` over
# /opt/mllm as a whole means touching every inode of a 3.6 GB node_modules tree
# and a 558 MB virtualenv, which cost SEVEN MINUTES of build time and fixed
# nothing -- npm, uv and `cp -a` already produce world-readable trees under the
# build's umask, and the offline verify above runs as 10001 and proves it.

# --------------------------------------------------------------------------- #
# 12. Posture.
# --------------------------------------------------------------------------- #
WORKDIR /work/sample-project

# The entrypoint takes --lane/--host/--row/... from harness.py, which appends
# them after the image name. Invoked through the interpreter rather than the
# shebang so the image does not depend on /usr/bin/env resolving inside a
# stripped PATH.
ENTRYPOINT ["/usr/bin/python3", "/opt/mllm/rig/entrypoint.py"]

# Nothing baked into this image can be root-only by accident.
USER 10001:10001
