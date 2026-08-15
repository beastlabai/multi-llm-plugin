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
ARG GOOSE_VERSION=1.41.0
ARG GOOSE_URL=https://github.com/aaif-goose/goose/releases/download/v1.41.0/goose-x86_64-unknown-linux-gnu.tar.bz2
ARG GOOSE_SHA256=416de595f3e9e7c54a822bce06a79960b9387e3cd51a832a3eadf876237d5a1c
ARG GROK_VERSION=0.2.93
ARG GROK_URL=https://storage.googleapis.com/grok-build-public-artifacts/cli/grok-0.2.93-linux-x86_64
ARG GROK_SHA256=4e0738d3b5550f3c842bc0ae69f468815c6329c008a110d0c27a694dc3401135
ARG MUSE_VERSION=0.1.0-R708.1
ARG MUSE_URL=https://lookaside.facebook.com/lookaside/muse/download/?channel=muse&version=0.1.0-R708.1&file=muse-x86-linux
ARG MUSE_SHA256=50937b6470cd0edf28eb683c352a5e7af3bcb1b015cd9a3b21dbf79d22af8182

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
        codex claude gemini opencode cline kilocode pi \
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
# The .yaml sources are copied as a group rather than named one by one so that
# redirect.yaml -- which Phase 3 writes beside posture.yaml -- lands without a
# Dockerfile edit. Every file the glob can reach is inside context_manifest.txt.
# --------------------------------------------------------------------------- #
COPY skills/multi-llm/tests/container/entrypoint.py     /opt/mllm/rig/entrypoint.py
COPY skills/multi-llm/tests/container/stub_server.py    /opt/mllm/rig/stub_server.py
COPY skills/multi-llm/tests/container/wrapper.py        /opt/mllm/rig/wrapper.py
COPY skills/multi-llm/tests/container/seed_workspace.py /opt/mllm/rig/seed_workspace.py
COPY skills/multi-llm/tests/container/seed_cli_surface.py /opt/mllm/rig/seed_cli_surface.py
COPY skills/multi-llm/tests/container/*.yaml            /opt/mllm/rig/
COPY skills/multi-llm/tests/container/home_seed         /opt/mllm/rig/home_seed

RUN chmod 0755 /opt/mllm/rig/*.py \
 && for b in codex claude gemini opencode cline kilocode pi aider goose grok muse uv; do \
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
# NOTE for Phase 3: wrapper.py builds its child's environment from scratch and
# carries over only PATH/TERM/LANG/LC_ALL/TZ and MLLM_*, so these names do NOT
# reach a CLI dispatched *through* the wrapper. They cover the entrypoint, the
# build, and any direct invocation; the per-CLI copies belong in redirect.yaml.
ENV CLINE_NO_AUTO_UPDATE=1 \
    GROK_AUTO_UPDATE=0 \
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
