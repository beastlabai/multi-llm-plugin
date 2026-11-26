# Throwaway capture image for the dsh adapter's fixtures (not the e2e image).
FROM node:24-bookworm-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-yaml git ca-certificates bubblewrap procps \
 && rm -rf /var/lib/apt/lists/*
ARG DSH_VERSION=0.2.2-rc.5
RUN npm install -g --no-fund --no-audit "@deepseek-ai/dsh@${DSH_VERSION}" \
 && test "$(dsh --version)" = "${DSH_VERSION}" \
 && npm cache clean --force
RUN useradd -u 10001 -m rig
