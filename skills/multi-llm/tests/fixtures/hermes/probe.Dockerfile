ARG BASE=mllm-e2e:local
FROM ${BASE}
USER root
COPY hermes-v0.23.4.tar.gz /tmp/hermes.tgz
RUN echo "08096c25464709e21b11f20611863ba6bc80ede7fbf2664f47fc98963da92c22  /tmp/hermes.tgz" | sha256sum -c - \
 && mkdir -p /opt/hermes/src && tar -xzf /tmp/hermes.tgz -C /opt/hermes/src --strip-components=1 \
 && cd /opt/hermes/src && UV_PYTHON_DOWNLOADS=automatic UV_PYTHON_INSTALL_DIR=/opt/hermes/python UV_PROJECT_ENVIRONMENT=/opt/hermes/venv /opt/mllm/real/uv sync --locked --no-dev --python 3.14 --managed-python \
 && ln -sf /opt/hermes/venv/bin/hermes /usr/local/bin/hermes && chmod -R a+rX /opt/hermes && rm /tmp/hermes.tgz
USER e2e
USER root
RUN cd /opt/hermes/src && /opt/hermes/venv/bin/python scripts/write_install_stamp.py -o /opt/hermes/src/install-stamp.json \
      --commit 8fc0374447313e276f6b9a2764bfc750190c966a --base-version 0.23.4 --display-version 0.23.4 --distance 0 \
      --source build --update-mechanism external \
 && chmod a+r /opt/hermes/src/install-stamp.json
USER e2e
