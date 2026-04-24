# syntax=docker/dockerfile:1.7
#
# AgentLab dashboard + CLI. Ships the ``agentlab`` entry point so operators
# can ``docker run agentlab run suite.yaml`` or ``docker run -p 8787:8787
# agentlab serve``.

FROM python:3.12-slim AS builder
WORKDIR /build

# System deps only for pip build; no git required (no VCS-sourced deps).
RUN pip install --no-cache-dir --upgrade pip build

COPY pyproject.toml README.md LICENSE ./
COPY agentlab ./agentlab
RUN pip wheel --no-cache-dir --wheel-dir /wheels .

FROM python:3.12-slim AS runner
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AGENTLAB_DB=/data/runs.db

RUN useradd --system --uid 10001 --no-create-home agentlab \
 && mkdir -p /data \
 && chown -R agentlab:agentlab /data

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl \
    'openai>=1.50' 'anthropic>=0.40' \
 && rm -rf /wheels

# Ship example suites so the image is runnable out of the box.
COPY --chown=agentlab:agentlab examples /app/examples

USER agentlab
VOLUME ["/data"]
EXPOSE 8787

HEALTHCHECK --interval=30s --timeout=3s --retries=3 --start-period=5s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8787/api/runs').read()" || exit 1

ENTRYPOINT ["agentlab"]
CMD ["serve", "--port", "8787", "--db", "/data/runs.db"]
