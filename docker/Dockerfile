# Requires an explicitly selected CUDA development image AND a validated lock file.
# Intentionally no guessed image tag and no automatic dependency resolution.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
WORKDIR /workspace/llm-ft-repro
COPY docker/requirements.lock /tmp/requirements.lock
RUN python -m pip install --require-hashes -r /tmp/requirements.lock
COPY . .
RUN python -m pip install --no-deps --no-build-isolation -e .
ENV PYTHONUNBUFFERED=1
CMD ["python", "scripts/repro.py", "--help"]
