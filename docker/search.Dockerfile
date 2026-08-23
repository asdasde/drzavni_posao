# Build search artifacts on the host first:
#   uv run dp-segment --input <transcripts dir> && uv run dp-build-bm25 && uv run dp-build-vectors --allow-download
# then: docker build -f docker/search.Dockerfile -t dp-search .
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev --extra search --extra api

COPY drzavni_posao ./drzavni_posao
RUN uv sync --frozen --no-dev --extra search --extra api

# Runtime is local_models_only=True, so bake the dense encoder (and reranker)
# into the image's HF cache while the build still has network access.
ARG DENSE_MODEL=intfloat/multilingual-e5-base
ARG RERANKER_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
RUN uv run python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('${DENSE_MODEL}'); \
CrossEncoder('${RERANKER_MODEL}')"

COPY data/segments.jsonl ./data/segments.jsonl
COPY artifacts ./artifacts

ENV DP_DEVICE=cpu
EXPOSE 8000
CMD ["uv", "run", "--no-sync", "dp-serve"]
