# Plan: HTTP search API + Docker packaging for drzavni_posao

Handoff plan for a fresh session. Goal: build the thin HTTP service the React
frontend (`frontend/`) already targets, then (phase 2) package frontend and
backend as two Docker images where the frontend depends on the backend.

## Context a fresh session needs

- Repo: uv-managed Python 3.12+ project (`pyproject.toml`, setuptools packaging,
  extras: `search`, `test`). Note the `[[tool.uv.index]]` Databricks pypi proxy —
  relevant for Docker builds.
- Search engine already exists and is complete — do NOT touch retrieval logic:
  - `drzavni_posao/search/retrieval/engine.py` — `SearchEngine.from_paths(*,
    segments_path, bm25_path, vector_path, dense_device="auto",
    local_models_only=True, reranker=None)` and
    `engine.search(query_text, *, method, top_k=10, candidate_k=50,
    group_by_episode=True, ...) -> list[SearchResult]`.
  - Methods: `bm25 | dense | hybrid | cross_encoder | hybrid_rerank`. Expose only
    the first four **minus** `cross_encoder` (it scores the whole corpus; the
    Streamlit app also hides it). The four exposed values must match the
    frontend's `SearchMethod` in `frontend/src/api/types.ts`.
  - `SearchResult.to_dict()` (`drzavni_posao/search/retrieval/models.py`) is the
    per-result JSON shape the frontend already types. Use it verbatim.
- Reference implementation for engine loading/config:
  `drzavni_posao/search/app/streamlit.py` — env vars `DP_SEGMENTS_PATH`
  (default `data/segments.jsonl`), `DP_BM25_PATH` (`artifacts/bm25_index.json`),
  `DP_VECTOR_PATH` (`artifacts/vector_index`), `DP_RERANKER_MODEL`,
  `DP_RERANKER_REVISION`, `DP_DEVICE`. It caches one engine per
  `(use_vectors, use_reranker)` combo and checks vector-index availability via
  the metadata file's existence. Mirror this pattern.
- Artifacts are gitignored and may be absent: build with `dp-segment` →
  `dp-build-bm25` → `dp-build-vectors` (the last needs `--allow-download` once
  to fetch `intfloat/multilingual-e5-base`; everything is local-only afterwards).
- Frontend contract (`frontend/README.md`, `frontend/src/api/client.ts`):
  - `POST /search` body `{query, method, top_k, candidate_k, group_by_episode}`
  - response `{took_seconds, results: [SearchResult.to_dict()...]}`
  - Frontend calls it when `VITE_API_BASE_URL` is set; dev server runs on
    `http://localhost:5173` → CORS must allow it.

## Phase 1 — FastAPI service

New package `drzavni_posao/search/api/` (keep it as thin as the Streamlit app):

1. `app.py` — FastAPI application:
   - `POST /search`: pydantic request model
     `SearchRequest(query: str, method: Literal["bm25","dense","hybrid","hybrid_rerank"] = "hybrid", top_k: int = 5 (1–20), candidate_k: int = 50 (10–100), group_by_episode: bool = True)`.
     Validate non-empty query (422 otherwise). Time the call, return
     `{"query", "method", "took_seconds", "results": [r.to_dict() for r in results]}`.
     Map engine errors (missing artifacts, unavailable vector index for
     dense/hybrid methods) to 503 with a clear message — the frontend renders
     `detail` strings inline.
   - `GET /health`: `{"status": "ok", "methods": [...]}` where methods reflects
     what's actually loadable (bm25 always; dense/hybrid only if the vector
     metadata file exists — reuse the Streamlit `_vector_metadata_exists` idea).
   - Engine caching: module-level `functools.lru_cache` keyed on
     `(use_vectors, use_reranker)` exactly like `load_engine` in
     `app/streamlit.py`; construct via `SearchEngine.from_paths` with the `DP_*`
     env vars. Lazy — don't load the dense model until a dense/hybrid request.
   - CORS middleware: origins from env `DP_CORS_ORIGINS`
     (comma-separated, default `http://localhost:5173`).
2. `pyproject.toml`: add extra `api = ["fastapi>=0.115", "uvicorn>=0.30"]`
   (or fold into `search`), and console script
   `dp-serve = "drzavni_posao.search.api.app:main"` where `main()` runs uvicorn
   on `0.0.0.0:8000` (port via `DP_API_PORT`).
3. Tests `tests/search/test_api.py` using `fastapi.testclient`:
   - follow the existing pattern in `tests/search/test_app.py` for isolating
     from torch imports (recent commit "isolate streamlit app tests from torch
     imports") — inject a tiny in-memory engine or monkeypatch the engine
     loader rather than loading real models.
   - cover: happy path bm25 (response shape matches `SearchResult.to_dict()`
     keys), empty query → 422, unknown method → 422, vector index missing +
    `method=dense` → 503, /health reports methods.

Verify phase 1 end-to-end:
```bash
uv sync --extra search --extra api --extra test
uv run pytest tests/search/test_api.py
uv run dp-serve &
curl -s localhost:8000/health
curl -s localhost:8000/search -X POST -H 'Content-Type: application/json' \
  -d '{"query":"svađa oko izbora","method":"bm25"}' | python -m json.tool
# then the real integration check:
cd frontend && echo 'VITE_API_BASE_URL=http://localhost:8000' > .env.local && npm run dev
# → run a query in the browser; results should come from the API, not the mock
```
(bm25 works with just `data/segments.jsonl` + `artifacts/bm25_index.json`;
rebuild via `uv run dp-segment && uv run dp-build-bm25` if absent.)

## Phase 2 — Docker images

1. `Dockerfile.search` (or `docker/search.Dockerfile`):
   - `python:3.12-slim`, install via uv (`uv sync --extra search --extra api
     --no-dev`); the Databricks index in `[[tool.uv.index]]` may be
     unreachable from a container build — pass `UV_INDEX_URL`/build-arg to fall
     back to pypi.org if needed.
   - COPY `data/segments.jsonl` + `artifacts/` into the image (build them on the
     host first) — or mount them as a volume; pick COPY for "contained units".
   - Pre-warm the HF cache for `intfloat/multilingual-e5-base` (and the
     reranker if desired) in a build stage, since runtime is
     `local_models_only=True`. Torch makes this image large; that's expected.
   - `EXPOSE 8000`, CMD `dp-serve`.
2. `frontend/Dockerfile`: standard two-stage — `node:22-alpine` build
   (`npm ci && npm run build`; `VITE_API_BASE_URL` is a **build arg** since Vite
   inlines it) → `nginx:alpine` serving `dist/`. Optionally have nginx proxy
   `/search` + `/health` to the backend service instead of baking an absolute
   URL — cleaner for compose and avoids CORS entirely; if so, set
   `VITE_API_BASE_URL=` empty… note the frontend treats empty as "use mock", so
   for the proxy approach set `VITE_API_BASE_URL=/` or adjust
   `frontend/src/api/client.ts` to distinguish "same-origin" from "mock"
   (small change: e.g. `VITE_API_MODE=proxy`).
3. `docker-compose.yml`: `search` (port 8000) + `frontend` (port 8080,
   depends_on search). Set `DP_CORS_ORIGINS` accordingly if not using the proxy.

Verify phase 2: `docker compose up --build`, open `http://localhost:8080`, run a
query for each of the four methods, confirm results and timestamped YouTube
links; `curl localhost:8000/health` shows all methods available.

## Out of scope
- No changes to retrieval/evaluation code, the Streamlit app, or the frontend
  UI (only the small client.ts tweak if the nginx-proxy option is chosen).
