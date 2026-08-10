# Državni posao — frontend

Newspaper-styled React UI for the scene search engine (ported from the Wonder
design, right artboard). Self-contained so it can later be packaged as its own
Docker image next to a search-backend image.

## Run

```bash
npm install
npm run dev        # http://localhost:5173
npm run build      # typecheck + production build to dist/
```

## Backend wiring

The app reads `VITE_API_BASE_URL` (see `.env.example`). When unset it serves
mocked results (`src/api/mock.ts`) so the UI runs standalone. When set, it
POSTs to `${VITE_API_BASE_URL}/search` with:

```json
{ "query": "...", "method": "bm25|dense|hybrid|hybrid_rerank",
  "top_k": 5, "candidate_k": 50, "group_by_episode": true }
```

and expects `{ took_seconds, results: [...] }` where each result matches
`SearchResult.to_dict()` from `drzavni_posao/search/retrieval/models.py`
(see `src/api/types.ts`). The HTTP service lives in
`drzavni_posao/search/api/app.py` — run it with `uv run dp-serve` (port 8000)
and set `VITE_API_BASE_URL=http://localhost:8000` in `.env.local`.

## Docker

`docker compose up --build` at the repo root serves the built frontend on
http://localhost:8080 with nginx proxying `/search` and `/health` to the
backend container (built with `VITE_API_BASE_URL=/`, i.e. same-origin).
