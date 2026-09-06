# Državni posao

This repository contains two connected workflows for *Državni posao* episodes:

1. a WhisperX-based transcription pipeline with optional diarization and
   voice-based character assignment, tracked in MLflow; and
2. a scene-search application that retrieves the most likely **episode, time
   interval, and transcript evidence** for a natural-language description.

The segment count depends on the chunking settings. Search operates at scene
level instead of treating an entire episode as one document.

## Installation

Python 3.12+ and [uv](https://docs.astral.sh/uv/) are recommended.

```bash
# Transcription pipeline (WhisperX, MLflow, Jupyter) and tests
uv sync --extra transcribe --extra test

# Search, semantic retrieval, and tests
uv sync --extra search --extra test

# HTTP search API (FastAPI + uvicorn)
uv sync --extra search --extra api
```

The `search` extra installs Sentence Transformers. The
transcription pipeline's WhisperX, MLflow, and Jupyter dependencies live in the
`transcribe` extra (WhisperX's torchcodec dependency has no Linux arm64 wheels,
so keeping it optional lets the search API Docker image build on arm64).

### Data utilities

`uv run scripts/episodes.py` processes season 3 by default: it uploads existing local
MP3s, then downloads and uploads missing episodes. Use comma-separated
`--season` or `--episode` values and `--dry-run` to preview the work.

Download locally without accessing Databricks:

```bash
uv run scripts/episodes.py --local --season 3 --episode 1,10,11,12
```

See [`docs/databricks-sync.md`](docs/databricks-sync.md) for package, notebook,
and audio upload commands.

### Runtime asset contract

Git tracks only curated search inputs: episode metadata and evaluation queries.
Transcript exports, conformed corpus files, indexes, and evaluation outputs are
generated artifacts and are ignored. Build them locally with the `dp-*`
commands below or publish them to an external artifact store.

Built wheels and source distributions are intentionally code-only. When running
outside this checkout, provide asset locations explicitly with `--segments`,
`--bm25`, `--vectors`, and `--queries`. For the HTTP API,
set `DP_SEGMENTS_PATH`, `DP_BM25_PATH`, and `DP_VECTOR_PATH`.

## Transcription and MLflow pipeline

The transcription workflow processes MP3 episodes from a `MediaStore` through
these stages:

1. **Transcription** — WhisperX ASR plus forced alignment. Serbian
   transcription defaults to
   `classla/wav2vec2-xls-r-parlaspeech-hr` for alignment when no alignment
   model is configured.
2. **Diarization** (optional) — `pyannote/speaker-diarization-3.1` assigns
   speaker clusters to aligned transcript segments.
3. **Character assignment** (optional) — speaker embeddings are compared with
   enrolled voice clips. With diarization, each cluster is matched once; without
   it, each transcript segment is matched independently.

`MediaStore` loads `data/youtube_episodes.json`, matches each flat `*.mp3` by
the catalog's normalized `youtube_title`, and constructs episodes containing
both audio and catalog metadata. Files missing from the catalog are rejected.
Without an audio directory, it constructs episodes directly from the catalog
so transcript-only workflows can run. It applies configured trims and writes
completed transcripts beneath its run directory.

Use `PipelineConfig.from_media(...)` to keep trimming configuration aligned
with the store, construct an `MlflowTracker` for an existing MLflow experiment,
then call `run_pipeline`:

```python
from pathlib import Path

from drzavni_posao.transcription import MediaStore, MlflowTracker, PipelineConfig
from drzavni_posao.transcription import TranscriberConfig, run_pipeline

media = MediaStore(
    audio_dir=Path("audio"),
    run_dir=Path("transcripts"),
    trim_start=25,
    trim_end=0,
)
config = PipelineConfig.from_media(
    media,
    transcription=TranscriberConfig(
        model_name="large-v3",
        language="sr",
        device="cuda",
    ),
)
tracker = MlflowTracker("drzavni-posao")
run_id = run_pipeline(
    tracker,
    media,
    config,
    episode_keys={"1", "2"},
    tags={"dataset": "local"},
)
```

Add a `DiarizerConfig` and/or `CharacterAssignerConfig` to
`PipelineConfig.from_media(...)` when those stages are required. Character
assignment requires an `EmbedderConfig`, a minimum cosine similarity, and
enrolled `speaker_samples` containing `episode_key`, `start`, and `end`.

Pass `through=Stage.TRANSCRIPTION`, `Stage.DIARIZATION`, or
`Stage.CHARACTER_ASSIGNER` to stop after a configured stage. `run_pipeline`
resumes an exact configuration-level MLflow run: it skips completed episode
stages, restores the preceding transcript artifact before later stages, clears
episode audio after each attempt, and continues with other episodes if one
fails.

`MlflowTracker` records a configuration artifact and parameters at the parent
run, and creates nested episode runs. Each completed stage logs:

- `stage/transcript.json` and `stage/transcript.txt`;
- finite, stage-prefixed metrics;
- a `stage.<name>=complete` tag.

The parent receives per-metric means and `metrics-by-episode.json`. Credentials
such as Hugging Face tokens are excluded from the tracked configuration.
`MlflowTracker.fork_run(...)` can create a new configuration run while copying
validated artifacts through a selected stage, as long as the prerequisite
configuration is unchanged.

WhisperX and pyannote models may require local installation, GPU memory, and
Hugging Face authentication or access approval. Provide secrets through the
environment or runtime configuration; do not commit them.

## Search corpus and indexing

The corpus is built locally from a labeling export produced by
`scripts/export_labeling_run.py`. Each episode uses the latest available stage
transcript (`character_assigner`, then `diarization`, then `transcription`).
It preserves original transcript evidence for display and derives normalized
lexical and semantic fields for retrieval. Semantic passages are tagged as
`[Epizoda …] [Govore …]` plus dialogue so embeddings and the reranker see
episode and speaker metadata separately from the spoken lines. Changing that
format invalidates `artifacts/vector_index`; rebuild it after `dp-segment`.
The segmenter performs Serbian Cyrillic-to-Latin transliteration, whitespace
cleanup, diacritic folding, controlled aliases, and overlapping, gap-aware
chunks of up to eight source utterances and 60 seconds.

Video timestamps add `audio.trim_start` from the export's `config.json` unless
`--source-time-offset` is set. Segment records therefore preserve both
transcript-relative and video-relative timestamps.

To rebuild the corpus and BM25 index from an MLflow export:

```bash
uv run python scripts/export_labeling_run.py --run-id <run_id>

uv run dp-segment \
  --input data/mlflow_exports/<run_id> \
  --segments-output data/segments.jsonl \
  --catalog data/youtube_episodes.json

uv run dp-build-bm25 \
  --segments data/segments.jsonl \
  --output artifacts/bm25_index.json
```

The project uses a dependency-free Okapi BM25 index. Semantic retrieval uses
multilingual E5 embeddings and an exact cosine vector index. Build real vectors
with an explicit model-download opt-in:

```bash
uv run --extra search dp-build-vectors \
  --segments data/segments.jsonl \
  --output artifacts/vector_index \
  --model intfloat/multilingual-e5-base \
  --allow-download
```

Vector and reranker models are **local-only by default**. The commands do not
download models unless `--allow-download` (or `--allow-model-download` for a
search or evaluation dense encoder or reranker) is passed. `--hashing-smoke` builds a
dependency-free, non-semantic test index only; it must not be presented as
semantic retrieval. Use `--model-revision` and `--reranker-revision` to pin a
Hugging Face commit, tag, or branch.

Artifact manifests record lexical and semantic corpus fingerprints, segment
order, model name and resolved snapshot revision, embedding dimension, E5
prefix behavior, and format version. Loading rejects stale or differently
ordered indexes after either normalized BM25 text or embedded semantic text
changes.

## Scene search

BM25 needs no semantic model:

```bash
uv run dp-search "Boškić Čvarkov izbori"
```

BM25 is the command and Python API default, so the out-of-box path does not
require vectors or a model. Select `--method dense` or `--method hybrid`
explicitly when semantic assets are available.

Hybrid search combines BM25 candidates, multilingual dense candidates, and
character evidence:

```bash
uv run --extra search dp-search "the election argument" --method hybrid --vectors artifacts/vector_index
```

Use `--json` for machine-readable results, `--top-k` to control result count,
and `--all-segments` to return overlapping scene chunks rather than the
default one-result-per-episode grouping. Available methods are `bm25`, `dense`,
`hybrid`, `cross_encoder`, and `hybrid_rerank`; the last two require a
configured local cross-encoder or an explicit download opt-in.

BM25 and cosine values are calibrated within each candidate pool before hybrid
fusion (`0.45 BM25 + 0.45 dense + 0.10 character coverage`). Displayed scores
are **relative relevance scores**, not probabilities. Results include the
episode, video time interval, known characters, score components, an
evidence-based explanation, and original transcript text.

The canonical episode catalog supplies every title, broadcast date, and
YouTube URL. `--source-time-offset` remains a corpus-build override when the export
config does not match how the transcript audio was trimmed. Results use the catalog URL
to create timestamped links.

## HTTP search API

`drzavni_posao/search/api/app.py` is a thin FastAPI wrapper around the same
engine, built for the React frontend in `frontend/`:

```bash
uv sync --extra search --extra api
uv run dp-serve   # 0.0.0.0:8000, override with DP_API_PORT
```

`POST /search` accepts `{query, method, top_k, candidate_k, group_by_episode}`
(methods `bm25`, `dense`, `hybrid`, `hybrid_rerank`) and returns
`{query, method, took_seconds, results}` where each result is
`SearchResult.to_dict()`. `GET /health` lists the currently loadable methods
(`bm25` only until the vector index is built).

The API reads `data/segments.jsonl`, `artifacts/bm25_index.json`, and
`artifacts/vector_index` by default. Override those locations with
`DP_SEGMENTS_PATH`, `DP_BM25_PATH`, and `DP_VECTOR_PATH`. `DP_DEVICE` selects the
compute device; `DP_RERANKER_MODEL` and `DP_RERANKER_REVISION` select the reranker.
Models load from the local cache when their search method is requested.
`DP_CORS_ORIGINS` controls allowed frontend origins
(comma-separated; default `http://localhost:5173` for the Vite dev server).
Missing artifacts surface as HTTP 503 with an explanatory message.

## Browser frontend

With the API running, start the React frontend in another terminal:

```bash
cd frontend
npm install
VITE_API_BASE_URL=http://localhost:8000 npm run dev
```

Open http://localhost:5173. See [`frontend/README.md`](frontend/README.md) for
frontend configuration and production builds.

## Docker

With `data/segments.jsonl` and `artifacts/` built on the host:

```bash
docker compose up --build
```

serves the API on http://localhost:8000 (image `docker/search.Dockerfile`,
artifacts and Hugging Face models baked in) and the frontend on
http://localhost:8080 (`frontend/Dockerfile`, nginx proxying `/search` and
`/health` to the backend container — no CORS involved).

## Evaluation

The benchmark contains 55 real viewer questions extracted from public TikTok
reply videos. Each query has one relevant YouTube episode and retains the
public TikTok URL that establishes its lineage. Evaluation ranks episodes, not
transcript windows.

```bash
uv run --extra search dp-evaluate --queries data/evaluation_queries_tiktok.jsonl --segments data/segments.jsonl --bm25 artifacts/bm25_index.json --vectors artifacts/vector_index --methods bm25,dense,hybrid --output artifacts/evaluation_results_tiktok.json
```

The reported metrics are:

- **Recall@5:** fraction of queries whose relevant episode appears in the first
  five results. Each query contributes either 1 or 0.
- **MRR@10:** mean reciprocal rank of the relevant episode in the first ten:
  rank 1 scores 1, rank 2 scores 1/2, and a miss scores 0.
- **nDCG@10:** position-sensitive score using a logarithmic discount. With one
  binary relevant episode, rank `r` scores `1/log2(r+1)` and a miss scores 0.
  It penalizes lower ranks less sharply than MRR.
- **Mean latency:** arithmetic mean wall-clock search time per query.
- **P95 latency:** the latency at or below which 95% of measured queries fall.
- **95% confidence interval:** query-level bootstrap interval around each
  quality metric; it describes uncertainty from this finite query sample.

BM25 is also the `dp-evaluate` default. Evaluation outputs include a complete
run manifest: query, corpus, index, and vector-file hashes; vector and reranker
model revisions; fusion and reranking weights; ranking, metric, and bootstrap
settings; project package, Python, and platform versions; and Git commit/dirty
state. Outputs remain generated artifacts under `artifacts/`; do not commit
them. Publish benchmark results to MLflow, object storage, or another artifact
store when they need to be shared.

Each per-query result and checkpoint repeats its `tiktok_url`, so a reported
success or failure can be traced back to the source question.

## Data contracts

A segment record contains provenance, retrieval fields, and both time bases:

```json
{
  "youtube_video_id": "i3ONQLQQ0Po",
  "episode_number": 42,
  "episode_title": "Košarka",
  "segment_id": "i3ONQLQQ0Po_0014",
  "start_seconds": 116.3,
  "end_seconds": 142.1,
  "video_start_seconds": 141.3,
  "video_end_seconds": 167.1,
  "youtube_url": "https://www.youtube.com/watch?v=REAL_ID",
  "text_original": "[Đorđe Čvarkov] ...",
  "text_semantic": "[Epizoda Košarka] [Govore Đorđe Čvarkov, Boškić] ...",
  "text_normalized": "kosarka cvarkov ...",
  "speaker_keys": ["cvarkov", "boskic"],
  "source_segment_indices": [42, 43, 44]
}
```

## Repository layout

```text
drzavni_posao/
  transcription/
    pipeline.py             # orchestration API
    tracking.py             # MLflow parent/episode tracking
    transcriber.py          # WhisperX ASR and alignment
    diarizer.py             # pyannote speaker diarization
    character_assigner.py   # voice-gallery character labels
    comparison.py           # transcript comparison and alignment metrics
    constants.py            # shared transcription constants
    embedder.py             # speaker embeddings
    episode.py              # episode metadata and catalogs
    evaluation.py           # evaluation helpers
    gallery.py              # speaker sample galleries
    metrics.py              # transcription and assignment metrics
    store.py                # media and artifact storage
    transcript.py           # transcript model
    vad_realigner.py         # VAD boundary realignment
  search/
    conform/                # transcript JSON -> corpus records
    indexing/               # BM25 and vector artifacts
    retrieval/              # query processing, search, fusion, reranking
    evaluation/             # judgments, metrics, evaluation runner
    api/                    # HTTP API for the React frontend
frontend/                   # React search interface
data/
  youtube_episodes.json    # canonical episode catalog
  segments.jsonl           # generated, ignored
  mlflow_exports/          # local MLflow run dumps, ignored
artifacts/                  # generated indexes/results, ignored
transcripts/                # pipeline scratch on Databricks, ignored
tests/
```

## Limitations

- Forced-alignment timestamps are approximate and should be spot-checked
  against video before public release.
- Voice-based character labels are only as reliable as the enrolled clips and
  diarization; unresolved speaker labels remain anonymous.
- ASR errors, dialect, slang, and Serbian morphology affect lexical and
  semantic retrieval.
- Overlapping chunks improve boundary recall but can create correlated hits;
  evaluation maps judgments once and the UI suppresses near-duplicates.
- Semantic retrieval can return a plausible but incorrect scene. Hybrid lexical
  evidence and visible transcript snippets are safeguards, not guarantees.
- The benchmark is small. Keep scene paraphrases within their existing
  development/test partition and report uncertainty intervals.
