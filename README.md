# Državni posao

This repository contains two connected workflows for *Državni posao* episodes:

1. a WhisperX-based transcription pipeline with optional diarization and
   voice-based character assignment, tracked in MLflow; and
2. a scene-search application that retrieves the most likely **episode, time
   interval, and transcript evidence** for a natural-language description.

The reference search build contains 60 episodes and 1,473 searchable segments.
Search operates at scene level instead of treating an entire episode as one
document.

## Installation

Python 3.12+ and [uv](https://docs.astral.sh/uv/) are recommended.

```bash
# Core pipeline and test dependencies
uv sync --extra test

# Search, semantic retrieval, Streamlit, and tests
uv sync --extra search --extra test
```

The `search` extra installs Sentence Transformers and Streamlit. The
transcription pipeline's WhisperX, PyTorch, MLflow, and Pydantic dependencies
are base-project dependencies.

### Runtime asset contract

Git tracks only curated search inputs: episode metadata and evaluation queries.
Transcript exports, conformed corpus files, indexes, and evaluation outputs are
generated artifacts and are ignored. Build them locally with the `dp-*`
commands below or publish them to an external artifact store.

Built wheels and source distributions are intentionally code-only. When running
outside this checkout, provide asset locations explicitly with `--segments`,
`--bm25`, `--vectors`, `--queries`, and `--additional-queries`. For Streamlit,
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

`MediaStore` indexes `*.mp3` files recursively, applies the configured opening
and closing trims to every episode, and writes each completed transcript as
both JSON and text beneath its run directory.

Use `PipelineConfig.from_media(...)` to keep trimming configuration aligned
with the store, construct an `MlflowTracker` for an existing MLflow experiment,
then call `run_pipeline`:

```python
from drzavni_posao import MediaStore, MlflowTracker, PipelineConfig
from drzavni_posao import TranscriberConfig, run_pipeline

media = MediaStore(
    audio_dir="audio",
    run_dir="transcripts",
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

The corpus is built locally from one canonical transcript snapshot per episode.
It preserves original transcript evidence for display and derives normalized
lexical and semantic fields for retrieval. The segmenter performs Unicode NFC
cleanup, Serbian Cyrillic-to-Latin transliteration, diacritic folding,
controlled aliases, and overlapping, gap-aware chunks of up to eight source
utterances and 60 seconds.

The reference build uses a 25-second source-time offset because the acquisition
notebook removed the intro before transcription. Segment records therefore
preserve both transcript-relative and video-relative timestamps.

To rebuild the corpus and BM25 index from the complete named snapshot:

```bash
uv run dp-segment \
  --input transcripts/d69ff97f/named \
  --episodes-output data/episodes.jsonl \
  --segments-output data/segments.jsonl \
  --metadata data/episode_metadata.jsonl \
  --source-time-offset 25

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
search or evaluation reranker) is passed. `--hashing-smoke` builds a
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

Run the browser interface:

```bash
uv run --extra search streamlit run drzavni_posao/search/app/streamlit.py
```

After those generated files exist, the application reads
`data/segments.jsonl`, `artifacts/bm25_index.json`, and
`artifacts/vector_index` by default. Override those locations with
`DP_SEGMENTS_PATH`, `DP_BM25_PATH`, and `DP_VECTOR_PATH`. Its initial selection
is BM25 even when vector metadata is present. It loads a semantic model only
after dense or hybrid search is selected, always local-only, and presents a
user-facing error rather than downloading one. `DP_RERANKER_MODEL` and
`DP_RERANKER_REVISION` select an exact locally cached reranker snapshot.

Optional episode metadata can add a `youtube_url` and source offset while
segmenting:

```json
{"episode_id":"DP-E0001","youtube_url":"https://www.youtube.com/watch?v=REAL_ID","source_time_offset_seconds":25}
```

When a URL is present, results create timestamped links and the Streamlit app
embeds the video from the scene's video-relative start time. All 60 bundled
episode-metadata records include source YouTube URLs.

## Evaluation

The benchmark contains 120 manually written, interval-labeled queries: 28
development queries and 92 test queries, with two independently phrased cases
for every corpus episode. It includes Serbian Latin and Cyrillic, omitted
diacritics, typos, partial memories, topics, names, and remembered quotations.

Evaluate the untouched test split:

```bash
uv run --extra search dp-evaluate --queries data/evaluation_queries.jsonl --additional-queries data/evaluation_queries_expansion.jsonl --segments data/segments.jsonl --bm25 artifacts/bm25_index.json --vectors artifacts/vector_index --methods bm25,dense,hybrid --split test --output artifacts/evaluation_results.json
```

Evaluation reports Recall@5, MRR@10, nDCG@10, mean/p95 latency, and bootstrap
95% confidence intervals. Relevance requires both the correct episode and an
overlapping temporal window; a retrieved segment can satisfy a judgment only
once. Use the `dev` split for tuning aliases, chunking, and fusion weights, and
report only the untouched `test` result. Read metric values from the written
evaluation artifact, not from example numbers or historical reports.

BM25 is also the `dp-evaluate` default. Evaluation outputs include a complete
run manifest: query, corpus, index, and vector-file hashes; vector and reranker
model revisions; fusion and reranking weights; ranking, metric, and bootstrap
settings; runtime/package versions; and Git commit/dirty state. Outputs remain
generated artifacts under `artifacts/`; do not commit them. Publish benchmark
results to MLflow, object storage, or another artifact store when they need to
be shared.

## Data contracts

A segment record contains provenance, retrieval fields, and both time bases:

```json
{
  "episode_id": "DP-E0022",
  "episode_title": "Košarka",
  "segment_id": "DP-E0022_0014",
  "start_seconds": 116.3,
  "end_seconds": 142.1,
  "video_start_seconds": 141.3,
  "video_end_seconds": 167.1,
  "youtube_url": "https://www.youtube.com/watch?v=REAL_ID",
  "text_original": "[Đorđe Čvarkov] ...",
  "text_semantic": "Košarka Đorđe Čvarkov ...",
  "text_normalized": "kosarka cvarkov ...",
  "speakers": ["Đorđe Čvarkov", "Boškić"],
  "source_segment_indices": [42, 43, 44]
}
```

## Repository layout

```text
drzavni_posao/
  pipeline.py               # orchestration API
  tracking.py               # MLflow parent/episode tracking
  transcriber.py            # WhisperX ASR and alignment
  diarizer.py               # pyannote speaker diarization
  character_assigner.py     # voice-gallery character labels
  search/
    conform/                # transcript JSON -> corpus records
    indexing/               # BM25 and vector artifacts
    retrieval/              # query processing, search, fusion, reranking
    evaluation/             # judgments, metrics, evaluation runner
    app/streamlit.py        # search interface
data/
  episode_metadata.jsonl   # curated input
  evaluation_queries*.jsonl
  episodes.jsonl           # generated, ignored
  segments.jsonl           # generated, ignored
artifacts/                  # generated indexes/results, ignored
transcripts/                # generated pipeline output, ignored
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
