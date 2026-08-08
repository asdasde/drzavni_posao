# Državni posao

Tools and notebooks for transcribing episodes of *Državni posao*, diarizing
speakers, assigning characters, and evaluating the results.

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```sh
uv sync
```

Run the checks with:

```sh
uv run ruff check .
uv run pytest
```

## Pipeline

The reusable pipeline lives in `drzavni_posao/` and consists of:

1. transcription and word alignment,
2. speaker diarization,
3. character assignment,
4. MLflow tracking and evaluation.

`notebooks/pipeline.ipynb` is the general pipeline entry point. The
`notebooks/large-v3-*.ipynb` notebooks contain reproducible experiments, while
comparison and EDA notebooks analyze completed MLflow runs.

Audio, transcripts, MLflow state, and generated artifacts are local data and
are intentionally excluded from Git.

## Data utilities

`uv run episodes.py` discovers season 3, uploads existing local MP3s, then
downloads and immediately uploads any missing episodes.

Use comma-separated `--season` or `--episode` values to select specific media,
and `--dry-run` to preview the work.

## Databricks

See [`docs/databricks-sync.md`](docs/databricks-sync.md) for package, notebook,
and audio upload commands.
