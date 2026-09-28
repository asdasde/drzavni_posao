#!/usr/bin/env bash
# Axis-A structure sweep: utterances-per-chunk x reduction, scored across all runs.
# Embeddings cache by (corpus, model, u), so each chunk size encodes once and the
# three reduction variants reuse it -> 3 encodes total, not 9.
#
# Lemmatization is on for every run, sharing one --lemma-cache: the lemma cache
# keys on whole-episode text (independent of utterances-per-chunk), so Stanza
# runs exactly once and all 9 runs reuse it.
set -euo pipefail

SEGMENTS="${1:-data/segments_topics.jsonl}"
ROOT="artifacts/full_sweep"
TOPICS="15,20,25,30,35,40,50,60,75,100"
SEEDS="42,43,44,45,46"
LEMMA_CACHE="$ROOT/lemmas.json"

run() {  # name, extra-args...
  local name="$1"; shift
  echo ">>> $name"
  uv run --extra search python scripts/run_dialogue_topic_sweep.py \
    --segments "$SEGMENTS" \
    --topics "$TOPICS" --seeds "$SEEDS" \
    --lemmatize --lemma-cache "$LEMMA_CACHE" --allow-lemma-download \
    --output-dir "$ROOT/$name" "$@"
}

for u in 8 12 16; do
  run "u${u}_none"          --utterances-per-chunk "$u"
  run "u${u}_pca128"        --utterances-per-chunk "$u" --pca-components 128
  run "u${u}_pca128_whiten" --utterances-per-chunk "$u" --pca-components 128 --pca-whiten
done

echo ">>> aggregate"
uv run --extra search python scripts/aggregate_sweeps.py "$ROOT"/*/ \
  --output "$ROOT/ranking.json" --top 20
