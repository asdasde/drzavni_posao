"""Build per-episode dialogue turns with premises and splits for the LSTM model."""

import argparse
import itertools
import json
import re
from collections import Counter
from pathlib import Path

from ...search.common.io import write_json
from ...transcription.constants import CORE_CHARACTERS
from ...transcription.store import DEFAULT_CATALOG_PATH, MediaStore
from ...transcription.transcript import Transcript
from ..dataset import (
    LONG_TURN_ISSUE,
    SPLITS,
    assign_folds,
    select_splits,
    transcript_quality_issues,
)
from .text import clean_dialogue_text, clean_premise_text


_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


def build_turns(
    transcript: Transcript,
    *,
    max_turn_chars: int,
    stats: Counter,
) -> list[dict[str, str]]:
    """Merge same-speaker segments, keep core characters, and split long turns.

    Turn boundaries come from the raw segments, so a segment rejected by
    cleaning never merges the turns around it. *stats* counts rejected segments
    by reason, dropped turns by speaker, and ``split_turns``.
    """
    segments = [s for s in transcript.segments if (s.get("text") or "").strip()]
    turns = []
    for speaker, group in itertools.groupby(segments, key=lambda s: s.get("speaker")):
        texts = []
        for segment in group:
            text, reason = clean_dialogue_text(segment["text"])
            if text is None:
                stats[reason] += 1
            else:
                texts.append(text)
        if not texts:
            continue
        if speaker not in CORE_CHARACTERS:
            stats[str(speaker)] += 1
            continue
        chunks = split_turn(" ".join(texts), max_turn_chars)
        if len(chunks) > 1:
            stats["split_turns"] += 1
        turns.extend({"speaker": speaker, "text": chunk} for chunk in chunks)
    return turns


def split_turn(text: str, max_chars: int) -> list[str]:
    """Split at sentence ends into chunks of at most *max_chars* characters."""
    if len(text) <= max_chars:
        return [text]
    pieces = []
    for sentence in _SENTENCE_END_RE.split(text):
        pieces.extend(_pack(sentence.split(" "), max_chars))
    return _pack(pieces, max_chars)


def _pack(pieces: list[str], max_chars: int) -> list[str]:
    chunks: list[str] = []
    for piece in pieces:
        while len(piece) > max_chars:
            chunks.append(piece[:max_chars])
            piece = piece[max_chars:]
        if chunks and len(chunks[-1]) + 1 + len(piece) <= max_chars:
            chunks[-1] = f"{chunks[-1]} {piece}"
        else:
            chunks.append(piece)
    return chunks


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def group_by_split(episodes: list[dict]) -> dict[str, list[dict]]:
    return {split: [e for e in episodes if e["split"] == split] for split in SPLITS}


def prepare_episodes(
    *,
    transcripts_dir: Path,
    premises_path: Path,
    catalog_path: Path,
    max_turn_chars: int,
) -> tuple[list[dict], dict]:
    media = MediaStore(audio_dir=None, run_dir=transcripts_dir, catalog_path=catalog_path)
    episodes = [
        e for e in media.episodes() if (transcripts_dir / "episodes" / e.key).is_dir()
    ]
    splits = select_splits(assign_folds(episodes), test_fold=0, validation_fold=1)
    premises = {r["episode_key"]: r["premise"] for r in read_jsonl(premises_path)}

    stats: Counter = Counter()
    excluded: dict[str, dict[str, object]] = {}
    records = []
    for episode in episodes:
        premise = clean_premise_text(premises.get(episode.key, ""))
        if not premise:
            excluded[episode.key] = {"reason": "missing_premise"}
            continue
        path = transcripts_dir / "episodes" / episode.key / "character_assigner"
        transcript = Transcript.from_dict(
            json.loads((path / "transcript.json").read_text(encoding="utf-8"))
        )
        # Long turns are split by build_turns, so they do not exclude the episode.
        issues = [i for i in transcript_quality_issues(transcript) if i != LONG_TURN_ISSUE]
        if issues:
            excluded[episode.key] = {"reason": "quality", "issues": issues}
            continue
        turns = build_turns(transcript, max_turn_chars=max_turn_chars, stats=stats)
        if not turns:
            excluded[episode.key] = {"reason": "no_turns"}
            continue
        records.append(
            {
                "episode_key": episode.key,
                "split": splits[episode.key],
                "premise": premise,
                "turns": turns,
            }
        )

    manifest = {
        "transcripts_dir": str(transcripts_dir),
        "premises_path": str(premises_path),
        "max_turn_chars": max_turn_chars,
        "episode_count": len(records),
        "excluded_episodes": excluded,
        "excluded_by_reason": dict(Counter(e["reason"] for e in excluded.values())),
        "stats": dict(stats),
        "splits": {
            split: {
                "episodes": len(group),
                "turns": sum(len(r["turns"]) for r in group),
                "characters": sum(len(t["text"]) for r in group for t in r["turns"]),
            }
            for split, group in group_by_split(records).items()
        },
    }
    return records, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--premises", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("data/lstm"))
    parser.add_argument("--max-turn-chars", type=int, default=300)
    args = parser.parse_args()

    records, manifest = prepare_episodes(
        transcripts_dir=args.transcripts,
        premises_path=args.premises,
        catalog_path=args.catalog,
        max_turn_chars=args.max_turn_chars,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_json(args.output_dir / "manifest.json", manifest)
    summary_keys = ("episode_count", "excluded_by_reason", "stats", "splits")
    summary = {key: manifest[key] for key in summary_keys}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
