"""Compare pipeline transcripts against a second ASR system's output.

Neither side is ground truth, so nothing here is a WER. What these metrics
measure is *disagreement*, which is where transcription errors concentrate:
the regions two independent systems dispute are the ones worth listening to
when building a gold set.

The reference format is an ElevenLabs Scribe export — a flat ``words`` list
whose entries carry ``text``/``start``/``end``/``speaker_id``/``logprob`` and a
``type`` of either ``word`` or ``spacing``.
"""

import difflib
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .episode import episode_order_key
from .transcript import Transcript

logger = logging.getLogger(__name__)

EPISODE_KEY = re.compile(r"Ep\.\s*(\d+)")

# Whisper omits these by design, so they are counted apart from lost content.
FILLERS = frozenset(
    {
        "a",
        "aha",
        "ah",
        "e",
        "eh",
        "ej",
        "hm",
        "ma",
        "mm",
        "o",
        "onaj",
        "ovaj",
        "pa",
        "uhm",
    }
)

CAST = {
    "Đorđe": (r"\bđor[đdžćc]\w*", r"đorđ\w*|đorđe"),
    "Boškić": (r"\bbo[šs]ki\w*", r"boški[ćc]\w*|boško"),
    "Torbica": (r"\bt\w{0,2}rbi\w*", r"torbic\w*|torbič\w*"),
    "Čvarkov": (r"\b[čc]varkov\w*", r"čvarkov\w*"),
    "Dobrosavljev": (r"\bdobrosa\w*", r"dobrosavljev\w*|dobrosavlje"),
    "Marijanović": (r"\bmarijan\w*", r"marijanović\w*"),
    "Šilovička": (r"\b(?:frau)?[šs]ilovi\w*", r"šilovičk\w*|šilović(?!k)\w*"),
    "Radosavljević": (r"\b\w*savljevi[ćčc]\w*", r"radosavljević\w*"),
}

_GAP_TOLERANCE = 0.05  # seconds of slack when testing word-in-gap containment


def scribe_to_transcript(export: dict) -> Transcript:
    """Re-shape a Scribe export into the pipeline's own transcript format.

    Scribe emits one flat list of ``word`` and ``spacing`` entries; the pipeline
    emits segments of words. Consecutive runs of the same ``speaker_id`` recover
    the segment level from data Scribe actually provides instead of from an
    invented pause threshold, and its labels land on the same ``speaker`` key
    the Diarizer writes, so both sides read identically from here on.

    Scribe's ``logprob`` is kept under its own name: WhisperX's ``score`` is a
    forced-alignment confidence in [0, 1] and the two are not comparable.
    """
    segments: list[dict] = []
    for entry in export["words"]:
        if entry.get("type") != "word":  # `spacing` entries are just whitespace
            continue
        speaker = entry.get("speaker_id")
        if not segments or segments[-1]["speaker"] != speaker:
            segments.append({"speaker": speaker, "words": []})
        segments[-1]["words"].append(
            {
                "word": entry["text"],
                "start": entry["start"],
                "end": entry["end"],
                "logprob": entry.get("logprob"),
            }
        )

    for segment in segments:
        segment["start"] = segment["words"][0]["start"]
        segment["end"] = segment["words"][-1]["end"]
        segment["text"] = " ".join(word["word"] for word in segment["words"])

    return Transcript(segments)


def scribe_transcripts(directory: Path) -> dict[str, Transcript]:
    """Episode key → Scribe export, converted to the pipeline transcript format."""
    found = {}
    for path in sorted(directory.glob("*.json")):
        match = EPISODE_KEY.search(path.name)
        if match is None:
            logger.warning("Skipping %s: no episode number in filename", path.name)
            continue
        export = json.loads(path.read_text(encoding="utf-8"))
        found[match.group(1)] = scribe_to_transcript(export)
    logger.info("Loaded %d Scribe transcripts from %s", len(found), directory)
    return found


def load_transcripts(directory: Path) -> dict[str, Transcript]:
    """Episode key → transcript for every ``<key>.json`` MediaStore wrote."""
    paths = sorted(directory.glob("*.json"))
    logger.info("Loaded %d transcripts from %s", len(paths), directory)
    return {
        path.stem: Transcript.from_dict(json.loads(path.read_text(encoding="utf-8")))
        for path in paths
    }


def words(transcript: Transcript) -> list[dict]:
    """Every word of a transcript, in order, whichever system produced it."""
    return [
        word for segment in transcript.segments for word in segment.get("words", [])
    ]


def text_of(words: list[dict]) -> str:
    return " ".join(word["word"] for word in words)


def _tokens(words: list[dict]) -> list[tuple[int, str]]:
    """(original index, comparable token) for each word that has content."""
    tokens = []
    for index, word in enumerate(words):
        token = re.sub(r"[^\w]", "", word["word"].lower())
        if token:
            tokens.append((index, token))
    return tokens


def vad_gaps(segments: list[dict], minimum_seconds: float = 0.3) -> list[tuple]:
    """Stretches between merged segments — audio the decoder never looked at."""
    blocks: list[list[float]] = []
    for start, end in sorted((s["start"], s["end"]) for s in segments):
        if blocks and start <= blocks[-1][1]:
            blocks[-1][1] = max(blocks[-1][1], end)
        else:
            blocks.append([start, end])
    return [
        (blocks[i][1], blocks[i + 1][0])
        for i in range(len(blocks) - 1)
        if blocks[i + 1][0] - blocks[i][1] > minimum_seconds
    ]


def turn_lengths(segments: list[dict]) -> dict[int, int]:
    """Word index → number of words in that word's speaker turn.

    Speakers live on segments in both formats, and consecutive segments can
    share one, so a turn is a run of segments rather than a single segment.
    """
    speakers = [
        segment.get("speaker") for segment in segments for _ in segment.get("words", [])
    ]
    lengths: dict[int, int] = {}
    start = 0
    for index in range(1, len(speakers) + 1):
        if index == len(speakers) or speakers[index] != speakers[start]:
            for position in range(start, index):
                lengths[position] = index - start
            start = index
    return lengths


@dataclass
class Comparison:
    """One episode's pipeline output lined up against the reference system."""

    key: str
    reference: list[dict]
    hypothesis: list[dict]
    reference_segments: list[dict]
    hypothesis_segments: list[dict]
    matched: set[int]  # indices into `reference` the pipeline reproduced
    counts: dict[str, int] = field(default_factory=dict)

    def dropped(self) -> list[dict]:
        """Reference words the pipeline did not reproduce."""
        return [
            word
            for index, word in enumerate(self.reference)
            if index not in self.matched
        ]

    def words_in_gaps(self, minimum_seconds: float = 0.3) -> list[dict]:
        """Reference words falling in a stretch the hypothesis left empty.

        This one stays directional even though the two sides now share a shape:
        Scribe emits a spacing entry for every inter-word interval and never
        leaves a hole longer than about a second, so reversing the comparison
        measures nothing. It is only informative against a VAD-gated system.
        """
        gaps = vad_gaps(self.hypothesis_segments, minimum_seconds)
        return [
            word
            for word in self.reference
            if any(
                start - _GAP_TOLERANCE <= word["start"]
                and word["end"] <= end + _GAP_TOLERANCE
                for start, end in gaps
            )
        ]

    def metrics(self) -> dict[str, float]:
        total = len(self.reference)
        if not total:
            return {}
        deleted = self.counts.get("deleted", 0)
        shrink = max(
            0,
            self.counts.get("replaced_reference", 0)
            - self.counts.get("replaced_hypothesis", 0),
        )
        dropped = self.dropped()
        fillers = sum(
            re.sub(r"[^\w]", "", word["word"].lower()) in FILLERS for word in dropped
        )
        return {
            "reference_words": total,
            "hypothesis_words": len(self.hypothesis),
            "word_deficit": (total - len(self.hypothesis)) / total,
            "agreement": len(self.matched) / total,
            "words_lost": deleted + shrink,
            "words_substituted": self.counts.get("replaced_reference", 0),
            "words_in_vad_gaps": len(self.words_in_gaps()),
            "vad_gaps": len(vad_gaps(self.hypothesis_segments)),
            "filler_share_of_dropped": fillers / len(dropped) if dropped else 0.0,
        }


def compare(key: str, reference: Transcript, hypothesis: Transcript) -> Comparison:
    """Align two transcripts of one episode, in either direction.

    Both sides are plain transcripts now, so this also compares two pipeline
    runs against each other, not only a run against Scribe.
    """
    reference_words = words(reference)
    hypothesis_words = words(hypothesis)
    reference_tokens = _tokens(reference_words)
    hypothesis_tokens = _tokens(hypothesis_words)

    matcher = difflib.SequenceMatcher(
        None,
        [token for _, token in reference_tokens],
        [token for _, token in hypothesis_tokens],
        autojunk=False,
    )

    matched: set[int] = set()
    counts: Counter = Counter()
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            matched.update(reference_tokens[i][0] for i in range(i1, i2))
            counts["equal"] += i2 - i1
        elif tag == "delete":
            counts["deleted"] += i2 - i1
        elif tag == "insert":
            counts["inserted"] += j2 - j1
        else:
            counts["replaced_reference"] += i2 - i1
            counts["replaced_hypothesis"] += j2 - j1

    return Comparison(
        key=key,
        reference=reference_words,
        hypothesis=hypothesis_words,
        reference_segments=reference.segments,
        hypothesis_segments=hypothesis.segments,
        matched=matched,
        counts=dict(counts),
    )


def _episode_order(key: str) -> tuple[bool, int, str]:
    """Sort numeric episode keys numerically, titled ones alphabetically after."""
    return episode_order_key(int(key) if key.isdigit() else None, key)


def compare_all(
    transcripts: dict[str, Transcript],
    references: dict[str, Transcript],
) -> list[Comparison]:
    """Compare every episode present in both *transcripts* and *references*."""
    shared = sorted(set(transcripts) & set(references), key=_episode_order)
    missing = sorted(set(references) - set(transcripts), key=_episode_order)
    if missing:
        logger.warning("No pipeline transcript for reference episodes: %s", missing)
    logger.info("Comparing %d episodes", len(shared))
    return [compare(key, references[key], transcripts[key]) for key in shared]


def disagreements(
    comparison: Comparison,
    maximum_span: int = 8,
) -> list[tuple[str, str]]:
    """(reference, pipeline) phrase pairs where the two systems differ.

    This is the adjudication worklist: listen to these spans, pick a winner,
    and you have gold-standard text without transcribing from scratch.
    """
    reference_tokens = _tokens(comparison.reference)
    hypothesis_tokens = _tokens(comparison.hypothesis)
    left = [token for _, token in reference_tokens]
    right = [token for _, token in hypothesis_tokens]

    pairs = []
    matcher = difflib.SequenceMatcher(None, left, right, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal" or (i2 - i1) > maximum_span or (j2 - j1) > maximum_span:
            continue
        pairs.append((" ".join(left[i1:i2]), " ".join(right[j1:j2])))
    return pairs


def spelling_accuracy(text: str, mentions: str, well_formed: str) -> dict:
    """How reliably *text* spells one recurring name."""
    hits = re.findall(mentions, text.lower())
    wrong = Counter(hit for hit in hits if not re.fullmatch(well_formed, hit))
    return {
        "mentions": len(hits),
        "well_formed": (len(hits) - sum(wrong.values())) / len(hits) if hits else 0.0,
        "variants": dict(wrong.most_common()),
    }


def cast_accuracy(text: str, cast: dict[str, tuple] | None = None) -> dict[str, dict]:
    """Spelling accuracy for every recurring cast member in *text*."""
    return {
        name: spelling_accuracy(text, mentions, well_formed)
        for name, (mentions, well_formed) in (cast or CAST).items()
    }
