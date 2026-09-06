import re
from itertools import islice

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from .constants import ANONYMOUS_LABEL
from .episode import Episode

UNCOVERED_LABEL = "[uncovered]"


def alignment_metrics(episode: Episode) -> dict[str, float]:
    segments = episode.transcript.segments
    words = [
        word
        for segment in segments
        for word in segment.get("words", [])
    ]
    scores = [
        word["score"]
        for word in words
        if isinstance(word.get("score"), (int, float))
    ]
    rates = [
        len(segment.get("words", []))
        / (segment["end"] - segment["start"])
        for segment in segments
        if segment.get("words")
        and segment.get("end", 0) > segment.get("start", 0)
    ]

    metrics = {
        "word_count": len(words),
        "segment_count": len(segments),
    }

    if scores:
        p10, median, p90 = np.quantile(scores, [0.1, 0.5, 0.9])
        metrics.update(
            {
                "alignment_score_p10": p10,
                "alignment_score_median": median,
                "alignment_score_p90": p90,
            }
        )

    if rates:
        metrics["median_words_per_second"] = np.median(rates)

    return metrics


def repetition_metrics(
    episode: Episode,
    pathological_run_length: int = 6,
) -> dict[str, int]:
    maximum_run = 0
    pathological_segments = 0

    for segment in episode.transcript.segments:
        tokens = re.findall(
            r"\w+",
            (segment.get("text") or "").casefold(),
            flags=re.UNICODE,
        )

        segment_maximum = 0
        previous = None
        current_run = 0

        for token in tokens:
            if token == previous:
                current_run += 1
            else:
                previous = token
                current_run = 1
            segment_maximum = max(segment_maximum, current_run)

        maximum_run = max(maximum_run, segment_maximum)
        if segment_maximum >= pathological_run_length:
            pathological_segments += 1

    return {
        "maximum_consecutive_token_run": maximum_run,
        "pathological_repetition_segments": pathological_segments,
    }


def diarization_metrics(episode: Episode) -> dict[str, int]:
    segments = episode.transcript.segments
    words = [
        word
        for segment in segments
        for word in segment.get("words", [])
    ]
    turns = episode.diarization or []

    mixed_segments = sum(
        len(
            {
                word["speaker"]
                for word in segment.get("words", [])
                if "speaker" in word
            }
        )
        > 1
        for segment in segments
    )

    return {
        "turn_count": len(turns),
        "detected_speakers": len({turn["speaker"] for turn in turns}),
        "unassigned_words": sum("speaker" not in word for word in words),
        "mixed_speaker_segments": mixed_segments,
    }


def assignment_metrics(episode: Episode) -> dict[str, float]:
    labels = [
        segment.get("speaker")
        for segment in episode.transcript.segments
    ]
    named = [
        label
        for label in labels
        if label and label != ANONYMOUS_LABEL
    ]

    return {
        "named_segments": len(named),
        "anonymous_segments": len(labels) - len(named),
        "anonymous_share": (len(labels) - len(named)) / len(labels) if labels else 0.0,
        "assigned_characters": len(set(named)),
    }


def labeled_segments(
    labels: list[dict],
    predictions: list[dict],
    trim_start: float = 0.0,
) -> pd.DataFrame:
    """Cut labeled time into rows of labeled speaker, predicted speaker, seconds.

    Labels live in raw episode time, so predictions from a trimmed run are
    shifted by ``trim_start`` before they are compared.
    """
    labeled = sorted(
        (s["start"], s["end"], s["speaker"])
        for s in labels
        if s["end"] > s["start"]
    )
    predicted = sorted(
        (
            s["start"] + trim_start,
            s["end"] + trim_start,
            s.get("speaker") or ANONYMOUS_LABEL,
        )
        for s in predictions
        if s["end"] > s["start"]
    )

    rows = []
    first = 0
    for start, end, truth in labeled:
        while first < len(predicted) and predicted[first][1] <= start:
            first += 1
        cursor = start
        for predicted_start, predicted_end, label in islice(predicted, first, None):
            if predicted_start >= end:
                break
            # Clipping to the cursor keeps overlapping predictions from
            # counting the same second twice.
            left, right = max(predicted_start, cursor), min(predicted_end, end)
            if right <= left:
                continue
            rows.append((left, right, truth, label))
            cursor = right
        if end > cursor:
            rows.append((cursor, end, truth, UNCOVERED_LABEL))

    segments = pd.DataFrame(rows, columns=["start", "end", "truth", "predicted"])
    segments["seconds"] = segments["end"] - segments["start"]
    segments["correct"] = segments["truth"] == segments["predicted"]
    return segments


def confusion_table(labeled_segments: pd.DataFrame) -> pd.DataFrame:
    truth = labeled_segments["truth"]
    predicted = labeled_segments["predicted"]
    speakers = sorted(set(truth) | set(predicted))
    matrix = confusion_matrix(
        truth,
        predicted,
        labels=speakers,
        sample_weight=labeled_segments["seconds"],
    )
    return pd.DataFrame(
        matrix,
        index=pd.Index(speakers, name="truth"),
        columns=pd.Index(speakers, name="predicted"),
    )


def character_table(labeled_segments: pd.DataFrame) -> pd.DataFrame:
    truth = labeled_segments["truth"]
    characters = sorted(truth.unique())
    precision, recall, f1, seconds = precision_recall_fscore_support(
        truth,
        labeled_segments["predicted"],
        labels=characters,
        sample_weight=labeled_segments["seconds"],
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "labeled_seconds": seconds,
        },
        index=pd.Index(characters, name="truth"),
    )


def assignment_scores(labeled_segments: pd.DataFrame) -> dict[str, float]:
    if labeled_segments.empty:
        raise ValueError("No labeled seconds to score")

    truth = labeled_segments["truth"]
    predicted = labeled_segments["predicted"]
    seconds = labeled_segments["seconds"]
    covered = predicted != UNCOVERED_LABEL

    scores = {
        "accuracy": accuracy_score(truth, predicted, sample_weight=seconds),
        "coverage": seconds[covered].sum() / seconds.sum(),
        "covered_accuracy": (
            accuracy_score(
                truth[covered],
                predicted[covered],
                sample_weight=seconds[covered],
            )
            if covered.any()
            else None
        ),
        "weighted_f1": f1_score(
            truth,
            predicted,
            labels=sorted(truth.unique()),
            average="weighted",
            sample_weight=seconds,
            zero_division=0,
        ),
        "labeled_seconds": seconds.sum(),
    }

    return {
        name: float(value)
        for name, value in scores.items()
        if value is not None
    }
