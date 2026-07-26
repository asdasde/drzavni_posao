import re

import numpy as np

from .episode import Episode


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


def assignment_metrics(
    episode: Episode,
    anonymous_label: str = "[anonymous]",
) -> dict[str, float]:
    labels = [
        segment.get("speaker")
        for segment in episode.transcript.segments
    ]
    named = [
        label
        for label in labels
        if label and label != anonymous_label
    ]

    return {
        "named_segments": len(named),
        "anonymous_segments": len(labels) - len(named),
        "anonymous_share": (len(labels) - len(named)) / len(labels) if labels else 0.0,
        "assigned_characters": len(set(named)),
    }

