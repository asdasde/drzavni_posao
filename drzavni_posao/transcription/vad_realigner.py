from contextlib import contextmanager
from dataclasses import dataclass
from itertools import groupby, pairwise
from typing import Self


@dataclass(frozen=True, order=True)
class Segment:
    start: float
    end: float

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError("Segment end must be greater than start")

    @property
    def sort_key(self) -> tuple[float, float]:
        return self.start, self.end

    @property
    def duration(self) -> float:
        return self.end - self.start

    def intersection(self, other: "Segment") -> Self | None:
        start = max(self.start, other.start)
        end = min(self.end, other.end)
        return type(self)(start, end) if start < end else None

    def overlap_duration(self, other: "Segment") -> float:
        intersection = self.intersection(other)
        return intersection.duration if intersection is not None else 0.0

    def padded(self, padding: float) -> Self:
        return type(self)(max(0.0, self.start - padding), self.end + padding)


class VadSegment(Segment):
    pass


@dataclass(frozen=True)
class VadRegion:
    segments: tuple[VadSegment, ...]

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("VadRegion requires at least one segment")
        object.__setattr__(self, "segments", tuple(sorted(self.segments)))

    @property
    def sort_key(self) -> tuple[float, float]:
        return self.start, self.end

    @property
    def start(self) -> float:
        return self.segments[0].start

    @property
    def end(self) -> float:
        return self.segments[-1].end

    @property
    def duration(self) -> float:
        return sum(segment.duration for segment in self.segments)

    @property
    def bounds(self) -> Segment:
        return Segment(self.start, self.end)

    def intersection(self, other: Segment) -> "VadRegion | None":
        segments = tuple(
            overlap
            for segment in self.segments
            if (overlap := segment.intersection(other)) is not None
        )
        return VadRegion(segments) if segments else None

    def overlap_duration(self, other: Segment) -> float:
        return sum(segment.overlap_duration(other) for segment in self.segments)


@contextmanager
def capture_vad_segments(vad_model, *, enabled: bool):
    vad_segments: list[VadSegment] = []
    if not enabled:
        yield vad_segments
        return

    original_merge_chunks = vad_model.merge_chunks

    def capture(*args, **kwargs):
        chunks = original_merge_chunks(*args, **kwargs)
        vad_segments.extend(
            VadSegment(*segment)
            for chunk in chunks
            for segment in chunk.get("segments", [])
        )
        return chunks

    vad_model.merge_chunks = capture
    try:
        yield vad_segments
    finally:
        del vad_model.merge_chunks


@dataclass(frozen=True)
class VadRealigner:
    regions: tuple[VadRegion, ...]
    padding: float

    @classmethod
    def from_segments(
        cls,
        vad_segments: list[VadSegment],
        *,
        min_silence: float,
        padding: float,
    ) -> Self:
        regions: list[VadRegion] = []
        for segment in sorted(vad_segments):
            if regions and segment.start - regions[-1].end < min_silence:
                regions[-1] = VadRegion((*regions[-1].segments, segment))
            else:
                regions.append(VadRegion((segment,)))
        return cls(tuple(regions), padding)

    def _dominant_region(self, segment: Segment) -> int | None:
        overlaps = [
            (region.overlap_duration(segment), region_idx)
            for region_idx, region in enumerate(self.regions)
            if region.start < segment.end and region.end > segment.start
        ]
        best_overlap = max((overlap for overlap, _ in overlaps), default=0.0)
        winners = [
            region_idx
            for overlap, region_idx in overlaps
            if overlap == best_overlap and overlap > 0
        ]
        if len(winners) != 1:
            return None
        return winners[0]

    def _parts(self, segment: dict) -> list[dict] | None:
        words = segment.get("words", [])
        if not words:
            return None
        try:
            word_segments = [Segment(word["start"], word["end"]) for word in words]
        except KeyError:
            return None

        assignments = [
            self._dominant_region(word_segment) for word_segment in word_segments
        ]
        if any(region is None for region in assignments):
            return None
        if any(left > right for left, right in pairwise(assignments)):
            return None

        segment_bounds = Segment(segment["start"], segment["end"]).padded(self.padding)
        parts = []
        grouped_words = groupby(
            zip(assignments, words, strict=True),
            key=lambda item: item[0],
        )
        for region_idx, group in grouped_words:
            window = (
                self.regions[region_idx]
                .bounds.padded(self.padding)
                .intersection(segment_bounds)
            )
            if window is None:
                return None
            group_words = [word for _, word in group]

            part = {
                "start": round(window.start, 3),
                "end": round(window.end, 3),
                "text": " ".join(word["word"] for word in group_words),
            }
            if "avg_logprob" in segment:
                part["avg_logprob"] = segment["avg_logprob"]
            parts.append(part)

        clips_segment = any(
            part["start"] > segment["start"] or part["end"] < segment["end"]
            for part in parts
        )
        return parts if len(parts) > 1 or clips_segment else None

    def prepare(self, segments: list[dict]) -> tuple[list[dict], list[dict]]:
        """Separate untouched segments from inputs that need VAD-bounded realignment."""
        untouched = []
        realignment_inputs = []
        for segment in segments:
            parts = self._parts(segment)
            if parts is None:
                untouched.append(segment)
            else:
                realignment_inputs.extend(parts)
        return untouched, realignment_inputs
