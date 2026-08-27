class Transcript:
    """A single episode's transcript: the native WhisperX segment dicts, evolving in place.

    The same segments flow through the pipeline, gaining keys per stage:
      Transcriber       → start/end/text/words
      Diarizer          → + speaker  (SPEAKER_00 …)
      CharacterAssigner → speaker overwritten with the character name
    """

    def __init__(
        self,
        segments: list[dict],
        file: str | None = None,
    ) -> None:
        self.segments = segments  # native WhisperX dicts, mutated in place by stages
        self.file = file  # source episode name, for reload/sort

    def render(self) -> str:
        """Speaker/character-labeled lines (plain text where no speaker is set)."""
        lines = []
        for seg in self.segments:
            spk = seg.get("speaker")
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            lines.append(f"[{spk}] {text}" if spk else text)
        return "\n".join(lines)

    def collapse_speaker_turns(self) -> "Transcript":
        """Return a new transcript with adjacent same-speaker segments merged."""
        turns = []
        for segment in self.segments:
            text = (segment.get("text") or "").strip()
            if not text:
                continue

            speaker = segment.get("speaker")
            if turns and turns[-1]["speaker"] == speaker:
                turns[-1]["text"] = f"{turns[-1]['text']} {text}"
                turns[-1]["end"] = segment.get("end", turns[-1]["end"])
                continue

            turns.append(
                {
                    "start": segment.get("start"),
                    "end": segment.get("end"),
                    "speaker": speaker,
                    "text": text,
                }
            )
        return Transcript(turns, file=self.file)

    def to_dict(self) -> dict:
        return {"file": self.file, "segments": self.segments}

    @classmethod
    def from_dict(cls, d: dict) -> "Transcript":
        return cls(segments=d.get("segments", []), file=d.get("file"))

    def __repr__(self) -> str:
        return f"Transcript(file={self.file!r}, segments={len(self.segments)})"
