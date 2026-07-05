import re
from datetime import date
from pathlib import Path

import numpy as np
import whisperx

from .constants import SR
from .transcript import Transcript


class Episode:
    """A single Državni Posao episode, parsed from its audio file path."""

    prefix = "DRŽAVNI POSAO [HQ] - "
    date_regex = re.compile(r"\((\d{2})\.(\d{2})\.(\d{4})\.\)")
    episode_regex = re.compile(r"Ep\.(\d+)")

    def __init__(
        self,
        full_name: str,
        date: date,
        audio_path: Path,
        key: str,
        transcript: Transcript | None = None,
        trim_start: float = 0.0,
        trim_end: float = 0.0,
    ) -> None:
        self.full_name = full_name
        self.date = date
        self.audio_path = audio_path
        self.key = key
        self.transcript = transcript  # filled/refined by the pipeline stages
        self.trim_start = trim_start  # seconds of intro to skip
        self.trim_end = trim_end  # seconds of outro to skip
        self._wav: np.ndarray | None = None  # lazily loaded, cached waveform

    @classmethod
    def from_path(
        cls, path: str | Path, trim_start: float = 0.0, trim_end: float = 0.0
    ) -> "Episode":
        """Build an Episode by parsing *path*'s filename."""
        path = Path(path)
        stem = path.stem

        m = cls.date_regex.search(stem)
        if not m:
            raise ValueError(f"No broadcast date found in filename: {path.name!r}")
        day, month, year = (int(g) for g in m.groups())

        ep = cls.episode_regex.search(stem)
        if ep:
            key = ep.group(1)
        else:  # special: title with prefix and date stripped, whitespace collapsed
            title = cls.date_regex.sub("", stem.removeprefix(cls.prefix))
            key = re.sub(r"\s{2,}", " ", title).strip()

        return cls(
            full_name=stem,
            date=date(year, month, day),
            audio_path=path,
            key=key,
            trim_start=trim_start,
            trim_end=trim_end,
        )

    def audio(self) -> np.ndarray:
        """
        The episode's waveform (intro/outro trimmed), loaded and cached on first access.

        The full waveform is cached; trim_start/trim_end are applied on the way out,
        so every pipeline stage sees the same frame of reference.
        """
        if self._wav is None:
            self._wav = whisperx.load_audio(str(self.audio_path))
        start = int(self.trim_start * SR)
        end = -int(self.trim_end * SR) or None  # 0 → None (full length)
        return self._wav[start:end]

    def clear_audio(self) -> None:
        """Release the cached waveform."""
        self._wav = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Episode):
            return NotImplemented
        return (self.full_name, self.date, self.audio_path, self.key) == (
            other.full_name,
            other.date,
            other.audio_path,
            other.key,
        )

    def __lt__(self, other: "Episode") -> bool:
        if not isinstance(other, Episode):
            return NotImplemented
        return self.date < other.date

    def __repr__(self) -> str:
        return f"Episode(key={self.key!r}, date={self.date.isoformat()}, full_name={self.full_name!r})"
