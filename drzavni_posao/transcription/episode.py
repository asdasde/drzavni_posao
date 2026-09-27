import logging
import unicodedata
from datetime import date as Date
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr

from .constants import SR
from .transcript import Transcript

logger = logging.getLogger(__name__)


def episode_order_key(number: int | None, tiebreak: str) -> tuple[bool, int, str]:
    """Sort key: numbered episodes first in order, then the rest by tiebreak."""
    return (number is None, number if number is not None else 0, tiebreak)


class EpisodeMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    episode_number: int | None
    title: str
    date: Date
    is_special: bool
    original_url: str
    youtube_video_id: str
    youtube_title: str
    duration_seconds: int
    view_count: int
    season: int
    playlist_id: str
    playlist_index: int | None


class EpisodeCatalog(BaseModel):
    episodes: list[EpisodeMetadata]
    _by_audio_name: dict[str, EpisodeMetadata] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: object) -> None:
        for metadata in self.episodes:
            audio_name = self._normalized_audio_name(metadata.youtube_title)
            if audio_name in self._by_audio_name:
                raise ValueError(f"Duplicate audio filename: {audio_name}")
            self._by_audio_name[audio_name] = metadata

    @staticmethod
    def _normalized_audio_name(name: str) -> str:
        if name.casefold().endswith(".mp3"):
            name = name[:-4]
        return unicodedata.normalize("NFKC", f"{name}.mp3")

    def metadata_for(self, source: Path) -> EpisodeMetadata:
        audio_name = self._normalized_audio_name(source.name)
        try:
            return self._by_audio_name[audio_name]
        except KeyError:
            raise ValueError(f"{audio_name!r} not found in episode catalog") from None


class Episode:
    """A single Državni Posao episode with its audio and catalog metadata."""

    def __init__(
        self,
        audio_path: Path,
        metadata: EpisodeMetadata,
        transcript: Transcript | None = None,
        trim_start: float = 0.0,
        trim_end: float = 0.0,
    ) -> None:
        self.audio_path = audio_path
        self.metadata = metadata
        self.full_name = self.audio_path.stem
        self.transcript = transcript  # filled/refined by the pipeline stages
        self.trim_start = trim_start  # seconds of intro to skip
        self.trim_end = trim_end  # seconds of outro to skip
        self.diarization: list[dict] | None = None
        self._wav: np.ndarray | None = None  # lazily loaded, cached waveform

    @property
    def key(self) -> str:
        """Stable storage key: episode number, or title for a special."""
        if self.metadata.episode_number is not None:
            return str(self.metadata.episode_number)
        return unicodedata.normalize(
            "NFKC", f"{self.metadata.title} ({self.metadata.date.isoformat()})"
        )

    def audio(self) -> np.ndarray:
        """
        The episode's waveform (intro/outro trimmed), loaded and cached on first access.

        The full waveform is cached; trim_start/trim_end are applied on the way out,
        so every pipeline stage sees the same frame of reference.
        """
        if self._wav is None:
            import whisperx

            logger.debug(
                "Loading audio for episode %r from %s",
                self.metadata.episode_number,
                self.audio_path,
            )
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
        return (self.audio_path, self.metadata) == (other.audio_path, other.metadata)

    def __lt__(self, other: "Episode") -> bool:
        if not isinstance(other, Episode):
            return NotImplemented
        return self.metadata.date < other.metadata.date

    def __repr__(self) -> str:
        return (
            f"Episode(number={self.metadata.episode_number!r}, "
            f"date={self.metadata.date.isoformat()}, full_name={self.full_name!r})"
        )
