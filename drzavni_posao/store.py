import json
import logging
from pathlib import Path
from typing import Iterator

from .episode import Episode
from .transcript import Transcript

logger = logging.getLogger(__name__)


class MediaStore:
    """Indexes episodes on disk and reads/writes their transcripts. Audio is Episode's job."""

    def __init__(
        self,
        audio_dir: str | Path,
        run_dir: str | Path,
        trim_start: float = 0.0,
        trim_end: float = 0.0,
    ) -> None:
        self.audio_dir = Path(audio_dir)
        self.run_dir = Path(run_dir)
        self.trim_start = trim_start  # applied to every episode's audio (intro)
        self.trim_end = trim_end  # applied to every episode's audio (outro)
        self._index: dict[str, Episode] = self._build_index()

    def _build_index(self) -> dict[str, Episode]:
        index = {
            ep.key: ep
            for p in sorted(self.audio_dir.rglob("*.mp3"))
            if (ep := Episode.from_path(p, self.trim_start, self.trim_end))
        }
        logger.info("Indexed %d episodes under %s", len(index), self.audio_dir)
        return index

    def episode(self, key: str) -> Episode:
        """The Episode for *key*."""
        try:
            return self._index[key]
        except KeyError:
            raise KeyError(f"Episode {key!r} not found in {self.audio_dir}") from None

    def audio_path(self, key: str) -> Path:
        """Full path of the audio file for episode *key*."""
        return self.episode(key).audio_path

    def episodes(self) -> Iterator[Episode]:
        """Iterate over episodes ordered by broadcast date."""
        yield from sorted(self._index.values())

    # ── Transcripts ───────────────────────────────────────────────────────

    def _out_dir(self, subdir: str = "") -> Path:
        """Resolve the output directory for *subdir* (defaults to run_dir)."""
        return self.run_dir / subdir if subdir else self.run_dir

    def transcript_path(self, key: str, subdir: str = "") -> Path:
        """<run_dir>/[subdir/]<key>.json"""
        return self._out_dir(subdir) / f"{key}.json"

    def exists(self, key: str, subdir: str = "") -> bool:
        """True if a transcript already exists for episode *key*."""
        return self.transcript_path(key, subdir).exists()

    def save(self, episode: Episode, subdir: str = "") -> None:
        """
        Persist *episode*'s transcript to <run_dir>/[subdir/]<key>.{txt,json}.

        The output directory is created if it does not exist.
        """
        if episode.transcript is None:
            raise ValueError(f"Episode {episode.key!r} has no transcript to save")
        out_dir = self._out_dir(subdir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{episode.key}.txt").write_text(
            episode.transcript.render(), encoding="utf-8"
        )
        with (out_dir / f"{episode.key}.json").open("w", encoding="utf-8") as fh:
            json.dump(
                episode.transcript.to_dict(),
                fh,
                ensure_ascii=False,
                default=lambda o: o.item() if hasattr(o, "item") else str(o),
            )
        logger.info("Saved transcript for episode %r to %s", episode.key, out_dir)

    def load(self, key: str, subdir: str = "") -> Transcript:
        """Read a saved transcript for episode *key*."""
        path = self.transcript_path(key, subdir)
        return Transcript.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def load_all(self, subdir: str = "") -> list[Transcript]:
        """
        Load all saved transcripts sorted by broadcast date.

        Returns an empty list if the directory does not exist.
        """
        out_dir = self._out_dir(subdir)
        if not out_dir.exists():
            return []
        transcripts = [self.load(p.stem, subdir) for p in out_dir.glob("*.json")]
        return sorted(transcripts, key=lambda t: Episode.from_path(t.file).date)
