import json
import logging
import unicodedata
from pathlib import Path
from typing import Iterator

from .episode import Episode, EpisodeCatalog
from .transcript import Transcript

logger = logging.getLogger(__name__)

DEFAULT_CATALOG_PATH = Path("data/youtube_episodes.json")


class MediaStore:
    """Indexes episodes on disk and reads/writes their transcripts. Audio is Episode's job."""

    def __init__(
        self,
        audio_dir: Path | None,
        run_dir: Path,
        trim_start: float = 0.0,
        trim_end: float = 0.0,
        catalog_path: Path = DEFAULT_CATALOG_PATH,
    ) -> None:
        self.audio_dir = audio_dir
        self.run_dir = run_dir
        self.trim_start = trim_start  # applied to every episode's audio (intro)
        self.trim_end = trim_end  # applied to every episode's audio (outro)
        self.catalog = EpisodeCatalog.model_validate_json(
            catalog_path.read_text(encoding="utf-8")
        )
        self._index: dict[str, Episode] = self._build_index()

    def _build_index(self) -> dict[str, Episode]:
        paths = (
            sorted(self.audio_dir.glob("*.mp3"))
            if self.audio_dir is not None
            else (Path(f"{item.youtube_title}.mp3") for item in self.catalog.episodes)
        )
        episodes = [
            Episode(
                path,
                self.catalog.metadata_for(path),
                trim_start=self.trim_start,
                trim_end=self.trim_end,
            )
            for path in paths
        ]
        index = {episode.key: episode for episode in episodes}
        logger.info("Indexed %d episodes under %s", len(index), self.audio_dir)
        return index

    def episode(self, key: str) -> Episode:
        """The Episode for *key*."""
        try:
            return self._index[unicodedata.normalize("NFKC", key)]
        except KeyError:
            raise KeyError(f"Episode {key!r} not found in {self.audio_dir}") from None

    def audio_path(self, key: str) -> Path:
        """Full path of the audio file for episode *key*."""
        return self.episode(key).audio_path

    def episodes(self) -> Iterator[Episode]:
        """Iterate over episodes ordered by broadcast date."""
        yield from sorted(self._index.values())

    # ── Transcripts ───────────────────────────────────────────────────────

    def _out_dir(self, subdir: Path = Path()) -> Path:
        """Resolve the output directory for *subdir* (defaults to run_dir)."""
        return self.run_dir / subdir

    def transcript_path(self, key: str, subdir: Path = Path()) -> Path:
        """<run_dir>/[subdir/]<key>.json"""
        return self._out_dir(subdir) / f"{key}.json"

    def exists(self, key: str, subdir: Path = Path()) -> bool:
        """True if a transcript already exists for episode *key*."""
        return self.transcript_path(key, subdir).exists()

    def save(self, episode: Episode, subdir: Path = Path()) -> None:
        """
        Persist *episode*'s transcript to <run_dir>/[subdir/]<key>.{txt,json}.

        The output directory is created if it does not exist.
        """
        key = episode.key
        if episode.transcript is None:
            raise ValueError(f"Episode {key!r} has no transcript to save")
        out_dir = self._out_dir(subdir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{key}.txt").write_text(
            episode.transcript.render(), encoding="utf-8"
        )
        with (out_dir / f"{key}.json").open("w", encoding="utf-8") as fh:
            json.dump(
                episode.transcript.to_dict(),
                fh,
                ensure_ascii=False,
                indent=2,
                default=lambda o: o.item() if hasattr(o, "item") else str(o),
            )
        logger.info("Saved transcript for episode %r to %s", key, out_dir)

    def load(self, key: str, subdir: Path = Path()) -> Transcript:
        """Read a saved transcript for episode *key*."""
        path = self.transcript_path(key, subdir)
        return Transcript.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def load_all(self, subdir: Path = Path()) -> list[Episode]:
        """
        Load saved transcripts into their episodes, sorted by broadcast date.

        Returns an empty list if the directory does not exist.
        """
        out_dir = self._out_dir(subdir)
        if not out_dir.exists():
            return []
        episodes = []
        for path in out_dir.glob("*.json"):
            episode = self.episode(path.stem)
            transcript = self.load(path.stem, subdir)
            episode.transcript = transcript
            episodes.append(episode)
        return sorted(episodes)
