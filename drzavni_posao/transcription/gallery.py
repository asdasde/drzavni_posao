import json
import logging
from pathlib import Path

import numpy as np

from .embedder import Embedder
from .store import MediaStore

logger = logging.getLogger(__name__)


def load_speaker_samples(path: Path) -> dict[str, list[dict]]:
    return json.loads(path.read_text(encoding="utf-8"))


class Gallery:
    """Enrolled reference voiceprints in two forms.

    Holds the raw per-character embeddings and the stacked, L2-normalized
    matrix used for cosine matching. A character may enroll several clips, so
    the matrix has one row per reference *vector* (not per character); each row
    carries its character's name in `reference_names`. Pure data: it knows
    nothing about queries, cosine, or thresholds — matching lives in the caller.
    """

    def __init__(self, references: dict[str, np.ndarray]) -> None:
        if not references:
            raise ValueError("Gallery requires at least one enrolled character")
        self.references = references  # {name: (n_clips, D)} raw reference vectors
        self.names = list(references)  # distinct character names
        self.reference_matrix, self.reference_names = self._build_matrix(references)

    @classmethod
    def from_clips(
        cls,
        samples: dict[str, list[dict]],
        embedder: Embedder,
        store: MediaStore,
    ) -> "Gallery":
        """Build a Gallery by embedding enrolled reference clips.

        *samples* maps each character name to its reference clips, whose
        start/end are in the **untrimmed** frame of the source episode's audio::
        
        Each clip is located via *store*, its times converted into the trimmed
        frame the episode's waveform actually uses (``ts - trim_start``), and
        embedded with *embedder*.
        """
        
        references: dict[str, np.ndarray] = {}
        for name, clips in samples.items():
            vectors = []
            for clip in clips:
                episode = store.episode(clip["episode_key"])
                start = clip["start"] - episode.trim_start
                end = clip["end"] - episode.trim_start
                vector = embedder.embed(episode.audio(), start, end)
                if vector is None:
                    logger.warning(
                        "%s clip ep %s %.1f-%.1fs too short to embed; skipped",
                        name,
                        clip["episode_key"],
                        clip["start"],
                        clip["end"],
                    )
                    continue
                vectors.append(vector)
            if not vectors:
                raise ValueError(f"No usable reference clips for character {name!r}")
            references[name] = np.vstack(vectors)
            logger.info("Enrolled %s: %d reference clip(s)", name, len(vectors))
        return cls(references)

    @staticmethod
    def _build_matrix(
        references: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Stack every clip vector into one L2-normalized matrix, tagging each
        row with its character's name."""
        rows = [np.atleast_2d(vectors) for vectors in references.values()]
        matrix = np.vstack(rows)
        matrix = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
        reference_names = np.array(
            [name for name, block in zip(references, rows) for _ in range(len(block))]
        )
        return matrix, reference_names

    def __repr__(self) -> str:
        return (
            f"Gallery(characters={len(self.names)}, refs={self.reference_matrix.shape})"
        )
