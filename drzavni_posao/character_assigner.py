import gc
import logging

import numpy as np
import torch
from pydantic import BaseModel

from .embedder import Embedder
from .episode import Episode
from .gallery import Gallery

logger = logging.getLogger(__name__)


class CharacterAssignerConfig(BaseModel):
    """Voice-matching hyperparameters."""

    min_similarity: float  # cosine cutoff; a best match below this → anonymous
    anonymous_label: str = "[anonymous]"


class CharacterAssigner:
    """Assigns character names to transcript segments by voice.

    Embeds each segment and matches it against an enrolled `Gallery`. Runs in
    one of two modes, chosen by whether the Diarizer has already run:

      no diarization  → each segment matched independently (method 3).
      diarization set → segments are pooled per pyannote speaker into one
                        averaged voiceprint, matched once, and the whole
                        cluster is labeled; segments with no speaker stay
                        anonymous.

    The matched name overwrites each segment's `speaker` field.
    """

    def __init__(
        self, config: CharacterAssignerConfig, embedder: Embedder, gallery: Gallery
    ) -> None:
        self.config = config
        self.embedder = embedder
        self.gallery = gallery
        logger.info(
            "CharacterAssigner ready: %d characters, min_similarity=%.2f",
            len(gallery.names),
            config.min_similarity,
        )

    def run(self, episode: Episode) -> Episode:
        """Label *episode*'s segments with character names; return the episode."""
        if episode.transcript is None:
            raise ValueError(
                f"Episode {episode.key!r} has no transcript; run Transcriber first"
            )
        segments = episode.transcript.segments
        audio = episode.audio()

        if episode.diarization is None:
            logger.info("Assigning characters in %r (per-segment)", episode.key)
            self._assign_per_segment(segments, audio)
        else:
            logger.info("Assigning characters in %r (per-cluster)", episode.key)
            self._assign_per_cluster(segments, audio)
        return episode

    def _match(self, queries: np.ndarray) -> list[tuple[str, float]]:
        """Match L2-normalized query vectors (n, D) against the gallery.

        Returns one (best_name, best_similarity) per query row, taking the max
        cosine over each character's reference clips.
        """
        similarities = queries @ self.gallery.reference_matrix.T  # (n, n_refs)
        per_character = np.column_stack(
            [
                similarities[:, self.gallery.reference_names == name].max(axis=1)
                for name in self.gallery.names
            ]
        )  # (n, n_characters)
        best = per_character.argmax(axis=1)
        return [
            (self.gallery.names[b], float(per_character[i, b]))
            for i, b in enumerate(best)
        ]

    def _label(self, similarity: float, name: str) -> str:
        """The character name if confident enough, else the anonymous label."""
        return (
            name
            if similarity >= self.config.min_similarity
            else self.config.anonymous_label
        )

    def _embed_segment(self, segment: dict, audio: np.ndarray) -> np.ndarray | None:
        """L2-normalized embedding for a segment, or None if it can't be embedded."""
        if "start" not in segment:
            return None
        vector = self.embedder.embed(audio, segment["start"], segment["end"])
        if vector is None:
            return None
        return vector / (np.linalg.norm(vector) + 1e-9)

    def _assign_per_segment(self, segments: list[dict], audio: np.ndarray) -> None:
        """Match every embeddable segment on its own; unembeddable → anonymous."""
        embeddable, queries = [], []
        for segment in segments:
            vector = self._embed_segment(segment, audio)
            if vector is None:
                segment["speaker"] = self.config.anonymous_label
                continue
            embeddable.append(segment)
            queries.append(vector)

        if not embeddable:
            return
        for segment, (name, similarity) in zip(
            embeddable, self._match(np.vstack(queries))
        ):
            segment["speaker"] = self._label(similarity, name)

    def _assign_per_cluster(self, segments: list[dict], audio: np.ndarray) -> None:
        """Pool each pyannote speaker's segments into one voiceprint, match once,
        and label the whole cluster. Segments with no speaker stay anonymous."""
        clusters: dict[str, list[np.ndarray]] = {}
        for segment in segments:
            speaker = segment.get("speaker")
            if speaker is None:
                segment["speaker"] = self.config.anonymous_label
                continue
            vector = self._embed_segment(segment, audio)
            if vector is not None:
                clusters.setdefault(speaker, []).append(vector)

        if not clusters:
            return
        speakers = list(clusters)
        voiceprints = np.vstack([np.mean(clusters[s], axis=0) for s in speakers])
        voiceprints /= np.linalg.norm(voiceprints, axis=1, keepdims=True) + 1e-9
        cluster_label = {
            speaker: self._label(similarity, name)
            for speaker, (name, similarity) in zip(speakers, self._match(voiceprints))
        }

        for segment in segments:
            speaker = segment.get("speaker")
            if speaker in cluster_label:
                segment["speaker"] = cluster_label[speaker]

    def free(self) -> None:
        """Release the embedder and clear the GPU cache."""
        logger.debug("Freeing CharacterAssigner")
        self.embedder.free()
        self.embedder = None
        gc.collect()
        torch.cuda.empty_cache()
