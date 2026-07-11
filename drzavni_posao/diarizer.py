import gc
import logging

import torch
import whisperx
from pydantic import BaseModel
from whisperx.diarize import DiarizationPipeline

from .episode import Episode

logger = logging.getLogger(__name__)


class DiarizerConfig(BaseModel):
    """Pyannote speaker-diarization-3.1 hyperparameters."""

    diarize_model: str = "pyannote/speaker-diarization-3.1"
    hf_token: str | None = None
    device: str = "cuda"
    min_speakers: int | None = None
    max_speakers: int | None = None

    # pipeline.instantiate() knobs
    clustering_method: str = "centroid"  # linkage rule for merging speaker embeddings
    clustering_threshold: float = 0.7  # distance cutoff → main speaker-count dial
    min_cluster_size: int = 12  # drop speakers with fewer embeddings than this
    min_duration_off: float = 0.0  # shortest silence allowed to break a turn


class Diarizer:
    """Pyannote speaker diarization. Loads its own model; adds `speaker` to each segment."""

    def __init__(self, config: DiarizerConfig) -> None:
        self.config = config
        c = config
        self.diarize = DiarizationPipeline(
            model_name=c.diarize_model, token=c.hf_token, device=c.device
        )
        self.diarize.model.instantiate(
            {
                "clustering": {
                    "method": c.clustering_method,
                    "min_cluster_size": c.min_cluster_size,
                    "threshold": c.clustering_threshold,
                },
                "segmentation": {"min_duration_off": c.min_duration_off},
            }
        )
        logger.info("Loaded diarization pipeline %r on %s", c.diarize_model, c.device)

    def run(self, episode: Episode) -> Episode:
        """Diarize *episode*'s audio and tag its segments with `speaker`; return the episode."""
        if episode.transcript is None:
            raise ValueError(
                f"Episode {episode.key!r} has no transcript; run Transcriber first"
            )
        logger.info("Diarizing episode %r", episode.key)
        audio = episode.audio()
        diarize_segments = self.diarize(
            audio,
            min_speakers=self.config.min_speakers,
            max_speakers=self.config.max_speakers,
        )
        # Retain the raw pyannote turns (as plain dicts) so callers can inspect the
        # actual speaker-change boundaries, independent of the word-level labels.
        episode.diarization = [
            {"start": float(t.start), "end": float(t.end), "speaker": t.speaker}
            for t in diarize_segments.itertuples()
        ]
        result = whisperx.assign_word_speakers(
            diarize_segments, {"segments": episode.transcript.segments}
        )
        episode.transcript.segments = result["segments"]
        logger.info(
            "Diarized episode %r: %d turns, %d speakers",
            episode.key,
            len(episode.diarization),
            len({t["speaker"] for t in episode.diarization}),
        )
        return episode

    def free(self) -> None:
        """Release the model and clear the GPU cache."""
        logger.debug("Freeing Diarizer model")
        self.diarize = None
        gc.collect()
        torch.cuda.empty_cache()
