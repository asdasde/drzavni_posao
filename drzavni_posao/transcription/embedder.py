import gc
import logging

import numpy as np
import torch
from pyannote.audio import Inference, Model
from pydantic import BaseModel

from .constants import SR

logger = logging.getLogger(__name__)


class EmbedderConfig(BaseModel):
    """Pyannote speaker-embedding model hyperparameters."""

    embed_model: str = "pyannote/wespeaker-voxceleb-resnet34-LM"
    hf_token: str | None = None
    device: str = "cuda"
    sample_rate: int = SR
    min_segment_seconds: float = 0.4


class Embedder:
    """Turns a (waveform, start, end) span into a speaker-embedding vector.

    Episode-agnostic: it knows nothing about episodes, transcripts, or trims.
    Times are interpreted in the frame of whatever waveform the caller passes,
    so the caller decides trimmed vs. untrimmed audio.
    """

    def __init__(self, config: EmbedderConfig) -> None:
        self.config = config
        model = Model.from_pretrained(
            config.embed_model, use_auth_token=config.hf_token
        )
        self.model = Inference(
            model, window="whole", device=torch.device(config.device)
        )
        logger.info(
            "Loaded embedding model %r on %s", config.embed_model, config.device
        )

    def embed(
        self, waveform: np.ndarray, start: float, end: float
    ) -> np.ndarray | None:
        """Embed the [start, end]-second segment of *waveform*.

        Returns a flat embedding vector, or None if the span is shorter than
        `min_segment_seconds` (too short to embed reliably).
        """
        segment = waveform[
            int(start * self.config.sample_rate) : int(end * self.config.sample_rate)
        ]
        if len(segment) < int(
            self.config.min_segment_seconds * self.config.sample_rate
        ):
            return None
        tensor = torch.from_numpy(segment).float().unsqueeze(0)  # (channel=1, time)
        vector = self.model(
            {"waveform": tensor, "sample_rate": self.config.sample_rate}
        )
        return np.asarray(vector).reshape(-1)

    def free(self) -> None:
        """Release the model and clear the GPU cache."""
        logger.debug("Freeing Embedder model")
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()
