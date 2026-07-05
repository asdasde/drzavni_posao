import gc

import torch
import whisperx
from pydantic import BaseModel, ConfigDict, Field

from .episode import Episode
from .transcript import Transcript


class TranscriberConfig(BaseModel):
    """WhisperX ASR + alignment hyperparameters."""
    model_config = ConfigDict(protected_namespaces=())  # allow the model_name field

    model_name:       str
    device:           str = "cuda"
    compute_type:     str = "float16"
    language:         str | None = None
    asr_options:      dict = Field(default_factory=dict)
    align_model_name: str | None = None
    batch_size:       int = 16


class Transcriber:
    """WhisperX ASR + forced alignment. Loads its own models; fills episode.transcript."""

    def __init__(self, config: TranscriberConfig) -> None:
        self.config = config
        c = config
        self.asr = whisperx.load_model(c.model_name, c.device, compute_type=c.compute_type,
                                       language=c.language, asr_options=c.asr_options)
        print(f"Loaded WhisperX '{c.model_name}' on {c.device}")
        self.align_model, self.align_meta = whisperx.load_align_model(
            language_code=c.language, device=c.device, model_name=c.align_model_name)

    def run(self, episode: Episode) -> Episode:
        """Transcribe + align *episode*'s audio; set episode.transcript and return it."""
        audio  = episode.audio()
        result = self.asr.transcribe(audio, batch_size=self.config.batch_size)
        if self.align_model is not None:
            result = whisperx.align(result["segments"], self.align_model, self.align_meta,
                                    audio, self.config.device, return_char_alignments=False)
        episode.transcript = Transcript(result["segments"], file=episode.full_name)
        return episode

    def free(self) -> None:
        """Release the models and clear the GPU cache."""
        self.asr = self.align_model = self.align_meta = None
        gc.collect()
        torch.cuda.empty_cache()
