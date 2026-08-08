import gc
import logging

import torch
import whisperx
from pydantic import BaseModel, ConfigDict, Field

from .constants import transliterate_serbian
from .episode import Episode
from .transcript import Transcript

logger = logging.getLogger(__name__)

DEFAULT_SERBIAN_ALIGN_MODEL = "classla/wav2vec2-xls-r-parlaspeech-hr"


class TranscriberConfig(BaseModel):
    """WhisperX ASR + alignment hyperparameters."""

    model_config = ConfigDict(protected_namespaces=())  # allow the model_name field

    model_name: str
    device: str = "cuda"
    compute_type: str = "float16"
    language: str | None = None
    asr_options: dict = Field(default_factory=dict)
    align_model_name: str | None = None
    batch_size: int = 16


class Transcriber:
    """WhisperX ASR + forced alignment. Loads its own models; fills episode.transcript."""

    def __init__(self, config: TranscriberConfig) -> None:
        self.config = config
        c = config
        self.asr = whisperx.load_model(
            c.model_name,
            c.device,
            compute_type=c.compute_type,
            language=c.language,
            asr_options=c.asr_options,
        )
        logger.info("Loaded WhisperX '%s' on %s", c.model_name, c.device)
        align_model_name = c.align_model_name
        if align_model_name is None and c.language == "sr":
            align_model_name = DEFAULT_SERBIAN_ALIGN_MODEL
        self.align_model, self.align_meta = whisperx.load_align_model(
            language_code=c.language,
            device=c.device,
            model_name=align_model_name,
        )
        # WhisperX converts spaces to "|" before forced alignment. Classla's
        # tokenizer uses a literal space token, so expose it under both names.
        dictionary = self.align_meta.get("dictionary", {})
        if "|" not in dictionary and " " in dictionary:
            dictionary["|"] = dictionary[" "]

    def run(self, episode: Episode) -> Episode:
        """Transcribe + align *episode*'s audio; set episode.transcript and return it."""
        logger.info("Transcribing episode %r", episode.key)
        audio = episode.audio()
        result = self.asr.transcribe(audio, batch_size=self.config.batch_size)
        for segment in result["segments"]:
            segment["text"] = transliterate_serbian(segment["text"])
        if self.align_model is not None:
            result = whisperx.align(
                result["segments"],
                self.align_model,
                self.align_meta,
                audio,
                self.config.device,
                return_char_alignments=False,
            )
        for segment in result["segments"]:
            for word in segment.get("words", []):
                word["word"] = transliterate_serbian(word["word"])
        episode.transcript = Transcript(result["segments"], file=episode.full_name)
        logger.info(
            "Transcribed episode %r: %d segments",
            episode.key,
            len(episode.transcript.segments),
        )
        return episode

    def free(self) -> None:
        """Release the models and clear the GPU cache."""
        logger.debug("Freeing Transcriber models")
        self.asr = self.align_model = self.align_meta = None
        gc.collect()
        torch.cuda.empty_cache()
