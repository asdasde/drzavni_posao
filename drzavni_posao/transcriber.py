import gc
import logging
import re
from copy import deepcopy

import numpy as np
import torch
import whisperx
from pydantic import BaseModel, ConfigDict, Field

from .constants import SR, transliterate_serbian
from .episode import Episode
from .transcript import Transcript
from .vad_realigner import VadRealigner, capture_vad_segments

logger = logging.getLogger(__name__)

DEFAULT_SERBIAN_ALIGN_MODEL = "classla/wav2vec2-xls-r-parlaspeech-hr"
ELLIPSIS_RE = re.compile(r"\.{3,}")


class TranscriberConfig(BaseModel):
    """WhisperX ASR + alignment hyperparameters."""

    model_config = ConfigDict(protected_namespaces=())  # allow the model_name field

    model_name: str
    device: str = "cuda"
    compute_type: str = "float16"
    language: str | None = None
    asr_options: dict = Field(default_factory=dict)
    vad_options: dict = Field(default_factory=dict)
    align_model_name: str | None = None
    batch_size: int = 16
    chunk_size: int = 30
    retry_collapsed: bool = True
    retry_collapsed_min_duration: float = 8.0
    retry_collapsed_max_chars_per_sec: float = 1.5
    retry_chunk_size: int = 10
    vad_repair_min_silence: float | None = Field(default=None, gt=0)
    vad_repair_padding: float = Field(default=0.1, ge=0)
    split_on_ellipsis: bool = False


def is_collapsed(
    segment: dict,
    *,
    min_duration: float,
    max_chars_per_sec: float,
) -> bool:
    duration = segment["end"] - segment["start"]
    if duration < min_duration:
        return False
    text = segment.get("text", "").strip()
    return (len(text) / duration) < max_chars_per_sec


def split_ellipsis_segments(segments: list[dict]) -> list[dict]:
    """Split aligned segments where an ellipsis ends one timed word."""
    result = []
    for segment in segments:
        text = segment.get("text", "")
        words = segment.get("words", [])
        matches = list(ELLIPSIS_RE.finditer(text))
        word_matches = [
            (index, match)
            for index, word in enumerate(words)
            for match in ELLIPSIS_RE.finditer(word.get("word", "").strip())
        ]
        fragments = []
        text_start = word_start = 0
        start = segment["start"]
        for index, (match, (word_index, word_match)) in enumerate(
            zip(matches, word_matches)
        ):
            left = text[matches[index - 1].end() if index else 0 : match.start()]
            right = text[
                match.end() : matches[index + 1].start()
                if index + 1 < len(matches)
                else len(text)
            ]
            word = words[word_index]
            next_word = words[word_index + 1] if word_index + 1 < len(words) else None
            if (
                not left.strip(" .")
                or not right.strip(" .")
                or word_match.end() != len(word.get("word", "").strip())
                or word.get("end") is None
                or next_word is None
                or next_word.get("start") is None
            ):
                continue
            split_at = (word["end"] + next_word["start"]) / 2
            if not start < split_at < segment["end"]:
                continue

            fragment = deepcopy(segment)
            fragment.update(
                text=text[text_start : match.start()].strip(),
                start=start,
                end=split_at,
            )
            fragment["words"] = [
                word
                for word in fragment.get("words", [])[word_start : word_index + 1]
                if word.get("word", "").strip(" .")
            ]
            fragments.append(fragment)
            text_start = match.end()
            word_start = word_index + 1
            start = split_at

        if fragments:
            fragment = deepcopy(segment)
            fragment.update(
                text=text[text_start:].strip(),
                start=start,
                end=segment["end"],
            )
            fragment["words"] = [
                word
                for word in fragment.get("words", [])[word_start:]
                if word.get("word", "").strip(" .")
            ]
            fragments.append(fragment)
        result.extend(fragments if len(fragments) > 1 else [segment])
    return result


class Transcriber:
    """WhisperX ASR + forced alignment. Loads its own models; fills episode.transcript."""

    def __init__(self, config: TranscriberConfig) -> None:
        self.config = config
        c = config
        load_kwargs = dict(
            compute_type=c.compute_type,
            language=c.language,
            asr_options=c.asr_options,
        )
        if c.vad_options:
            load_kwargs["vad_options"] = c.vad_options
        self.asr = whisperx.load_model(
            c.model_name,
            c.device,
            **load_kwargs,
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
        logger.info("Transcribing episode %r", episode.metadata.episode_number)
        audio = episode.audio()
        with capture_vad_segments(
            self.asr.vad_model,
            enabled=self.config.vad_repair_min_silence is not None,
        ) as vad_segments:
            result = self.asr.transcribe(
                audio,
                batch_size=self.config.batch_size,
                chunk_size=self.config.chunk_size,
            )
        for segment in result["segments"]:
            segment["text"] = transliterate_serbian(segment["text"])
        result["segments"] = self._retry_collapsed(audio, result["segments"])
        if self.align_model is not None:
            result = whisperx.align(
                result["segments"],
                self.align_model,
                self.align_meta,
                audio,
                self.config.device,
                return_char_alignments=False,
            )
            if self.config.vad_repair_min_silence is not None:
                realigner = VadRealigner.from_segments(
                    vad_segments,
                    min_silence=self.config.vad_repair_min_silence,
                    padding=self.config.vad_repair_padding,
                )
                untouched, realignment_inputs = realigner.prepare(result["segments"])
                if realignment_inputs:
                    realigned = whisperx.align(
                        realignment_inputs,
                        self.align_model,
                        self.align_meta,
                        audio,
                        self.config.device,
                        return_char_alignments=False,
                    )
                    result["segments"] = sorted(
                        [*untouched, *realigned["segments"]],
                        key=lambda segment: segment["start"],
                    )
        if self.config.split_on_ellipsis:
            result["segments"] = split_ellipsis_segments(result["segments"])
        for segment in result["segments"]:
            for word in segment.get("words", []):
                word["word"] = transliterate_serbian(word["word"])
        episode.transcript = Transcript(
            result["segments"],
            file=episode.metadata.youtube_title,
        )
        logger.info(
            "Transcribed episode %r: %d segments",
            episode.metadata.episode_number,
            len(episode.transcript.segments),
        )
        return episode

    def _retry_collapsed(self, audio: np.ndarray, segments: list[dict]) -> list[dict]:
        """Re-decode long chunks that collapsed to almost no text.

        Full-episode ASR keeps the default 30s merge. Only flagged spans are
        cut out and transcribed again with ``retry_chunk_size``, then spliced
        back before alignment.
        """
        config = self.config
        if not config.retry_collapsed:
            return segments

        repaired: list[dict] = []
        retries = 0
        for segment in segments:
            if not is_collapsed(
                segment,
                min_duration=config.retry_collapsed_min_duration,
                max_chars_per_sec=config.retry_collapsed_max_chars_per_sec,
            ):
                repaired.append(segment)
                continue

            start, end = segment["start"], segment["end"]
            clip = audio[int(start * SR) : int(end * SR)]
            retry = self.asr.transcribe(
                clip,
                batch_size=config.batch_size,
                chunk_size=config.retry_chunk_size,
            )
            retry_segments = []
            for item in retry["segments"]:
                item = deepcopy(item)
                item["text"] = transliterate_serbian(item["text"])
                item["start"] += start
                item["end"] += start
                retry_segments.append(item)

            if not retry_segments:
                repaired.append(segment)
                continue

            repaired.extend(retry_segments)
            retries += 1
            logger.info(
                "Retried collapsed %.2f–%.2f (%d chars / %.1fs) -> %d segments",
                start,
                end,
                len(segment.get("text", "").strip()),
                end - start,
                len(retry_segments),
            )

        if retries:
            logger.info("Retried %d collapsed chunk(s)", retries)
        return repaired

    def free(self) -> None:
        """Release the models and clear the GPU cache."""
        logger.debug("Freeing Transcriber models")
        self.asr = self.align_model = self.align_meta = None
        gc.collect()
        torch.cuda.empty_cache()
