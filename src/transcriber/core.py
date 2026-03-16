"""WhisperModel wrapper with lazy segment generator."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

AVAILABLE_MODELS = [
    "tiny", "tiny.en",
    "base", "base.en",
    "small", "small.en",
    "medium", "medium.en",
    "large-v1", "large-v2", "large-v3",
    "large-v3-turbo", "turbo",
    "distil-small.en", "distil-medium.en", "distil-large-v2", "distil-large-v3",
]

MODEL_SIZES = {
    "tiny": "~75 MB",
    "tiny.en": "~75 MB",
    "base": "~145 MB",
    "base.en": "~145 MB",
    "small": "~465 MB",
    "small.en": "~465 MB",
    "medium": "~1.5 GB",
    "medium.en": "~1.5 GB",
    "large-v1": "~2.9 GB",
    "large-v2": "~2.9 GB",
    "large-v3": "~2.9 GB",
    "large-v3-turbo": "~1.6 GB",
    "turbo": "~1.6 GB",
    "distil-small.en": "~335 MB",
    "distil-medium.en": "~750 MB",
    "distil-large-v2": "~1.5 GB",
    "distil-large-v3": "~1.5 GB",
}


class DeviceError(Exception):
    pass


class ModelError(Exception):
    pass


@dataclass
class WordResult:
    word: str
    start: float
    end: float
    probability: float


@dataclass
class SegmentResult:
    id: int
    start: float
    end: float
    text: str
    words: list[WordResult] = field(default_factory=list)
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0


def resolve_device(device: str) -> tuple[str, str]:
    """Return (device, compute_type) pair.

    Raises DeviceError if CUDA is requested but unavailable.
    """
    if device == "auto":
        try:
            import ctranslate2
            if ctranslate2.get_cuda_device_count() > 0:
                return "cuda", "float16"
        except Exception:
            pass
        return "cpu", "int8"

    if device == "cuda":
        try:
            import ctranslate2
            if ctranslate2.get_cuda_device_count() == 0:
                raise DeviceError(
                    "CUDA device requested but none is available. "
                    "Use --device cpu or --device auto."
                )
        except ImportError:
            raise DeviceError("ctranslate2 is not installed; cannot check CUDA availability.")
        return "cuda", "float16"

    return device, "int8"


def load_model(model_size: str, device: str, compute_type: str):
    """Load and return a WhisperModel instance."""
    try:
        from faster_whisper import WhisperModel
        return WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as exc:
        raise ModelError(f"Failed to load model {model_size!r}: {exc}") from exc


def transcribe_file(
    model,
    path: Path,
    *,
    language: str | None = None,
    task: str = "transcribe",
    vad: bool = True,
    word_timestamps: bool = False,
    beam_size: int = 5,
    initial_prompt: str | None = None,
) -> tuple[Iterator[SegmentResult], dict[str, Any]]:
    """Transcribe *path* and return a lazy segment iterator plus metadata dict."""
    segments_raw, info = model.transcribe(
        str(path),
        language=language,
        task=task,
        vad_filter=vad,
        word_timestamps=word_timestamps,
        beam_size=beam_size,
        initial_prompt=initial_prompt,
    )

    info_dict = {
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
    }

    def _generate() -> Iterator[SegmentResult]:
        for seg in segments_raw:
            words: list[WordResult] = []
            if word_timestamps and seg.words:
                words = [
                    WordResult(
                        word=w.word,
                        start=w.start,
                        end=w.end,
                        probability=w.probability,
                    )
                    for w in seg.words
                ]
            yield SegmentResult(
                id=seg.id,
                start=seg.start,
                end=seg.end,
                text=seg.text,
                words=words,
                avg_logprob=seg.avg_logprob,
                no_speech_prob=seg.no_speech_prob,
            )

    return _generate(), info_dict


def get_cached_models() -> set[str]:
    """Return set of model name strings that are locally cached via HuggingFace."""
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        cached = set()
        for repo in cache_info.repos:
            repo_id = repo.repo_id  # e.g. "Systran/faster-whisper-base"
            if repo_id.startswith("Systran/faster-whisper-"):
                name = repo_id.removeprefix("Systran/faster-whisper-")
                cached.add(name)
        return cached
    except Exception:
        return set()
