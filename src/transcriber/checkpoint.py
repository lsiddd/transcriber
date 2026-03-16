"""Checkpoint persistence for resumable transcription."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from transcriber.core import SegmentResult, WordResult


def checkpoint_path(output_base: Path) -> Path:
    return output_base.with_suffix(".ckpt.json")


def save_checkpoint(
    output_base: Path,
    *,
    source: Path,
    resume_from: float,
    segments: list[SegmentResult],
    info: dict[str, Any],
) -> Path:
    data = {
        "version": 1,
        "source": str(source),
        "source_mtime": source.stat().st_mtime if source.exists() else None,
        "resume_from": resume_from,
        "info": info,
        "segments": [asdict(s) for s in segments],
    }
    path = checkpoint_path(output_base)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_checkpoint(output_base: Path) -> dict[str, Any] | None:
    path = checkpoint_path(output_base)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("version") != 1:
        return None
    return data


def deserialize_segments(raw: list[dict]) -> list[SegmentResult]:
    segments = []
    for s in raw:
        words = [WordResult(**w) for w in s.get("words", [])]
        segments.append(SegmentResult(
            id=s["id"],
            start=s["start"],
            end=s["end"],
            text=s["text"],
            words=words,
            avg_logprob=s.get("avg_logprob", 0.0),
            no_speech_prob=s.get("no_speech_prob", 0.0),
        ))
    return segments


def delete_checkpoint(output_base: Path) -> None:
    path = checkpoint_path(output_base)
    path.unlink(missing_ok=True)
