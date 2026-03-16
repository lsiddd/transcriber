"""Subtitle and transcript format writers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transcriber.core import SegmentResult

SUPPORTED_FORMATS = {"txt", "srt", "vtt", "tsv", "json"}


def _format_time_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_time_vtt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def write_txt(segments: list[SegmentResult], path: Path) -> None:
    path.write_text("\n".join(seg.text.strip() for seg in segments) + "\n", encoding="utf-8")


def write_srt(segments: list[SegmentResult], path: Path) -> None:
    lines: list[str] = []
    for seg in segments:
        lines.append(str(seg.id + 1))
        lines.append(f"{_format_time_srt(seg.start)} --> {_format_time_srt(seg.end)}")
        lines.append(seg.text.strip())
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_vtt(segments: list[SegmentResult], path: Path) -> None:
    lines = ["WEBVTT", ""]
    for seg in segments:
        lines.append(f"{_format_time_vtt(seg.start)} --> {_format_time_vtt(seg.end)}")
        lines.append(seg.text.strip())
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_tsv(segments: list[SegmentResult], path: Path) -> None:
    lines = ["start\tend\ttext"]
    for seg in segments:
        start_ms = int(seg.start * 1000)
        end_ms = int(seg.end * 1000)
        lines.append(f"{start_ms}\t{end_ms}\t{seg.text.strip()}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(segments: list[SegmentResult], path: Path, meta: dict[str, Any] | None = None) -> None:
    data = {
        "meta": meta or {},
        "segments": [
            {
                "id": seg.id,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
                "avg_logprob": seg.avg_logprob,
                "no_speech_prob": seg.no_speech_prob,
                "words": [
                    {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
                    for w in seg.words
                ],
            }
            for seg in segments
        ],
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


_WRITERS = {
    "txt": write_txt,
    "srt": write_srt,
    "vtt": write_vtt,
    "tsv": write_tsv,
}


def _expand_formats(formats: list[str]) -> list[str]:
    result: list[str] = []
    for fmt in formats:
        for part in fmt.split(","):
            part = part.strip().lower()
            if part == "all":
                result.extend(sorted(SUPPORTED_FORMATS))
            elif part in SUPPORTED_FORMATS:
                result.append(part)
            else:
                raise ValueError(f"Unsupported format: {part!r}. Choose from: {', '.join(sorted(SUPPORTED_FORMATS))}")
    # deduplicate while preserving order
    seen: set[str] = set()
    deduped = []
    for f in result:
        if f not in seen:
            seen.add(f)
            deduped.append(f)
    return deduped


def write_outputs(
    segments: list[SegmentResult],
    formats: list[str],
    output_base: Path,
    meta: dict[str, Any] | None = None,
) -> list[Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in _expand_formats(formats):
        out_path = output_base.with_suffix(f".{fmt}")
        if fmt == "json":
            write_json(segments, out_path, meta)
        else:
            _WRITERS[fmt](segments, out_path)
        written.append(out_path)
    return written
