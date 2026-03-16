"""ffmpeg/ffprobe subprocess wrappers for audio extraction."""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".flac", ".ogg", ".opus", ".m4a", ".aac",
    ".wma", ".aiff", ".au", ".ra", ".amr", ".ape", ".mka",
}

VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm",
    ".m4v", ".mpeg", ".mpg", ".ts", ".mts", ".m2ts", ".vob",
    ".ogv", ".3gp", ".3g2", ".f4v", ".rmvb",
}

ALL_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


class AudioExtractionError(Exception):
    pass


def needs_conversion(path: Path) -> bool:
    """Return True for anything that isn't already a 16kHz mono WAV."""
    if path.suffix.lower() != ".wav":
        return True
    # We could inspect the WAV headers here, but it's safer to always
    # re-encode via ffmpeg to guarantee the expected format.
    return True


def extract_audio(path: Path, tmp_dir: Path, start_time: float = 0.0) -> Path:
    """Convert *path* to a 16kHz mono PCM WAV in *tmp_dir*.

    If *start_time* > 0, only the audio from that position onward is extracted
    (used when resuming an interrupted transcription).

    Raises AudioExtractionError if ffmpeg is unavailable or fails.
    """
    suffix = f"_from{start_time:.1f}" if start_time > 0 else ""
    out = tmp_dir / (path.stem + f"_16k{suffix}.wav")
    cmd = ["ffmpeg", "-y"]
    if start_time > 0:
        cmd += ["-ss", str(start_time)]
    cmd += [
        "-i", str(path),
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(out),
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        raise AudioExtractionError(
            "ffmpeg not found. Install it with your package manager "
            "(e.g. `sudo apt install ffmpeg` or `brew install ffmpeg`)."
        )

    if result.returncode != 0:
        stderr_tail = result.stderr[-800:].strip() if result.stderr else "(no output)"
        raise AudioExtractionError(
            f"ffmpeg failed for {path.name}:\n{stderr_tail}"
        )

    return out


def get_duration(path: Path) -> float | None:
    """Return duration in seconds via ffprobe, or None on failure."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        str(path),
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except FileNotFoundError:
        return None

    if result.returncode != 0:
        return None

    try:
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError):
        return None
