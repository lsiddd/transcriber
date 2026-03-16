"""Rich-based UI: live transcription view, stats table, model table."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from transcriber.core import SegmentResult, AVAILABLE_MODELS, MODEL_SIZES

# All UI goes to stderr; stdout is reserved for --stdout mode.
console = Console(stderr=True)


class LiveTranscriber:
    """Context manager that renders a live transcript panel."""

    def __init__(self, quiet: bool = False) -> None:
        self._quiet = quiet
        self._lines: list[str] = []
        self._segment_count = 0
        self._audio_end = 0.0
        self._start_time = 0.0
        self._live: Live | None = None

    def __enter__(self) -> "LiveTranscriber":
        self._start_time = time.monotonic()
        if not self._quiet:
            self._live = Live(
                self._render(),
                console=console,
                refresh_per_second=8,
                transient=False,
            )
            self._live.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        if self._live is not None:
            self._live.__exit__(*args)

    def update(self, segment: SegmentResult) -> None:
        self._segment_count += 1
        self._audio_end = segment.end
        text = segment.text.strip()
        if text:
            self._lines.append(text)
        if self._live is not None:
            self._live.update(self._render())

    def _render(self) -> Panel:
        elapsed = time.monotonic() - self._start_time
        speed = self._audio_end / elapsed if elapsed > 0 else 0.0
        title = (
            f"[bold cyan]Transcribing[/] "
            f"[dim]| segments: {self._segment_count} "
            f"| elapsed: {elapsed:.1f}s "
            f"| speed: {speed:.1f}x[/]"
        )
        display_lines = self._lines[-20:]
        body = Text("\n".join(display_lines), overflow="fold")
        return Panel(body, title=title, border_style="blue", expand=True)

    def get_stats(self) -> dict[str, Any]:
        return {
            "segment_count": self._segment_count,
            "processing_time": time.monotonic() - self._start_time,
            "audio_end": self._audio_end,
        }


def print_stats_table(
    *,
    file: Path,
    duration: float | None,
    language: str,
    language_probability: float,
    segments: list[SegmentResult],
    processing_time: float,
    output_paths: list[Path],
    speed_ratio: float,
) -> None:
    words = " ".join(s.text for s in segments).split()
    word_count = len(words)
    wpm = (word_count / (duration / 60)) if duration and duration > 0 else 0.0

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="dim", width=18)
    table.add_column()

    table.add_row("File", str(file))
    if duration is not None:
        table.add_row("Duration", f"{duration:.1f}s")
    table.add_row("Language", f"{language} ({language_probability:.0%})")
    table.add_row("Segments", str(len(segments)))
    table.add_row("Words", f"{word_count} ({wpm:.0f} wpm)")
    table.add_row("Speed ratio", f"{speed_ratio:.2f}x")
    table.add_row("Processing", f"{processing_time:.1f}s")
    if output_paths:
        table.add_row("Output", str(output_paths[0]))
        for p in output_paths[1:]:
            table.add_row("", str(p))

    console.print(Panel(table, title="[bold green]Transcription complete[/]", border_style="green"))


def print_model_table(cached: set[str]) -> None:
    table = Table(title="Available models", show_lines=False)
    table.add_column("Model", style="bold")
    table.add_column("Cached", justify="center")
    table.add_column("Approx. size", justify="right", style="dim")

    for name in AVAILABLE_MODELS:
        is_cached = name in cached
        cached_cell = "[green]yes[/]" if is_cached else "[dim]no[/]"
        size = MODEL_SIZES.get(name, "?")
        table.add_row(name, cached_cell, size)

    console.print(table)


def print_error(message: str) -> None:
    console.print(f"[bold red]Error:[/] {message}")


def print_warning(message: str) -> None:
    console.print(f"[bold yellow]Warning:[/] {message}")
