"""Typer CLI: transcribe, models, config subcommands."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Annotated, Optional

import typer

from transcriber.audio import (
    ALL_EXTENSIONS,
    AudioExtractionError,
    extract_audio,
    get_duration,
    needs_conversion,
)
from transcriber.config import (
    TranscriberConfig,
    load_config,
    save_config,
    set_config_value,
)
from transcriber.core import (
    DeviceError,
    ModelError,
    get_cached_models,
    load_model,
    resolve_device,
    transcribe_file,
    AVAILABLE_MODELS,
    MODEL_SIZES,
)
from transcriber.display import (
    LiveTranscriber,
    console,
    print_error,
    print_model_table,
    print_stats_table,
    print_warning,
)
from transcriber.formats import write_outputs
from transcriber.checkpoint import (
    checkpoint_path,
    save_checkpoint,
    load_checkpoint,
    deserialize_segments,
    delete_checkpoint,
)

app = typer.Typer(
    name="transcriber",
    help="Local audio/video transcription using faster-whisper.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)

models_app = typer.Typer(help="Manage Whisper models.", no_args_is_help=True)
config_app = typer.Typer(help="View and edit configuration.", no_args_is_help=True)

app.add_typer(models_app, name="models")
app.add_typer(config_app, name="config")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _expand_inputs(files: list[Path]) -> list[Path]:
    """Yield individual files; recurse into directories."""
    result: list[Path] = []
    for p in files:
        if p.is_dir():
            for ext in ALL_EXTENSIONS:
                result.extend(sorted(p.rglob(f"*{ext}")))
        else:
            result.append(p)
    return result


def _resolve_output_base(file: Path, output: Path | None) -> Path:
    if output is None:
        return file.parent / file.stem
    if output.is_dir():
        return output / file.stem
    # A path without suffix that doesn't exist yet is treated as a directory
    # intent (e.g. --output ./out/ or --output ./out).  A path that has a
    # suffix is treated as an explicit file stem.
    if not output.suffix:
        output.mkdir(parents=True, exist_ok=True)
        return output / file.stem
    return output.parent / output.stem


def _parse_formats(fmt_string: str | None, cfg_formats: list[str]) -> list[str]:
    if fmt_string:
        parts = [f.strip() for f in fmt_string.split(",")]
        return parts
    return cfg_formats


# ---------------------------------------------------------------------------
# transcribe command
# ---------------------------------------------------------------------------


@app.command()
def transcribe(
    files: Annotated[list[Path], typer.Argument(help="Audio/video files or directories.")],
    model: Annotated[Optional[str], typer.Option("--model", "-m", help="Whisper model size.")] = None,
    language: Annotated[Optional[str], typer.Option("--language", "-l", help="Language code (auto-detect if omitted).")] = None,
    task: Annotated[Optional[str], typer.Option("--task", help="'transcribe' or 'translate'.")] = None,
    format: Annotated[Optional[str], typer.Option("--format", "-f", help="Output format(s): txt,srt,vtt,tsv,json,all.")] = None,
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Output directory or file stem.")] = None,
    device: Annotated[Optional[str], typer.Option("--device", help="Device: auto, cpu, cuda.")] = None,
    compute_type: Annotated[Optional[str], typer.Option("--compute-type", help="Quantization: auto, int8, float16, etc.")] = None,
    vad: Annotated[Optional[bool], typer.Option("--vad/--no-vad", help="Enable/disable VAD filter.")] = None,
    word_timestamps: Annotated[bool, typer.Option("--word-timestamps", help="Enable word-level timestamps.")] = False,
    beam_size: Annotated[Optional[int], typer.Option("--beam-size", help="Beam size for decoding.")] = None,
    initial_prompt: Annotated[Optional[str], typer.Option("--initial-prompt", help="Initial prompt for the model.")] = None,
    copy: Annotated[bool, typer.Option("--copy", help="Copy transcript to clipboard.")] = False,
    stdout: Annotated[bool, typer.Option("--stdout", help="Print transcript to stdout.")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Suppress live UI.")] = False,
    resume: Annotated[bool, typer.Option("--resume", help="Resume from a previous interrupted transcription.")] = False,
) -> None:
    """Transcribe audio/video files."""
    cfg = load_config()

    # Option resolution: CLI value → config → hardcoded default
    resolved_model = model or cfg.model
    resolved_language = language or cfg.language
    resolved_task = task or cfg.task
    resolved_formats = _parse_formats(format, cfg.format)
    resolved_device = device or cfg.device
    resolved_compute = compute_type if compute_type else (None if cfg.compute_type == "auto" else cfg.compute_type)
    resolved_vad = vad if vad is not None else cfg.vad
    resolved_beam = beam_size or cfg.beam_size
    resolved_quiet = quiet or cfg.quiet

    # Validate formats early
    from transcriber.formats import SUPPORTED_FORMATS, _expand_formats
    try:
        _expand_formats(resolved_formats)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--format")

    # Resolve device/compute
    try:
        dev, ctype = resolve_device(resolved_device)
        if resolved_compute:
            ctype = resolved_compute
    except DeviceError as exc:
        print_error(str(exc))
        raise typer.Exit(1)

    # Load model once for all files
    if not resolved_quiet:
        console.print(f"[dim]Loading model [bold]{resolved_model}[/] on {dev} ({ctype})...[/]")
    try:
        whisper_model = load_model(resolved_model, dev, ctype)
    except ModelError as exc:
        print_error(str(exc))
        raise typer.Exit(1)

    exit_code = 0
    interrupted = False
    all_texts: list[str] = []

    expanded = _expand_inputs(files)
    if not expanded:
        print_error("No input files found.")
        raise typer.Exit(1)

    for file in expanded:
        if not file.exists():
            print_error(f"File not found: {file}")
            exit_code = 1
            continue

        duration = get_duration(file)
        output_base = _resolve_output_base(file, output)

        # --- checkpoint / resume ---
        prior_segments: list = []
        resume_from = 0.0
        info: dict = {}

        if resume:
            ckpt = load_checkpoint(output_base)
            if ckpt is None:
                if not resolved_quiet:
                    console.print(f"[dim]No checkpoint found for {file.name}, starting from scratch.[/]")
            else:
                src_mtime = file.stat().st_mtime if file.exists() else None
                if ckpt.get("source_mtime") and src_mtime and abs(ckpt["source_mtime"] - src_mtime) > 1:
                    print_warning(f"Source file {file.name} has changed since checkpoint was created. Ignoring checkpoint.")
                else:
                    resume_from = float(ckpt["resume_from"])
                    prior_segments = deserialize_segments(ckpt.get("segments", []))
                    info = ckpt.get("info", {})
                    if not resolved_quiet:
                        console.print(
                            f"[dim]Resuming {file.name} from "
                            f"{resume_from:.1f}s ({len(prior_segments)} segment(s) already done).[/]"
                        )

        audio_path = file

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            if needs_conversion(file):
                if not resolved_quiet:
                    action = "Extracting remaining audio" if resume_from > 0 else "Extracting audio"
                    console.print(f"[dim]{action} from {file.name}...[/]")
                try:
                    audio_path = extract_audio(file, tmp_path, start_time=resume_from)
                except AudioExtractionError as exc:
                    print_error(str(exc))
                    exit_code = 1
                    continue
            elif resume_from > 0:
                # Already a WAV but we still need to trim it.
                try:
                    audio_path = extract_audio(file, tmp_path, start_time=resume_from)
                except AudioExtractionError as exc:
                    print_error(str(exc))
                    exit_code = 1
                    continue

            try:
                segment_iter, info = transcribe_file(
                    whisper_model,
                    audio_path,
                    language=resolved_language,
                    task=resolved_task,
                    vad=resolved_vad,
                    word_timestamps=word_timestamps,
                    beam_size=resolved_beam,
                    initial_prompt=initial_prompt,
                )
            except Exception as exc:
                print_error(f"Transcription failed for {file.name}: {exc}")
                exit_code = 1
                continue

            new_segments = []
            interrupted = False
            with LiveTranscriber(quiet=resolved_quiet) as live:
                # Replay prior segments into the live display without re-emitting to stdout.
                for seg in prior_segments:
                    live.update(seg)
                try:
                    for seg in segment_iter:
                        # Offset timestamps to be relative to the original file.
                        if resume_from > 0:
                            seg.start += resume_from
                            seg.end += resume_from
                            for w in seg.words:
                                w.start += resume_from
                                w.end += resume_from
                        new_segments.append(seg)
                        live.update(seg)
                        if stdout:
                            print(seg.text.strip(), flush=True)
                except KeyboardInterrupt:
                    interrupted = True
                stats = live.get_stats()

        segments = prior_segments + new_segments

        if interrupted and not segments:
            raise KeyboardInterrupt

        if interrupted:
            print_warning(f"Interrupted after {len(new_segments)} new segment(s) — saving checkpoint.")
            save_checkpoint(
                output_base,
                source=file,
                resume_from=segments[-1].end,
                segments=segments,
                info=info,
            )
        meta = {
            "file": str(file),
            "language": info["language"],
            "language_probability": info["language_probability"],
            "duration": info["duration"],
        }

        written_paths: list[Path] = []
        if resolved_formats:
            try:
                written_paths = write_outputs(segments, resolved_formats, output_base, meta)
            except Exception as exc:
                print_error(f"Failed to write output for {file.name}: {exc}")
                exit_code = 1

        if not interrupted:
            delete_checkpoint(output_base)

        audio_duration = duration or info.get("duration") or stats["audio_end"]
        proc_time = stats["processing_time"]
        speed = stats["audio_end"] / proc_time if proc_time > 0 else 0.0

        if not resolved_quiet:
            print_stats_table(
                file=file,
                duration=audio_duration,
                language=info["language"],
                language_probability=info["language_probability"],
                segments=segments,
                processing_time=proc_time,
                output_paths=written_paths,
                speed_ratio=speed,
            )

        all_texts.append("\n".join(s.text.strip() for s in segments))

        if interrupted:
            break

    if copy and all_texts:
        try:
            import pyperclip
            pyperclip.copy("\n\n".join(all_texts))
            if not resolved_quiet:
                console.print("[dim]Transcript copied to clipboard.[/]")
        except Exception as exc:
            print_warning(f"Clipboard unavailable: {exc}")

    if interrupted:
        raise typer.Exit(130)

    if exit_code != 0:
        raise typer.Exit(exit_code)


# ---------------------------------------------------------------------------
# models subcommands
# ---------------------------------------------------------------------------


@models_app.command("list")
def models_list() -> None:
    """List available Whisper models and their cache status."""
    cached = get_cached_models()
    print_model_table(cached)


@models_app.command("download")
def models_download(
    name: Annotated[str, typer.Argument(help="Model name to download.")],
) -> None:
    """Download a Whisper model from HuggingFace."""
    if name not in AVAILABLE_MODELS:
        print_error(f"Unknown model: {name!r}. Available: {', '.join(AVAILABLE_MODELS)}")
        raise typer.Exit(1)
    console.print(f"[dim]Downloading [bold]{name}[/] (this may take a while)...[/]")
    try:
        load_model(name, "cpu", "int8")
        console.print(f"[green]Model [bold]{name}[/] ready.[/]")
    except ModelError as exc:
        print_error(str(exc))
        raise typer.Exit(1)


@models_app.command("clear")
def models_clear(
    name: Annotated[Optional[str], typer.Argument(help="Model to remove (omit for all).")] = None,
) -> None:
    """Remove cached model(s) from HuggingFace cache."""
    try:
        from huggingface_hub import scan_cache_dir
        from huggingface_hub._cache_manager import DeleteCacheStrategy
    except ImportError:
        print_error("huggingface_hub is not installed.")
        raise typer.Exit(1)

    cache_info = scan_cache_dir()
    to_delete: list[str] = []

    for repo in cache_info.repos:
        if not repo.repo_id.startswith("Systran/faster-whisper-"):
            continue
        repo_name = repo.repo_id.removeprefix("Systran/faster-whisper-")
        if name is None or repo_name == name:
            to_delete.extend(rev.commit_hash for rev in repo.revisions)

    if not to_delete:
        target = f"model {name!r}" if name else "any cached models"
        console.print(f"[dim]No cached files found for {target}.[/]")
        return

    strategy = cache_info.delete_revisions(*to_delete)
    freed = strategy.expected_freed_size_str
    console.print(f"[yellow]This will free ~{freed}. Proceeding...[/]")
    strategy.execute()
    console.print("[green]Done.[/]")


# ---------------------------------------------------------------------------
# config subcommands
# ---------------------------------------------------------------------------


@config_app.command("show")
def config_show() -> None:
    """Display the current configuration."""
    from dataclasses import asdict, fields
    cfg = load_config()
    from rich.table import Table
    table = Table(title="Current configuration", show_lines=False)
    table.add_column("Key", style="bold")
    table.add_column("Value")
    for fld in fields(cfg):
        val = getattr(cfg, fld.name)
        table.add_row(fld.name, str(val) if val is not None else "[dim]None[/]")
    console.print(table)


@config_app.command("set")
def config_set(
    key: Annotated[str, typer.Argument(help="Config key to set.")],
    value: Annotated[str, typer.Argument(help="Value to assign.")],
) -> None:
    """Set a configuration value."""
    try:
        set_config_value(key, value)
        console.print(f"[green]Set [bold]{key}[/] = {value}[/]")
    except KeyError as exc:
        print_error(str(exc))
        raise typer.Exit(1)


@config_app.command("reset")
def config_reset() -> None:
    """Reset configuration to defaults."""
    save_config(TranscriberConfig())
    console.print("[green]Configuration reset to defaults.[/]")
