# transcriber

A local audio and video transcription tool. It runs entirely on your machine, which means your files stay private and you do not have to pay per minute to a cloud API that will inevitably raise its prices.

Built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper), a CTranslate2 reimplementation of OpenAI's Whisper that is meaningfully faster and uses less memory than the original.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- ffmpeg in your PATH (`sudo apt install ffmpeg`, `brew install ffmpeg`, etc.)
- A GPU is optional but appreciated

## Installation

```bash
git clone <repo>
cd transcriber
uv sync
```

After that, `uv run transcriber` works from the project directory. If you want a global `transcriber` command, `uv tool install .` does it.

## Usage

### Transcribe a file

```bash
uv run transcriber transcribe interview.mp3
```

By default this uses the `base` model and writes a `.txt` file next to the source. If you want something more accurate, specify a larger model and accept that it will take longer.

```bash
uv run transcriber transcribe lecture.mp4 --model medium --format srt,vtt
```

### Batch transcription

Pass a directory and every supported audio/video file inside it (recursively) will be processed.

```bash
uv run transcriber transcribe ./recordings/ --output ./transcripts/ --model small
```

### Output formats

`--format` accepts a comma-separated list of `txt`, `srt`, `vtt`, `tsv`, `json`, or `all`.

```bash
uv run transcriber transcribe talk.mp4 --format all
```

### Translation

Whisper can translate to English directly without a separate translation step.

```bash
uv run transcriber transcribe foreign_film.mkv --task translate --model large-v3
```

### Word-level timestamps and clipboard

```bash
uv run transcriber transcribe meeting.mp3 --word-timestamps --format json --copy
```

`--copy` puts the plain-text transcript on your clipboard when done. `--stdout` prints segments to stdout as they arrive, if you need to pipe them somewhere.

### Resuming an interrupted transcription

If you interrupt a long transcription with Ctrl+C, the tool saves a checkpoint file. Run the same command with `--resume` to pick up where it stopped instead of starting over.

```bash
# first run, interrupted
uv run transcriber transcribe three_hour_conference.mp4 --model large-v3

# continue from the checkpoint
uv run transcriber transcribe three_hour_conference.mp4 --model large-v3 --resume
```

The checkpoint is deleted automatically once the transcription completes successfully.

## Models

```bash
uv run transcriber models list      # shows all models and which ones are cached locally
uv run transcriber models download small
uv run transcriber models clear large-v2   # frees disk space
```

Approximate sizes for reference:

| Model | Size |
|---|---|
| tiny / tiny.en | ~75 MB |
| base / base.en | ~145 MB |
| small / small.en | ~465 MB |
| medium / medium.en | ~1.5 GB |
| large-v3 | ~2.9 GB |
| large-v3-turbo / turbo | ~1.6 GB |

Models are downloaded from HuggingFace on first use and cached in the standard HuggingFace cache directory (`~/.cache/huggingface`).

## Configuration

Persistent defaults live at `~/.config/transcriber/config.toml`. Rather than typing `--model medium --language pt --format srt` on every invocation, set them once:

```bash
uv run transcriber config set model medium
uv run transcriber config set language pt
uv run transcriber config set format srt

uv run transcriber config show    # inspect current values
uv run transcriber config reset   # back to defaults
```

CLI flags always override config values.

## Device selection

By default the tool picks CUDA if a GPU is available and falls back to CPU otherwise. Override explicitly with `--device cpu` or `--device cuda`. Quantization can be set with `--compute-type` (e.g. `int8`, `float16`, `int8_float16`).

## Supported formats

Audio: `.mp3` `.wav` `.flac` `.ogg` `.opus` `.m4a` `.aac` `.wma` `.aiff` `.ape` and others.

Video: `.mp4` `.mkv` `.avi` `.mov` `.webm` `.ts` `.m4v` and others.

Any file format that ffmpeg can read will work. The audio is extracted to a temporary 16kHz mono WAV before being passed to the model, and the temporary file is deleted when the process exits.

## License

MIT.
