"""TOML-backed configuration at ~/.config/transcriber/config.toml."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any

import tomli_w

CONFIG_PATH = Path.home() / ".config" / "transcriber" / "config.toml"


@dataclass
class TranscriberConfig:
    model: str = "base"
    language: str | None = None
    task: str = "transcribe"
    format: list[str] = field(default_factory=lambda: ["txt"])
    device: str = "auto"
    compute_type: str = "auto"
    vad: bool = True
    word_timestamps: bool = False
    beam_size: int = 5
    output_dir: str | None = None
    quiet: bool = False


def load_config() -> TranscriberConfig:
    if not CONFIG_PATH.exists():
        return TranscriberConfig()
    with open(CONFIG_PATH, "rb") as f:
        data = tomllib.load(f)
    cfg = TranscriberConfig()
    for fld in fields(cfg):
        if fld.name in data:
            setattr(cfg, fld.name, data[fld.name])
    return cfg


def save_config(cfg: TranscriberConfig) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    # tomli_w cannot serialize None; omit None-valued keys entirely.
    data = {k: v for k, v in asdict(cfg).items() if v is not None}
    with open(CONFIG_PATH, "wb") as f:
        tomli_w.dump(data, f)


def set_config_value(key: str, value: str) -> None:
    cfg = load_config()
    fld_map = {fld.name: fld for fld in fields(cfg)}
    if key not in fld_map:
        raise KeyError(f"Unknown config key: {key!r}")

    fld = fld_map[key]
    coerced: Any
    if fld.type in ("bool", bool) or (isinstance(fld.type, str) and "bool" in fld.type):
        coerced = value.lower() in ("true", "1", "yes")
    elif fld.type in ("int", int) or (isinstance(fld.type, str) and fld.type == "int"):
        coerced = int(value)
    elif fld.type in ("list[str]",) or (isinstance(fld.type, str) and "list" in fld.type):
        coerced = [v.strip() for v in value.split(",")]
    elif fld.type in ("str | None", "Optional[str]") or (
        isinstance(fld.type, str) and "None" in fld.type
    ):
        coerced = None if value.lower() in ("none", "null", "") else value
    else:
        coerced = value

    setattr(cfg, key, coerced)
    save_config(cfg)
