"""Dynaconf centralized configuration loader."""

from __future__ import annotations

from pathlib import Path

from dynaconf import Dynaconf

_settings_path = Path(__file__).resolve().parent / "settings.toml"

settings = Dynaconf(
    envvar_prefix="INDEXER",
    settings_files=[str(_settings_path)],
    environments=True,
    load_dotenv=True,
)
