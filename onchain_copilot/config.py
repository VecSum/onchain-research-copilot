"""Configuration loading.

Config is a single TOML file. Every key has a default, so an empty file still
produces a working pipeline. Secret-shaped values (API keys) are never read
from the file: the file names an environment variable instead, which keeps
secrets out of version control by construction.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when the config file is missing or inconsistent."""


DEFAULTS: dict[str, dict[str, Any]] = {
    "run": {"data_mode": "fixtures"},
    "providers": {
        "order": ["geckoterminal", "dexscreener"],
        "cache_dir": "output/cache",
        "cache_ttl_seconds": 900,
        "timeout_seconds": 12,
        "retries": 2,
        "user_agent": "onchain-research-copilot/0.1",
    },
    "metrics": {
        "volatility_window": 24,
        "momentum_fast": 7,
        "momentum_slow": 21,
        "rsi_period": 14,
        "risk_free_rate": 0.0,
        "periods_per_year": 365,
    },
    "retrieval": {
        "corpus_dir": "data/corpus",
        "index_dir": "output/index",
        "chunk_chars": 460,
        "chunk_overlap": 60,
        "rrf_k": 60,
        "lexical_weight": 1.0,
        "semantic_weight": 1.0,
        "dim": 256,
        "top_k": 5,
        "rerank_enabled": True,
        "refusal_threshold": 0.11,
        "max_evidence_per_report": 6,
    },
    "llm": {
        "provider": "deterministic",
        "base_url": "http://127.0.0.1:8080/v1",
        "model": "qwen2.5-7b-instruct",
        "api_key_env": "LLM_API_KEY",
        "temperature": 0.1,
        "max_tokens": 1200,
        "timeout_seconds": 60,
        "fallback_to_deterministic": True,
    },
    "report": {
        "title": "On-chain Research Report",
        "language": "en",
        "include_evidence_quotes": True,
        "disclaimer": "Research artefact generated from configured data sources. Not investment advice.",
    },
}

VALID_DATA_MODES = {"fixtures", "live", "synthetic"}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass(frozen=True)
class Config:
    data: dict[str, Any] = field(default_factory=lambda: _merge({}, DEFAULTS))
    source_path: Path | None = None

    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        if path is None:
            return cls()
        resolved = Path(path)
        if not resolved.exists():
            raise ConfigError(f"config file not found: {resolved}")
        with resolved.open("rb") as handle:
            loaded = tomllib.load(handle)
        merged = _merge(DEFAULTS, loaded)
        _validate(merged)
        return cls(data=merged, source_path=resolved)

    def section(self, name: str) -> dict[str, Any]:
        return self.data.get(name, {})

    def get(self, section: str, key: str) -> Any:
        return self.section(section).get(key)


def _validate(section_map: dict[str, Any]) -> None:
    mode = section_map["run"]["data_mode"]
    if mode not in VALID_DATA_MODES:
        raise ConfigError(
            f"run.data_mode must be one of {sorted(VALID_DATA_MODES)}, got {mode!r}"
        )
    retrieval = section_map["retrieval"]
    if retrieval["chunk_overlap"] >= retrieval["chunk_chars"]:
        raise ConfigError("retrieval.chunk_overlap must be smaller than chunk_chars")
    if retrieval["top_k"] < 1:
        raise ConfigError("retrieval.top_k must be at least 1")
    if not section_map["providers"]["order"]:
        raise ConfigError("providers.order must list at least one provider")


def load_config(path: str | Path | None = "config.toml") -> Config:
    """Load config if the file exists, otherwise return defaults."""

    if path is not None and not Path(path).exists():
        return Config()
    return Config.load(path)
