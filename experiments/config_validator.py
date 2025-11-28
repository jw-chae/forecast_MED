from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence


_REQUIRED_TOP_LEVEL = {"experiment", "data", "model", "evaluation", "logging", "resources"}


class ConfigValidationError(ValueError):
    """Raised when a configuration file fails validation."""


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigValidationError(message)


def _validate_experiment_section(section: Dict[str, Any]) -> None:
    required_keys = {"name", "description", "random_seed"}
    missing = required_keys - section.keys()
    _assert(not missing, f"experiment section missing keys: {sorted(missing)}")
    _assert(isinstance(section.get("name"), str) and section["name"], "experiment.name must be a non-empty string")
    _assert(isinstance(section.get("random_seed"), (int, float)), "experiment.random_seed must be numeric")


def _validate_data_section(section: Dict[str, Any], search_dirs: Sequence[Path]) -> None:
    required_keys = {"source", "train_start", "train_end", "test_start", "test_end"}
    missing = required_keys - section.keys()
    _assert(not missing, f"data section missing keys: {sorted(missing)}")

    source = section["source"]

    def _resolve(path_str: str) -> Path | None:
        candidate = Path(path_str)
        if candidate.is_absolute() and candidate.exists():
            return candidate
        for root in search_dirs:
            candidate = root / path_str
            if candidate.exists():
                return candidate
        return None

    if isinstance(source, str):
        resolved = _resolve(source)
        _assert(resolved is not None, f"data source not found: {source}")
    elif isinstance(source, (list, tuple)):
        for path in source:
            resolved = _resolve(str(path))
            _assert(resolved is not None, f"data source not found: {path}")
    else:
        raise ConfigValidationError("data.source must be a string or list of strings")

    rolling = section.get("rolling", {})
    if rolling.get("enabled"):
        _assert(isinstance(rolling.get("step_size"), int) and rolling["step_size"] > 0, "rolling.step_size must be positive integer")
        _assert(isinstance(rolling.get("min_train_weeks"), int) and rolling["min_train_weeks"] > 0, "rolling.min_train_weeks must be positive integer")


def _validate_model_section(section: Dict[str, Any]) -> None:
    model_type = section.get("type")
    _assert(isinstance(model_type, str) and model_type, "model.type must be a non-empty string")
    forecast = section.get("forecast", {})
    if forecast:
        horizon = forecast.get("horizon")
        _assert(isinstance(horizon, int) and horizon > 0, "model.forecast.horizon must be a positive integer")
        quantiles = forecast.get("quantiles", [])
        if quantiles:
            _assert(isinstance(quantiles, Iterable), "model.forecast.quantiles must be iterable")
            for q in quantiles:
                _assert(0.0 < float(q) < 1.0, f"invalid quantile value: {q}")


def _validate_evaluation_section(section: Dict[str, Any]) -> None:
    metrics = section.get("metrics", [])
    _assert(metrics, "evaluation.metrics must contain at least one metric")
    _assert(all(isinstance(m, str) for m in metrics), "evaluation.metrics must be strings")


def validate_config(
    config: Dict[str, Any],
    *,
    config_path: str | os.PathLike[str] | None = None,
    project_root: Path | None = None,
) -> None:
    """Validate a loaded configuration dictionary."""
    missing_sections = _REQUIRED_TOP_LEVEL - config.keys()
    _assert(not missing_sections, f"config missing top-level sections: {sorted(missing_sections)}")

    config_dir = Path(config_path).resolve().parent if config_path else Path.cwd()
    search_dirs = [config_dir]
    if project_root and project_root not in search_dirs:
        search_dirs.append(project_root)

    _assert(isinstance(config["experiment"], dict), "experiment section must be a mapping")
    _assert(isinstance(config["data"], dict), "data section must be a mapping")
    _assert(isinstance(config["model"], dict), "model section must be a mapping")
    _assert(isinstance(config["evaluation"], dict), "evaluation section must be a mapping")

    _validate_experiment_section(config["experiment"])
    _validate_data_section(config["data"], tuple(search_dirs))
    _validate_model_section(config["model"])
    _validate_evaluation_section(config["evaluation"])


__all__ = ["validate_config", "ConfigValidationError"]
