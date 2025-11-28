from __future__ import annotations

import copy
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _parse_override(value: str) -> Any:
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None"}:
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def apply_overrides(config: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    updated = copy.deepcopy(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}'. Expected format key=value")
        key, raw_value = override.split("=", 1)
        value = _parse_override(raw_value)
        keys = key.split(".")
        target = updated
        for part in keys[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]
        target[keys[-1]] = value
    return updated


def generate_run_directory(base_dir: Path, experiment_name: str, batch_name: Optional[str] = None) -> Path:
    """실험 디렉토리 생성. batch_name이 있으면 batch 폴더 내에 v1, v2, v3 형식으로 생성."""
    sanitized_name = re.sub(r"[^A-Za-z0-9_\-]", "_", experiment_name)
    
    if batch_name:
        # Batch 폴더 내에 버전별로 저장
        batch_dir = base_dir / batch_name
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        # 기존 버전 확인
        existing_versions = []
        for item in batch_dir.iterdir():
            if item.is_dir() and item.name.startswith(f"{sanitized_name}_v"):
                try:
                    version_num = int(item.name.split("_v")[-1])
                    existing_versions.append(version_num)
                except ValueError:
                    pass
        
        # 다음 버전 번호 결정
        next_version = max(existing_versions) + 1 if existing_versions else 1
        run_dir = batch_dir / f"{sanitized_name}_v{next_version}"
    else:
        # 기존 방식: 타임스탬프 기반
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_dir = base_dir / f"{timestamp}_{sanitized_name}"
        counter = 1
        while run_dir.exists():
            run_dir = base_dir / f"{timestamp}_{sanitized_name}_{counter:02d}"
            counter += 1
    
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def set_reproducibility(seed: int) -> None:
    import random

    random.seed(seed)
    try:  # numpy may not be present in some environments
        import numpy as np

        np.random.seed(seed)
    except Exception:  # pragma: no cover - optional dependency
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover - optional dependency
        pass


def list_result_directories(base_dir: Path) -> list[Path]:
    if not base_dir.exists():
        return []
    return sorted([p for p in base_dir.iterdir() if p.is_dir()])


__all__ = [
    "load_yaml",
    "save_yaml",
    "save_json",
    "apply_overrides",
    "deep_update",
    "generate_run_directory",
    "set_reproducibility",
    "list_result_directories",
]
