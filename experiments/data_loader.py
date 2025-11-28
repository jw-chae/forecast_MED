from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


DISEASE_ALIASES: Dict[str, Iterable[str]] = {
    "influenza": ["\u6d41\u884c\u6027\u611f\u5192", "flu"],
    "hand_foot_mouth": ["\u624b\u8db3\u53e3\u75c5", "hfmd"],
    "\u624b\u8db3\u53e3\u75c5": ["hand_foot_mouth", "hfmd"],
}


@dataclass(slots=True)
class DatasetSplit:
    """Container for train/validation/test splits."""

    train: pd.DataFrame
    validation: Optional[pd.DataFrame]
    test: pd.DataFrame
    metadata: Dict[str, object]


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "diagnosis_time" in df.columns:
        df["diagnosis_time"] = pd.to_datetime(df["diagnosis_time"])
        df = df.set_index("diagnosis_time").sort_index()
    elif "INSPECTION_DATE" in df.columns:
        df["INSPECTION_DATE"] = pd.to_datetime(df["INSPECTION_DATE"])
        df = df.set_index("INSPECTION_DATE").sort_index()
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
    else:
        raise ValueError(f"Unsupported CSV schema (missing date column) in {path}")
    return df


def _select_disease(df: pd.DataFrame, disease: str | Iterable[str]) -> pd.DataFrame:
    if isinstance(disease, str):
        target = disease
        if target not in df.columns:
            aliases = DISEASE_ALIASES.get(disease.lower())
            if aliases:
                for alias in aliases:
                    if alias in df.columns:
                        target = alias
                        break
        if target not in df.columns:
            raise KeyError(f"disease '{disease}' not present in data columns")
        series = df[[target]].rename(columns={target: "value"})
        series["disease"] = disease
        return series
    selected: List[pd.DataFrame] = []
    for d in disease:
        target = d
        if target not in df.columns and isinstance(d, str):
            aliases = DISEASE_ALIASES.get(d.lower())
            if aliases:
                for alias in aliases:
                    if alias in df.columns:
                        target = alias
                        break
        if target not in df.columns:
            raise KeyError(f"disease '{d}' not present in data columns")
        tmp = df[[target]].rename(columns={target: "value"})
        tmp["disease"] = d
        selected.append(tmp)
    return pd.concat(selected, axis=0)


def load_timeseries(config: Dict[str, object], base_dir: Path) -> pd.DataFrame:
    """Load time series data based on the configuration."""
    source = config["source"]
    if isinstance(source, str):
        paths = [base_dir / source]
    else:
        paths = [base_dir / str(p) for p in source]

    frames: List[pd.DataFrame] = []
    for path in paths:
        if path.suffix == ".json":
            data = json.loads(path.read_text())
            df = pd.DataFrame(data)
            if "date" not in df.columns:
                raise ValueError(f"JSON source missing 'date' column: {path}")
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
        else:
            df = _read_csv(path)
        frames.append(df)
    merged = frames[0] if len(frames) == 1 else pd.concat(frames, axis=0).groupby(level=0).sum()

    disease = config.get("disease", "influenza")
    series = _select_disease(merged, disease)
    if isinstance(disease, Iterable) and not isinstance(disease, str):
        # Pivot to multi-disease wide format maintaining temporal alignment
        series = series.reset_index().pivot_table(index="diagnosis_time" if "diagnosis_time" in series.index.names else "date", columns="disease", values="value")

    return series.sort_index()


def slice_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df.loc[pd.Timestamp(start) : pd.Timestamp(end)]


def create_splits(config: Dict[str, object], base_dir: Path) -> DatasetSplit:
    df = load_timeseries(config, base_dir)

    train = slice_period(df, str(config["train_start"]), str(config["train_end"]))
    test = slice_period(df, str(config["test_start"]), str(config["test_end"]))

    validation: Optional[pd.DataFrame] = None
    val_ratio = config.get("validation_split")
    if val_ratio:
        if not 0 < float(val_ratio) < 1:
            raise ValueError("validation_split must be between 0 and 1")
        val_weeks = max(1, int(len(train) * float(val_ratio)))
        validation = train.iloc[-val_weeks:]
        train = train.iloc[:-val_weeks]

    metadata = {
        "train_start": config["train_start"],
        "train_end": config["train_end"],
        "test_start": config["test_start"],
        "test_end": config["test_end"],
        "rolling": config.get("rolling", {}),
    }

    return DatasetSplit(train=train, validation=validation, test=test, metadata=metadata)


def generate_rolling_windows(
    full_series: pd.DataFrame,
    *,
    min_train_weeks: int,
    step_size: int,
    horizon: int,
    start_date: str,
    end_date: str,
    forecast_start: Optional[str] = None,
    forecast_end: Optional[str] = None,
) -> Iterable[Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]]:
    series = full_series.copy()
    series = series.loc[pd.Timestamp(start_date) : pd.Timestamp(end_date)]

    target_index = series.index
    start_idx = max(min_train_weeks, 0)
    if forecast_start:
        forecast_ts = pd.Timestamp(forecast_start)
        start_idx = max(start_idx, target_index.searchsorted(forecast_ts, side="left"))
    stop_idx = len(series) - horizon + 1
    if forecast_end:
        forecast_ts = pd.Timestamp(forecast_end)
        stop_idx = min(stop_idx, target_index.searchsorted(forecast_ts, side="right") - horizon + 1)
    if stop_idx <= start_idx:
        return
    for idx in range(start_idx, stop_idx, step_size):
        train_slice = series.iloc[idx - min_train_weeks : idx]
        test_slice = series.iloc[idx : idx + horizon]
        as_of = target_index[idx - 1]
        yield train_slice, test_slice, as_of


__all__ = [
    "DatasetSplit",
    "create_splits",
    "generate_rolling_windows",
    "load_timeseries",
]
