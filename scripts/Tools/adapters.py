from __future__ import annotations

from typing import Tuple, List
import pandas as pd
import numpy as np


def load_his_outpatient_series(
    csv_path: str, disease: str, fill_na: bool = False
) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    """Load HIS outpatient weekly time series data for a specific disease."""
    df = pd.read_csv(csv_path)
    dates = pd.to_datetime(df["diagnosis_time"])
    series = df[disease].values
    if fill_na:
        series = pd.Series(series).fillna(0).values
    return dates, series


def available_diseases(csv_path: str) -> List[str]:
    df = pd.read_csv(csv_path, nrows=1)
    return [c for c in df.columns if c != "diagnosis_time"]

