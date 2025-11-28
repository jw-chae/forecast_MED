from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional, Mapping

# matplotlib 백엔드를 먼저 설정
os.environ.setdefault("MPLBACKEND", "Agg")

try:
    import matplotlib
    matplotlib.use("Agg", force=True)  # GUI 백엔드 없이 사용
    import matplotlib.pyplot as plt
    import seaborn as sns
    _VISUALIZATION_AVAILABLE = True
except (ImportError, Exception) as e:
    _VISUALIZATION_AVAILABLE = False
    plt = None
    sns = None

import numpy as np
import pandas as pd

from .metrics import PredictionBundle

if _VISUALIZATION_AVAILABLE:
    # Seaborn 스타일 설정
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    plt.rcParams["figure.figsize"] = (14, 6)
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10


def plot_forecast_vs_actual(
    actual: pd.Series,
    bundle: PredictionBundle,
    title: str = "Forecast vs Actual",
    save_path: Optional[Path] = None,
    y_max: Optional[float] = 50.0,
    show_title: bool = False,
) -> None:
    """예측값과 실제값을 비교하는 시계열 플롯.
    
    논문 스타일 (NeurIPS/ICLR):
    - 90% prediction interval (q05-q95)
    - Shaded interval: #EECFA1, alpha=0.20
    - Actual: solid black line with dot markers (●)
    - Prediction: dashed line without markers
    - No title (move to figure caption)
    """
    if not _VISUALIZATION_AVAILABLE:
        return
    
    # 데이터 준비
    df_plot = pd.DataFrame({
        "date": actual.index,
        "Actual": actual.values,
        "Prediction": bundle.point.values,
    })
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 90% Prediction Interval 표시 (q05-q95)
    if bundle.quantiles:
        quantiles_sorted = sorted(bundle.quantiles.keys())
        # 90% interval: q05 and q95
        lower_q = 0.05 if 0.05 in bundle.quantiles else min(quantiles_sorted)
        upper_q = 0.95 if 0.95 in bundle.quantiles else max(quantiles_sorted)
        lower_series = bundle.quantiles[lower_q]
        upper_series = bundle.quantiles[upper_q]
        
        # 구간 채우기 - 논문 스타일: subtle beige/tan color
        ax.fill_between(
            lower_series.index,
            lower_series.values,
            upper_series.values,
            alpha=0.20,
            color="#EECFA1",  # Soft beige/tan color
            label="90% Prediction Interval",
            linewidth=0,
        )
    
    # Actual: solid black line with dot markers (●)
    sns.lineplot(data=df_plot, x="date", y="Actual", label="Actual", 
                 color="black", linewidth=2.0, marker="o", markersize=4, ax=ax)
    
    # Prediction: dashed line WITHOUT markers (clean look)
    sns.lineplot(data=df_plot, x="date", y="Prediction", label="Prediction",
                 color="#E07B39",  # Warm orange for contrast
                 linewidth=2.0, linestyle="--", ax=ax)
    
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cases", fontsize=12)
    
    # 제목은 기본적으로 숨김 (논문에서는 figure caption으로 이동)
    if show_title:
        ax.set_title(title, fontsize=14, fontweight="bold")
    
    # Y축 범위 설정 - 시즌 패턴이 더 잘 보이도록
    if y_max is not None:
        ax.set_ylim(bottom=0, top=y_max)
    else:
        ax.set_ylim(bottom=0)
    
    ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="gray")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_residuals(
    actual: pd.Series,
    bundle: PredictionBundle,
    title: str = "Residuals Plot",
    save_path: Optional[Path] = None,
) -> None:
    """잔차(residuals) 플롯."""
    if not _VISUALIZATION_AVAILABLE:
        return
    residuals = actual - bundle.point
    df_residuals = pd.DataFrame({
        "date": residuals.index,
        "residuals": residuals.values,
    })
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 시간에 따른 잔차 - Seaborn scatterplot 사용
    sns.scatterplot(data=df_residuals, x="date", y="residuals", 
                   alpha=0.7, s=50, color=sns.color_palette("husl", 8)[2], ax=axes[0])
    axes[0].axhline(y=0, color="red", linestyle="--", linewidth=2, alpha=0.8)
    axes[0].set_xlabel("Date", fontsize=11)
    axes[0].set_ylabel("Residuals", fontsize=11)
    axes[0].set_title("Residuals Over Time", fontsize=12, fontweight="bold")
    
    # 잔차 히스토그램 - Seaborn histplot 사용
    sns.histplot(data=df_residuals, x="residuals", bins=20, 
                kde=True, color=sns.color_palette("husl", 8)[3], 
                edgecolor="black", alpha=0.7, ax=axes[1])
    axes[1].axvline(x=0, color="red", linestyle="--", linewidth=2, alpha=0.8)
    axes[1].set_xlabel("Residuals", fontsize=11)
    axes[1].set_ylabel("Frequency", fontsize=11)
    axes[1].set_title("Residuals Distribution", fontsize=12, fontweight="bold")
    
    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_coverage_calibration(
    actual: pd.Series,
    bundle: PredictionBundle,
    title: str = "Coverage Calibration",
    save_path: Optional[Path] = None,
) -> None:
    """Coverage calibration 플롯 (예측 구간이 실제로 얼마나 커버하는지)."""
    if not _VISUALIZATION_AVAILABLE:
        return
    if not bundle.quantiles:
        return
    
    quantiles_sorted = sorted(bundle.quantiles.keys())
    coverage_levels = []
    empirical_coverages = []
    
    for i in range(len(quantiles_sorted) // 2):
        lower_idx = i
        upper_idx = len(quantiles_sorted) - 1 - i
        if lower_idx >= upper_idx:
            break
        
        lower_q = quantiles_sorted[lower_idx]
        upper_q = quantiles_sorted[upper_idx]
        expected_coverage = upper_q - lower_q
        
        lower_series = bundle.quantiles[lower_q]
        upper_series = bundle.quantiles[upper_q]
        
        # 실제 커버리지 계산
        aligned = pd.concat([actual, lower_series, upper_series], axis=1, join="inner")
        if len(aligned) > 0:
            inside = (aligned.iloc[:, 0] >= aligned.iloc[:, 1]) & (aligned.iloc[:, 0] <= aligned.iloc[:, 2])
            empirical_coverage = inside.mean()
            
            coverage_levels.append(expected_coverage)
            empirical_coverages.append(empirical_coverage)
    
    if not coverage_levels:
        return
    
    # 데이터 준비
    df_coverage = pd.DataFrame({
        "Expected Coverage": coverage_levels,
        "Empirical Coverage": empirical_coverages,
        "Level": [f"{int(exp*100)}%" for exp in coverage_levels],
    })
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "r--", linewidth=2.5, label="Perfect Calibration", alpha=0.8)
    
    # Seaborn scatterplot 사용
    sns.scatterplot(data=df_coverage, x="Expected Coverage", y="Empirical Coverage",
                   s=150, alpha=0.8, color=sns.color_palette("husl", 8)[4], 
                   edgecolor="black", linewidth=1.5, zorder=5, ax=ax)
    
    # 레이블 추가
    for _, row in df_coverage.iterrows():
        ax.annotate(row["Level"], (row["Expected Coverage"], row["Empirical Coverage"]),
                   xytext=(8, 8), textcoords="offset points", fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    ax.set_xlabel("Expected Coverage", fontsize=12)
    ax.set_ylabel("Empirical Coverage", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best", frameon=True, fancybox=True, shadow=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    metric_name: str = "crps",
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> None:
    """여러 실험의 메트릭을 비교하는 막대 그래프."""
    if not _VISUALIZATION_AVAILABLE:
        return
    if not metrics_dict:
        return
    
    model_names = list(metrics_dict.keys())
    values = [metrics_dict[name].get(metric_name, np.nan) for name in model_names]
    
    # NaN 제거
    valid_data = [(name, val) for name, val in zip(model_names, values) if not np.isnan(val)]
    if not valid_data:
        return
    
    model_names_clean, values_clean = zip(*valid_data)
    
    # 데이터 준비
    df_metrics = pd.DataFrame({
        "Model": model_names_clean,
        metric_name.upper(): values_clean,
    })
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Seaborn barplot 사용
    sns.barplot(data=df_metrics, x="Model", y=metric_name.upper(),
               palette="husl", alpha=0.8, edgecolor="black", linewidth=1.5, ax=ax)
    
    # 값 표시
    for i, val in enumerate(values_clean):
        ax.text(i, val, f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel(metric_name.upper(), fontsize=12)
    ax.set_title(title or f"{metric_name.upper()} Comparison", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_rolling_metrics(
    predictions_df: pd.DataFrame,
    metrics_list: list[str],
    title: str = "Rolling Metrics Over Time",
    save_path: Optional[Path] = None,
) -> None:
    """단일 실험의 rolling forecast에 대한 MAE 추이를 시각화."""
    if not _VISUALIZATION_AVAILABLE:
        return
    if "as_of" not in predictions_df.columns:
        return
    
    # 간단한 rolling metrics 계산 (예: MAE per as_of)
    if "actual" in predictions_df.columns and "prediction" in predictions_df.columns:
        rolling_mae = predictions_df.groupby("as_of").apply(
            lambda x: np.abs(x["actual"] - x["prediction"]).mean()
        )
        
        # 데이터 준비
        df_rolling = pd.DataFrame({
            "As Of Date": pd.to_datetime(rolling_mae.index),
            "MAE": rolling_mae.values,
        })
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Seaborn lineplot 사용
        sns.lineplot(data=df_rolling, x="As Of Date", y="MAE",
                    marker="o", linewidth=2.5, markersize=8,
                    color=sns.color_palette("husl", 8)[5], ax=ax)
        
        ax.set_xlabel("As Of Date", fontsize=12)
        ax.set_ylabel("MAE", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def compute_rolling_mae(predictions_df: pd.DataFrame) -> pd.Series:
    """as_of별 MAE 시계열을 계산하는 유틸리티.

    - 입력: rolling forecast 모드에서 생성된 predictions.csv (as_of, actual, prediction 포함)
    - 출력: index=as_of(pd.Timestamp), value=MAE
    """
    if "as_of" not in predictions_df.columns:
        raise ValueError("compute_rolling_mae는 as_of 컬럼이 있는 rolling 예측 결과에만 사용할 수 있습니다.")
    if not {"actual", "prediction"}.issubset(predictions_df.columns):
        raise ValueError("predictions_df에는 actual, prediction 컬럼이 필요합니다.")

    df = predictions_df.copy()
    df["abs_error"] = (df["actual"] - df["prediction"]).abs()
    rolling_mae = df.groupby("as_of")["abs_error"].mean()
    # as_of를 시간축으로 정렬
    rolling_mae.index = pd.to_datetime(rolling_mae.index)
    rolling_mae = rolling_mae.sort_index()
    return rolling_mae


def plot_multi_model_rolling_metric(
    rolling_series_dict: Mapping[str, pd.Series],
    *,
    metric_name: str = "MAE",
    title: str = "Rolling Performance Over Time",
    save_path: Optional[Path] = None,
) -> None:
    """여러 모델의 rolling 메트릭 곡선을 한 플롯에서 비교.

    Parameters
    ----------
    rolling_series_dict:
        key=모델/실험 이름, value=as_of 인덱스를 가진 시리즈 (예: compute_rolling_mae 결과)
    metric_name:
        y축에 표시할 메트릭 이름 (예: \"MAE\", \"CRPS\" 등)
    """
    if not _VISUALIZATION_AVAILABLE:
        return
    if not rolling_series_dict:
        return

    # Long-format DataFrame 생성
    records = []
    for model_name, series in rolling_series_dict.items():
        if series is None or series.empty:
            continue
        s = series.sort_index()
        for ts, val in s.items():
            records.append(
                {
                    "As Of Date": pd.to_datetime(ts),
                    metric_name: float(val),
                    "Model": model_name,
                }
            )
    if not records:
        return

    df_plot = pd.DataFrame.from_records(records)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.lineplot(
        data=df_plot,
        x="As Of Date",
        y=metric_name,
        hue="Model",
        linewidth=2.0,
        ax=ax,
    )

    ax.set_xlabel("As Of Date", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Model", loc="best", frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def create_all_visualizations(
    actual: pd.Series,
    bundle: PredictionBundle,
    predictions_df: pd.DataFrame,
    metrics: Dict[str, float],
    experiment_name: str,
    output_dir: Path,
    y_max: Optional[float] = 50.0,
    show_title: bool = False,
) -> None:
    """모든 시각화를 생성하고 저장.
    
    Parameters
    ----------
    y_max : float, optional
        Y축 상한. 기본값 50 (시즌 패턴이 더 잘 보임)
    show_title : bool
        제목 표시 여부. 논문에서는 False (figure caption 사용)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Forecast vs Actual - 논문 스타일
    plot_forecast_vs_actual(
        actual,
        bundle,
        title=f"{experiment_name} - Forecast vs Actual",
        save_path=output_dir / "forecast_vs_actual.png",
        y_max=y_max,
        show_title=show_title,
    )
    
    # 2. Residuals
    plot_residuals(
        actual,
        bundle,
        title=f"{experiment_name} - Residuals",
        save_path=output_dir / "residuals.png",
    )
    
    # 3. Coverage Calibration
    if bundle.quantiles:
        plot_coverage_calibration(
            actual,
            bundle,
            title=f"{experiment_name} - Coverage Calibration",
            save_path=output_dir / "coverage_calibration.png",
        )
    
    # 4. Rolling Metrics (rolling forecast인 경우)
    if "as_of" in predictions_df.columns:
        plot_rolling_metrics(
            predictions_df,
            list(metrics.keys()),
            title=f"{experiment_name} - Rolling Metrics",
            save_path=output_dir / "rolling_metrics.png",
        )


__all__ = [
    "plot_forecast_vs_actual",
    "plot_residuals",
    "plot_coverage_calibration",
    "plot_metrics_comparison",
    "plot_rolling_metrics",
    "compute_rolling_mae",
    "plot_multi_model_rolling_metric",
    "create_all_visualizations",
]

