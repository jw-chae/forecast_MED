#!/usr/bin/env python3
"""
Scientific Paper Style Figure Generator
- 600 DPI, 7-9pt fonts, muted colors
- A-B-C composite layout (A 위, B·C 아래 2×2)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================
# Scientific Paper Style Configuration (npj/NeurIPS/Nature)
# ============================================================
DPI = 600
FIGURE_WIDTH_INCH = 6.5   # For composite
FIGURE_HEIGHT_INCH = 5.0

# Font settings (7-9pt for journals)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': DPI,
    'savefig.dpi': DPI,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 2,
    'ytick.major.size': 2,
    'lines.linewidth': 1.0,
    'lines.markersize': 3,
    'grid.linewidth': 0.3,
    'grid.alpha': 0.3,
})

# Scientific color palette (muted, print-friendly)
COLORS = {
    'actual': '#2C3E50',          # Dark blue-gray
    'prediction': '#E74C3C',      # Muted red
    'interval': '#BDC3C7',        # Light gray
    'residual_scatter': '#3498DB', # Muted blue
    'residual_hist': '#7F8C8D',   # Gray
    'zero_line': '#95A5A6',       # Light gray
    'kde': '#2C3E50',             # Dark for KDE line
}


def plot_forecast_ax(ax, actual, prediction, q05=None, q95=None):
    """Plot forecast on given axes."""
    dates = actual.index
    
    # 90% Prediction Interval
    if q05 is not None and q95 is not None:
        ax.fill_between(
            dates, q05.values, q95.values,
            alpha=0.25, color=COLORS['interval'], linewidth=0, label='90% PI',
        )
    
    # Actual values
    ax.plot(dates, actual.values, color=COLORS['actual'], linewidth=1.0, 
            marker='o', markersize=2.5, label='Observed', zorder=3)
    
    # Predictions
    ax.plot(dates, prediction.values, color=COLORS['prediction'], linewidth=1.0,
            linestyle='--', label='Predicted', zorder=2)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Weekly Cases')
    
    # Y-axis: start from 0
    max_val = max(actual.max(), prediction.max())
    if q95 is not None:
        max_val = max(max_val, q95.max())
    ax.set_ylim(bottom=0, top=max_val * 1.1)
    
    # Subtle grid
    ax.grid(True, alpha=0.3, linewidth=0.3)
    ax.set_axisbelow(True)
    
    # Legend - small, no frame
    ax.legend(loc='upper right', frameon=False, fontsize=6)
    
    # Rotate x-axis labels
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_residuals_overtime_ax(ax, actual, prediction):
    """Plot residuals over time on given axes."""
    residuals = actual - prediction
    dates = actual.index
    
    # Scatter plot of residuals
    ax.scatter(dates, residuals.values, c=COLORS['residual_scatter'], 
               s=12, alpha=0.7, edgecolors='none', zorder=3)
    
    # Zero line
    ax.axhline(y=0, color=COLORS['zero_line'], linestyle='-', linewidth=0.8, zorder=1)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Residual')
    
    # Subtle grid
    ax.grid(True, alpha=0.3, linewidth=0.3)
    ax.set_axisbelow(True)
    
    # Rotate x-axis labels
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_residuals_distribution_ax(ax, actual, prediction):
    """Plot residuals distribution on given axes."""
    residuals = (actual - prediction).dropna()
    
    # Histogram
    n, bins, patches = ax.hist(residuals.values, bins=15, color=COLORS['residual_hist'],
                                alpha=0.7, edgecolor='white', linewidth=0.5, density=True)
    
    # KDE overlay
    if len(residuals) > 5:
        kde_x = np.linspace(residuals.min() - 1, residuals.max() + 1, 100)
        try:
            kde = stats.gaussian_kde(residuals.values)
            ax.plot(kde_x, kde(kde_x), color=COLORS['kde'], linewidth=1.0, label='KDE')
        except:
            pass
    
    # Zero line
    ax.axvline(x=0, color=COLORS['zero_line'], linestyle='--', linewidth=0.8, alpha=0.8)
    
    ax.set_xlabel('Residual')
    ax.set_ylabel('Density')
    
    # Subtle grid
    ax.grid(True, alpha=0.3, linewidth=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Stats annotation (small)
    mean_res = residuals.mean()
    std_res = residuals.std()
    ax.text(0.95, 0.95, f'μ={mean_res:.2f}\nσ={std_res:.2f}',
            transform=ax.transAxes, fontsize=6, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none', pad=0.3))


def create_composite_figure(pred_file: Path, model_name: str, output_path: Path):
    """
    Create A-B-C composite figure with improved layout.
    A: Forecast (top, full width)
    B: Residuals over time (bottom left)
    C: Residuals distribution (bottom right)
    """
    df = pd.read_csv(pred_file)
    
    # Find date column
    date_col = 'date' if 'date' in df.columns else 'target_date'
    if date_col not in df.columns:
        print(f"  ⚠️ {model_name}: date column not found")
        return False
    
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    
    if 'actual' not in df.columns or 'prediction' not in df.columns:
        print(f"  ⚠️ {model_name}: actual/prediction columns not found")
        return False
    
    actual = df['actual']
    prediction = df['prediction']
    q05 = df['q_0.05'] if 'q_0.05' in df.columns else None
    q95 = df['q_0.95'] if 'q_0.95' in df.columns else None
    
    # Create figure with gridspec
    fig = plt.figure(figsize=(FIGURE_WIDTH_INCH, FIGURE_HEIGHT_INCH), dpi=DPI)
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[2.0, 1.6],   # A를 더 크게
        hspace=0.45,                # 위–아래 간격
        wspace=0.35,                # B–C 간격
    )
    
    axA = fig.add_subplot(gs[0, :])   # 윗줄 전체
    axB = fig.add_subplot(gs[1, 0])   # 아래 왼쪽
    axC = fig.add_subplot(gs[1, 1])   # 아래 오른쪽
    
    # Plot each panel
    plot_forecast_ax(axA, actual, prediction, q05, q95)
    plot_residuals_overtime_ax(axB, actual, prediction)
    plot_residuals_distribution_ax(axC, actual, prediction)
    
    # Add panel labels
    for ax, label in zip([axA, axB, axC], ['(A)', '(B)', '(C)']):
        ax.text(
            -0.08, 1.05, label,
            transform=ax.transAxes,
            fontsize=10,
            fontweight='bold',
            va='bottom', ha='left'
        )
    
    # Add model name as title
    fig.suptitle(model_name.replace('_', ' '), fontsize=10, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close(fig)
    
    return True


def generate_individual_figures(pred_file: Path, model_name: str, output_dir: Path):
    """Generate individual figures (forecast, residuals_overtime, residuals_distribution)."""
    df = pd.read_csv(pred_file)
    
    date_col = 'date' if 'date' in df.columns else 'target_date'
    if date_col not in df.columns:
        return False
    
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    
    if 'actual' not in df.columns or 'prediction' not in df.columns:
        return False
    
    actual = df['actual']
    prediction = df['prediction']
    q05 = df['q_0.05'] if 'q_0.05' in df.columns else None
    q95 = df['q_0.95'] if 'q_0.95' in df.columns else None
    
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Forecast
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    plot_forecast_ax(ax, actual, prediction, q05, q95)
    plt.tight_layout()
    fig.savefig(model_dir / f"{model_name}_forecast.png", dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    
    # 2. Residuals over time
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    plot_residuals_overtime_ax(ax, actual, prediction)
    plt.tight_layout()
    fig.savefig(model_dir / f"{model_name}_residuals_overtime.png", dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    
    # 3. Residuals distribution
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    plot_residuals_distribution_ax(ax, actual, prediction)
    plt.tight_layout()
    fig.savefig(model_dir / f"{model_name}_residuals_distribution.png", dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    
    return True


def main():
    """Main function to process all models."""
    output_dir = Path("experiments/results/figures_scientific")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Generating Scientific Paper Style Figures (600 DPI)")
    print("=" * 70)
    
    # Define all model paths
    models_to_process = [
        # Li Shui (Hangzhou) LLM models (advanced)
        ("LiShui_Qwen", "experiments/results/advanced_hangzhou_qwen_dashscope/simplified______2024-02-01_2024-09-30_v1/predictions.csv"),
        ("LiShui_OpenAI", "experiments/results/advanced_hangzhou_openai_openai/simplified______2024-02-01_2024-09-30_v1/predictions.csv"),
        ("LiShui_DeepSeek", "experiments/results/advanced_hangzhou_deepseek_deepseek/simplified______2024-02-01_2024-09-30_v1/predictions.csv"),
        
        # Hong Kong LLM models (2023-01 ~ 2024-10)
        ("HK_Qwen", "experiments/results/advanced_hongkong_qwen_dashscope/simplified______2023-01-01_2024-09-30_v1/predictions.csv"),
        ("HK_OpenAI", "experiments/results/advanced_hongkong_openai_openai/simplified______2023-01-01_2024-09-30_v1/predictions.csv"),
        ("HK_DeepSeek", "experiments/results/advanced_hongkong_deepseek_chat_deepseek/simplified______2023-01-01_2024-09-30_v1/predictions.csv"),
        
        # Li Shui (Hangzhou) Baselines
        ("LiShui_ARIMA", "experiments/results/hz_hfmd_baselines/arima_hz_hfmd_v2/predictions.csv"),
        ("LiShui_Prophet", "experiments/results/hz_hfmd_baselines/prophet_hz_hfmd_v2/predictions.csv"),
        ("LiShui_LSTM", "experiments/results/hz_hfmd_baselines/lstm_hz_hfmd_v2/predictions.csv"),
        ("LiShui_XGBoost", "experiments/results/hz_hfmd_baselines/xgboost_hz_hfmd_v2/predictions.csv"),
        ("LiShui_Chronos", "experiments/results/hz_hfmd_baselines/chronos_hz_hfmd_v2/predictions.csv"),
        ("LiShui_Moirai", "experiments/results/hz_hfmd_baselines/moirai_hz_hfmd_v1/predictions.csv"),
        ("LiShui_TimesFM", "experiments/results/hz_hfmd_baselines/timesfm_hz_hfmd_v1/predictions.csv"),
        
        # Hong Kong Baselines (2023-01 ~ 2024-10)
        ("HK_ARIMA", "experiments/results/baseline_hongkong_2023/arima_hk_hfmd_v1/predictions.csv"),
        ("HK_Prophet", "experiments/results/baseline_hongkong_2023/prophet_hk_hfmd_v1/predictions.csv"),
        ("HK_LSTM", "experiments/results/baseline_hongkong_2023/lstm_hk_hfmd_v1/predictions.csv"),
        ("HK_XGBoost", "experiments/results/baseline_hongkong_2023/xgboost_hk_hfmd_v1/predictions.csv"),
        ("HK_Chronos", "experiments/results/baseline_hongkong_2023/chronos_hk_hfmd_v1/predictions.csv"),
        ("HK_Moirai", "experiments/results/baseline_hongkong_2023/moirai_hk_hfmd_v1/predictions.csv"),
        ("HK_TimesFM", "experiments/results/baseline_hongkong_2023/timesfm_hk_hfmd_v1/predictions.csv"),
    ]
    
    print("\n📈 Processing models...\n")
    
    success_count = 0
    for model_name, pred_path in models_to_process:
        pred_file = Path(pred_path)
        if pred_file.exists():
            print(f"  📊 {model_name}")
            
            # Generate individual figures
            if generate_individual_figures(pred_file, model_name, output_dir):
                print(f"     ✓ individual figures")
            
            # Generate composite figure
            model_dir = output_dir / model_name
            if create_composite_figure(pred_file, model_name, model_dir / f"{model_name}_composite.png"):
                print(f"     ✓ composite figure")
                success_count += 1
        else:
            print(f"  ⚠️  {model_name}: file not found")
    
    print(f"\n{'=' * 70}")
    print(f"✅ Generated figures for {success_count}/{len(models_to_process)} models")
    print(f"📁 Output directory: {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
