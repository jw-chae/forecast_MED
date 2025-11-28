from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    import scipy.stats as stats
except Exception:  # pragma: no cover
    stats = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import plotly.express as px
except Exception:  # pragma: no cover
    px = None  # type: ignore

from .visualization import compute_rolling_mae, plot_multi_model_rolling_metric


def _load_metrics(run_dir: Path) -> Dict[str, float]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found in {run_dir}")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _collect_runs(paths: Iterable[Path]) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Results directory not found: {path}")
        try:
            metrics = _load_metrics(path)
        except FileNotFoundError:
            # metrics.json 이 없는 실험은 비교/시각화에서 제외
            continue
        record = {"run": path.name, "run_dir": str(path)}
        record.update({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        records.append(record)
    return pd.DataFrame(records)


def _stat_test(df: pd.DataFrame, metric: str, method: str) -> Optional[float]:
    if stats is None:
        return None
    if metric not in df.columns:
        return None
    values = df[metric].dropna().values
    baseline = df.iloc[0][metric]
    others = values[1:]
    if not len(others):
        return None
    if method == "wilcoxon":
        stat, p = stats.wilcoxon(others - baseline)  # type: ignore[arg-type]
    elif method == "paired_t_test":
        stat, p = stats.ttest_rel(others, baseline)  # type: ignore[arg-type]
    else:
        raise ValueError(f"Unsupported statistical test '{method}'")
    return float(p)


def compare_runs(paths: Iterable[Path], metrics: Iterable[str], statistical_test: Optional[str], alpha: float) -> pd.DataFrame:
    df = _collect_runs(paths)
    df = df.sort_values("run").reset_index(drop=True)
    if statistical_test and len(df) >= 2:
        for metric in metrics:
            p = _stat_test(df[["run", metric]], metric, statistical_test)
            if p is not None:
                df[f"{metric}_p"] = p
                df[f"{metric}_significant"] = p < alpha
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare experiment results")
    parser.add_argument("--exp1", type=str, help="First experiment directory")
    parser.add_argument("--exp2", type=str, help="Second experiment directory")
    parser.add_argument("--experiments", type=str, nargs="*", help="List or glob of experiment directories")
    parser.add_argument("--metrics", type=str, default="crps,mae,coverage_95", help="Comma separated metrics to include")
    parser.add_argument("--output", type=str, help="Output file path (csv or html)")
    parser.add_argument("--plot", action="store_true", help="Generate bar plot for metrics")
    parser.add_argument("--statistical-test", type=str, choices=["wilcoxon", "paired_t_test"], help="Statistical significance test")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    parser.add_argument(
        "--rolling-curve-output",
        type=str,
        help="If set, generates a multi-model rolling MAE curve and saves to the given image path (PNG).",
    )
    parser.add_argument(
        "--rolling-curve-html",
        type=str,
        help="If set, generates an interactive HTML version of the multi-model rolling MAE curve.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()]

    paths: List[Path] = []
    if args.experiments:
        for pattern in args.experiments:
            matched = [Path(p) for p in glob.glob(pattern)]
            if not matched and Path(pattern).is_dir():
                matched = [Path(pattern)]
            paths.extend(m for m in matched if m.is_dir())
    if args.exp1:
        paths.append(Path(args.exp1))
    if args.exp2:
        paths.append(Path(args.exp2))

    if not paths:
        raise SystemExit("No experiment directories provided")

    df = compare_runs(paths, metric_names, args.statistical_test, args.alpha)
    print(df)

    if args.output:
        output_path = Path(args.output)
        if output_path.suffix.lower() == ".csv":
            df.to_csv(output_path, index=False)
        elif output_path.suffix.lower() in {".html", ".htm"}:
            df.to_html(output_path, index=False)
        else:
            raise ValueError("Output format not supported; use .csv or .html")
        print(f"Saved comparison to {output_path}")

    if args.plot and px is not None:
        melted = df.melt(id_vars=["run"], value_vars=metric_names, var_name="metric", value_name="value")
        fig = px.bar(melted, x="run", y="value", color="metric", barmode="group", title="Experiment Comparison")
        fig.show()

    # 멀티 모델 rolling MAE 곡선 (PNG / HTML) 생성 (rolling forecast 결과가 있는 경우에만)
    if args.rolling_curve_output or args.rolling_curve_html:
        rolling_series: Dict[str, pd.Series] = {}
        for path in paths:
            pred_path = path / "predictions.csv"
            if not pred_path.exists():
                continue
            try:
                pred_df = pd.read_csv(pred_path)
            except Exception:
                continue
            if "as_of" not in pred_df.columns:
                # single forecast 실험은 제외
                continue
            try:
                mae_series = compute_rolling_mae(pred_df)
            except Exception:
                continue
            if not mae_series.empty:
                rolling_series[path.name] = mae_series

        if not rolling_series:
            print("No rolling predictions (with 'as_of' column) found; skipping rolling curve plots.")
        else:
            title = "Rolling MAE Over Time"
            # 정적 PNG
            if args.rolling_curve_output:
                output_path = Path(args.rolling_curve_output)
                plot_multi_model_rolling_metric(
                    rolling_series,
                    metric_name="MAE",
                    title=title,
                    save_path=output_path,
                )
                print(f"Saved rolling curve plot to {output_path}")

            # 인터랙티브 HTML (Plotly)
            if args.rolling_curve_html and px is not None:
                html_path = Path(args.rolling_curve_html)
                records = []
                for model_name, series in rolling_series.items():
                    s = series.sort_index()
                    for ts, val in s.items():
                        records.append(
                            {
                                "As Of Date": pd.to_datetime(ts),
                                "MAE": float(val),
                                "Model": model_name,
                            }
                        )
                if records:
                    df_plot = pd.DataFrame.from_records(records)
                    fig = px.line(
                        df_plot,
                        x="As Of Date",
                        y="MAE",
                        color="Model",
                        title=title,
                    )
                    fig.write_html(html_path)
                    print(f"Saved interactive rolling curve HTML to {html_path}")


if __name__ == "__main__":
    main()
