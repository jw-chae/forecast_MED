from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from .visualization import compute_rolling_mae, plot_multi_model_rolling_metric


def _load_predictions(run_dir: Path) -> pd.DataFrame:
    preds_path = run_dir / "predictions.csv"
    if not preds_path.exists():
        raise FileNotFoundError(f"predictions.csv not found in {run_dir}")
    return pd.read_csv(preds_path)


def collect_rolling_series(paths: Iterable[Path]) -> Dict[str, pd.Series]:
    """주어진 결과 디렉토리들에서 rolling MAE 시리즈를 수집."""
    series_dict: Dict[str, pd.Series] = {}
    for path in paths:
        if not path.is_dir():
            continue
        preds = _load_predictions(path)
        if "as_of" not in preds.columns:
            # rolling forecast가 아닌 실험은 건너뜀
            continue
        try:
            s = compute_rolling_mae(preds)
        except Exception:
            continue
        series_dict[path.name] = s
    return series_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="여러 rolling 실험의 MAE 추이를 한 그래프에서 비교하는 도구.",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        required=True,
        help="결과 디렉토리 경로 또는 glob 패턴 (예: experiments/results/baseline_bundle/*_v3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="저장할 PNG 파일 경로",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Rolling MAE Comparison",
        help="플롯 제목",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    paths: List[Path] = []
    for pattern in args.experiments:
        matched = [Path(p) for p in glob.glob(pattern)]
        if not matched and Path(pattern).is_dir():
            matched = [Path(pattern)]
        paths.extend(m for m in matched if m.is_dir())

    if not paths:
        raise SystemExit("No experiment result directories found for the given patterns.")

    series_dict = collect_rolling_series(paths)
    if not series_dict:
        raise SystemExit("No rolling predictions (with as_of) found in the provided directories.")

    output_path = Path(args.output)
    plot_multi_model_rolling_metric(
        series_dict,
        metric_name="MAE",
        title=args.title,
        save_path=output_path,
    )
    print(f"Rolling comparison plot saved to {output_path}")


if __name__ == "__main__":
    main()


