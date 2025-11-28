from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# matplotlib 백엔드를 가장 먼저 설정 (timesfm import 전에)
os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd

from .config_validator import ConfigValidationError, validate_config
from .data_loader import DatasetSplit, create_splits, generate_rolling_windows
from .logger import setup_logging
from .metrics import PredictionBundle, evaluate_metrics
from .models import BaseModel, get_model_class
from .models.base_model import ModelOutput
from .utils import (
    apply_overrides,
    generate_run_directory,
    load_yaml,
    save_json,
    save_yaml,
    set_reproducibility,
)
from .visualization import create_all_visualizations

try:  # pragma: no cover - optional dependency
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


@dataclass
class ExperimentContext:
    config: Dict[str, object]
    run_dir: Path
    checkpoint_dir: Path
    logger_name: str


class ExperimentRunner:
    def __init__(
        self,
        *,
        config_path: Path,
        overrides: Iterable[str] | None = None,
        resume_dir: Optional[Path] = None,
        debug: bool = False,
        batch_name: Optional[str] = None,
    ) -> None:
        self.config_path = config_path
        self.base_dir = Path(__file__).resolve().parents[1]
        self.overrides = list(overrides or [])
        self.resume_dir = resume_dir
        self.debug = debug
        self.batch_name = batch_name
        self.logger = None
        self.ctx: Optional[ExperimentContext] = None

    def _load_config(self) -> Dict[str, object]:
        if self.resume_dir:
            cfg_path = self.resume_dir / "config.yaml"
            config = load_yaml(cfg_path)
        else:
            config = load_yaml(self.config_path)
        if self.overrides and not self.resume_dir:
            config = apply_overrides(config, self.overrides)
        return config

    def _prepare_context(self, config: Dict[str, object]) -> ExperimentContext:
        if self.resume_dir:
            run_dir = self.resume_dir
        else:
            results_dir = self.base_dir / "experiments" / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            # batch_name 우선순위: 명령줄 인자 > config > None
            batch_name = self.batch_name or config.get("experiment", {}).get("batch_name") or config.get("batch_name")
            run_dir = generate_run_directory(results_dir, str(config["experiment"]["name"]), batch_name=batch_name)
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger_name = f"experiments.{run_dir.name}"
        return ExperimentContext(config=config, run_dir=run_dir, checkpoint_dir=checkpoint_dir, logger_name=logger_name)

    def setup(self) -> None:
        config = self._load_config()
        config_source_path = self.resume_dir / "config.yaml" if self.resume_dir else self.config_path
        try:
            validate_config(config, config_path=str(config_source_path), project_root=self.base_dir)
        except ConfigValidationError as exc:
            raise SystemExit(f"Configuration error: {exc}") from exc
        self.ctx = self._prepare_context(config)
        logging_cfg = config.get("logging", {})
        self.logger = setup_logging(
            logging_cfg.get("level", "INFO"),
            log_dir=self.ctx.run_dir,
            console=logging_cfg.get("console", True),
            file=logging_cfg.get("file", True),
        )
        self.logger.info("Experiment directory: %s", self.ctx.run_dir)
        if not self.resume_dir:
            save_yaml(self.ctx.run_dir / "config.yaml", config)

    def _instantiate_model(self, config: Dict[str, object]) -> BaseModel:
        model_cfg = dict(config["model"])
        model_type = model_cfg.pop("type")
        model_class = get_model_class(model_type)
        return model_class(model_cfg)

    def _run_single_forecast(self, model: BaseModel, split: DatasetSplit, horizon: int) -> ModelOutput:
        return model.rolling_forecast(split.train, split.test.iloc[:horizon], validation_df=split.validation)

    def _run_rolling_forecast(self, split: DatasetSplit, horizon: int, rolling_cfg: Dict[str, object]) -> pd.DataFrame:
        all_records: List[Dict[str, object]] = []
        forecast_start = rolling_cfg.get("forecast_start") or str(split.test.index[0].date())
        forecast_end = rolling_cfg.get("forecast_end") or str(split.test.index[-1].date())
        windows = generate_rolling_windows(
            pd.concat([split.train, split.test]),
            min_train_weeks=int(rolling_cfg.get("min_train_weeks", len(split.train))),
            step_size=int(rolling_cfg.get("step_size", 1)),
            horizon=horizon,
            start_date=str(split.train.index[0].date()),
            end_date=str(split.test.index[-1].date()),
            forecast_start=forecast_start,
            forecast_end=forecast_end,
        )
        iterator = list(enumerate(windows))
        processed_files = sorted(self.ctx.checkpoint_dir.glob("step_*.pkl")) if self.ctx else []
        processed = len(processed_files)
        all_records: List[Dict[str, object]]
        if processed:
            self.logger.info("Resuming from checkpoint: skipping %d steps", processed)
            with processed_files[-1].open("rb") as f:
                all_records = pickle.load(f)
        else:
            all_records = []
        progress_iter = iterator[processed:]
        progress = tqdm(progress_iter, total=len(progress_iter), desc="Rolling Forecast", leave=True) if tqdm else progress_iter
        scale_logged = False  # Only log scale once to avoid spam
        for idx, (train_slice, test_slice, as_of) in progress:
            model = self._instantiate_model(self.ctx.config)
            model.fit(train_slice)
            output = model.forecast(len(test_slice), start_index=test_slice.index)
            
            # Log scale check on first iteration
            if not scale_logged and hasattr(model, '_log_scale_info'):
                model._log_scale_info(train_slice, output)
                scale_logged = True
            
            bundle = output.bundle
            for date, actual_value in test_slice.iloc[:, 0].items():
                record = {
                    "as_of": as_of.isoformat(),
                    "date": str(date),
                    "actual": float(actual_value),
                    "prediction": float(bundle.point.loc[date]),
                }
                if bundle.quantiles:
                    for q, series in bundle.quantiles.items():
                        record[f"q_{q}"] = float(series.loc[date])
                all_records.append(record)
            checkpoint_path = self.ctx.checkpoint_dir / f"step_{idx:03d}.pkl"
            with checkpoint_path.open("wb") as f:
                pickle.dump(all_records, f)
        return pd.DataFrame(all_records)

    def run(self) -> Dict[str, object]:
        if self.ctx is None or self.logger is None:
            raise RuntimeError("ExperimentRunner must be setup before run()")
        config = self.ctx.config
        set_reproducibility(int(config["experiment"].get("random_seed", 42)))

        split = create_splits(config["data"], base_dir=self.base_dir)
        horizon = int(config.get("model", {}).get("forecast", {}).get("horizon", len(split.test)))

        rolling_cfg = config["data"].get("rolling", {})
        if rolling_cfg.get("enabled"):
            self.logger.info("Running rolling forecast with horizon=%d", horizon)
            predictions_df = self._run_rolling_forecast(split, horizon, rolling_cfg)
        else:
            self.logger.info("Running single forecast horizon=%d", horizon)
            model = self._instantiate_model(config)
            output = self._run_single_forecast(model, split, horizon)
            predictions_df = pd.DataFrame(
                {
                    "date": split.test.index[:horizon].astype(str),
                    "actual": split.test.iloc[:horizon, 0].to_numpy(),
                    "prediction": output.bundle.point.to_numpy(),
                }
            )
            if output.bundle.quantiles:
                for q, series in output.bundle.quantiles.items():
                    predictions_df[f"q_{q}"] = series.to_numpy()

        predictions_path = self.ctx.run_dir / "predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)
        self.logger.info("Predictions saved to %s", predictions_path)

        if "as_of" in predictions_df.columns:
            predictions_df["date"] = pd.to_datetime(predictions_df["date"])
            predictions_df = predictions_df.sort_values(["date", "as_of"])
            latest = predictions_df.groupby("date", sort=True).tail(1).set_index("date")
            actual_series = latest["actual"].copy()
            point_series = latest["prediction"].copy()
            quantile_series = {
                float(col.split("_")[1]): latest[col].copy()
                for col in latest.columns
                if col.startswith("q_")
            }
        else:
            actual_series = split.test.iloc[: len(predictions_df), 0]
            point_series = pd.Series(predictions_df["prediction"].values, index=actual_series.index)
            quantile_series = {
                float(col.split("_")[1]): pd.Series(predictions_df[col].values, index=actual_series.index)
                for col in predictions_df.columns
                if col.startswith("q_")
            }

        bundle = PredictionBundle(point=point_series, quantiles=quantile_series)
        metrics = evaluate_metrics(actual_series, bundle, config["evaluation"]["metrics"])
        metrics_path = self.ctx.run_dir / "metrics.json"
        save_json(metrics_path, metrics)
        self.logger.info("Metrics saved to %s", metrics_path)

        # 시각화 생성
        try:
            experiment_name = str(config["experiment"]["name"])
            model_type = str(config["model"]["type"])
            create_all_visualizations(
                actual=actual_series,
                bundle=bundle,
                predictions_df=predictions_df,
                metrics=metrics,
                experiment_name=f"{experiment_name} ({model_type})",
                output_dir=self.ctx.run_dir / "plots",
            )
            self.logger.info("Visualizations saved to %s", self.ctx.run_dir / "plots")
        except Exception as e:
            self.logger.warning("Failed to create visualizations: %s", e)

        summary = {
            "run_dir": str(self.ctx.run_dir),
            "metrics": metrics,
        }
        (self.ctx.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary


def _expand_config_paths(patterns: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No config files matched pattern '{pattern}'")
        paths.extend(Path(m) for m in matches)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Med-DeepSeek experiments")
    parser.add_argument("--config", nargs="*", help="Path or glob pattern to YAML config")
    parser.add_argument("--override", action="append", default=[], help="Override config values, e.g. model.llm.model=gpt-3.5-turbo")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--resume", type=str, help="Resume from an existing results directory")
    parser.add_argument("--n-runs", type=int, default=1, help="Number of repeated runs for reproducibility check")
    parser.add_argument("--seeds", type=str, help="Comma separated list of seeds")
    parser.add_argument("--parallel", type=int, help="Number of parallel workers (not yet implemented)")
    parser.add_argument("--batch", type=str, help="Batch name for organizing experiments (e.g., 'batch_20241112')")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.parallel:
        print("[WARN] Parallel execution is not implemented yet. Running sequentially.")
    seeds: List[int] = []
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s]
    if args.resume:
        runner = ExperimentRunner(
            config_path=Path(args.config[0]) if args.config else Path(""),
            overrides=args.override,
            resume_dir=Path(args.resume),
            debug=args.debug,
            batch_name=args.batch,
        )
        runner.setup()
        runner.run()
        return
    config_paths = _expand_config_paths(args.config) if args.config else []
    for config_path in config_paths:
        for run_idx in range(args.n_runs):
            overrides = list(args.override)
            if run_idx < len(seeds):
                overrides.append(f"experiment.random_seed={seeds[run_idx]}")
            runner = ExperimentRunner(
                config_path=config_path,
                overrides=overrides,
                debug=args.debug,
                batch_name=args.batch,
            )
            runner.setup()
            summary = runner.run()
            if args.verbose:
                print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
