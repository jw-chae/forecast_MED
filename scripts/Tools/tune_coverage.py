from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

from run_sim_wrapper import run_sim, SimConfig
from adapters import load_his_outpatient_series
from evidence_pack import build_evidence_pack_from_gov_monthly_csv
from scenario_engine import extract_growth_episodes, generate_paths_conditional
from evt import fit_pot, replace_tail_with_evt
from metrics import interval_coverage, crps_gaussian, smape, mae, peak_metrics


def objective(metrics: Dict[str, Any], target_low: float = 0.90, target_high: float = 0.98) -> float:
    cov = metrics.get("coverage95")
    crps = metrics.get("crps", 1e6)
    mae = metrics.get("mae_median", 1e6)
    recall = metrics.get("recall_pm2w", 1.0)
    if cov is None:
        return 1e9
    # penalty if outside band; prefer center of band ~0.94
    mid = 0.94
    pen = 0.0
    if cov < target_low:
        pen = (target_low - cov) * 200.0
    elif cov > target_high:
        pen = (cov - target_high) * 100.0
    else:
        pen = abs(cov - mid) * 50.0
    rec_pen = 0.0 if recall >= 0.5 else (0.5 - float(recall)) * 100.0
    return float(pen + rec_pen + 0.1 * crps + 0.02 * mae)


def _eval_chain_metrics(disease: str, train_until: str, k: int, params: Dict[str, Any], gov_monthly_csv: str, chain_particles: int = 500, posthoc_cov: float = 0.95) -> Dict[str, Any]:
    base = Path(__file__).resolve().parents[2]
    csv_path = base / "processed_data" / "his_outpatient_weekly_epi_counts.csv"
    dt_index, series_all = load_his_outpatient_series(str(csv_path), disease)
    dates_all = [d.strftime("%Y-%m-%d") for d in dt_index]
    split_idx = max(i for i, d in enumerate(dates_all) if d <= train_until)
    hist = series_all[: split_idx + 1].astype(float)
    ev = build_evidence_pack_from_gov_monthly_csv(gov_monthly_csv, asof=train_until, future_weeks=k, future_decay=[1.0,0.95,0.9,0.85,0.8,0.7,0.6,0.5][:k])
    nv = (ev.get("external_signals", {}) or {}).get("news_signal_weekly") or []
    news_vec = [float(nv[i] if i < len(nv) else (nv[-1] if nv else params.get("news_signal", 0.1))) for i in range(k)]
    n_particles = max(100, int(chain_particles))
    y = hist.copy()
    q05_list: List[float] = []
    q50_list: List[float] = []
    q95_list: List[float] = []
    for step in range(k):
        episodes = extract_growth_episodes(y)
        season_start_override = float(np.median(y[-8:]))
        paths = generate_paths_conditional(
            series=y,
            horizon=1,
            n_paths=n_particles,
            episodes=episodes,
            news_signal=float(news_vec[step]),
            quality=float(params.get("quality", 0.68)),
            recent_baseline_window=int(params.get("recent_baseline_window", 8)),
            amplitude_quantile=float(params.get("amplitude_quantile", 0.9)),
            amplitude_multiplier=float(params.get("amplitude_multiplier", 2.2)),
            ratio_cap_quantile=float(params.get("ratio_cap_quantile", 0.99)),
            warmup_weeks=0,
            use_delta_quantile=bool(params.get("use_delta_quantile", True)),
            delta_quantile=float(params.get("delta_quantile", 0.05)),
            nb_dispersion_k=(None if params.get("nb_dispersion_k") is None else float(params.get("nb_dispersion_k"))),
            start_value_override=float(params.get("start_value_override", season_start_override)),
            r_boost_cap=float(params.get("r_boost_cap", 2.0)),
            scale_cap=float(params.get("scale_cap", 1.6)),
            x_cap_multiplier=float(params.get("x_cap_multiplier", 2.0)),
        )
        u = float(np.quantile(y, float(params.get("evt_u_quantile", 0.9))))
        gpd = fit_pot(y, threshold=u, min_excess=int(params.get("min_excess", 5)))
        paths = replace_tail_with_evt(paths, gpd_params=gpd, threshold=u)
        term = paths[:, -1]
        q05_list.append(float(np.quantile(term, 0.05)))
        q50_list.append(float(np.quantile(term, 0.50)))
        q95_list.append(float(np.quantile(term, 0.95)))
        y = np.append(y, float(np.median(term)))
    dates = dates_all
    split_idx2 = max(i for i, d in enumerate(dates) if d <= train_until)
    target = series_all.astype(float)[split_idx2 + 1 : split_idx2 + 1 + k]
    q05 = np.array(q05_list); q50 = np.array(q50_list); q95 = np.array(q95_list)
    if posthoc_cov and len(target) == k and k > 0:
        desired = float(max(0.5, min(0.98, posthoc_cov)))
        def cov_with_scale(s: float) -> float:
            lo = q50 - s * (q50 - q05)
            hi = q50 + s * (q95 - q50)
            return float(np.mean((target >= lo) & (target <= hi)))
        cov0 = cov_with_scale(1.0)
        if cov0 < desired:
            s_lo, s_hi = 1.0, 10.0
            for _ in range(16):
                s_mid = 0.5 * (s_lo + s_hi)
                if cov_with_scale(s_mid) < desired: s_lo = s_mid
                else: s_hi = s_mid
            s_star = s_hi
            q05 = q50 - s_star * (q50 - q05)
            q95 = q50 + s_star * (q95 - q50)
    lower = np.minimum(q05, q95); upper = np.maximum(q05, q95)
    met = {
        "mae_median": float(mae(target, q50, use_median=True)),
        "smape": float(smape(target, q50)),
        "coverage95": float(interval_coverage(target, lower, upper)),
        "crps": float(crps_gaussian(target, q50, np.maximum(1e-6, (q95 - q05) / 3.92))),
    }
    met.update(peak_metrics(target, q50, alpha_top=float(params.get("peak_alpha", 0.1)), window_recall=2))
    return met


def tune(disease: str, train_until: str, end: str, n_trials: int = 0, n_paths: int = 800, gov_monthly_csv: str = "", mode: str = "base", chain_k: int = 12, chain_particles: int = 500, posthoc_cov: float = 0.95) -> Dict[str, Any]:
    cfg = SimConfig(disease=disease, train_until=train_until, end=end, season_profile="flu")
    base_params = {
        "amplitude_quantile": 0.9,
        "amplitude_multiplier": 1.8,
        "ratio_cap_quantile": 0.98,
        "warmup_weeks": 1,
        "use_delta_quantile": True,
        "delta_quantile": 0.05,
        "nb_dispersion_k": 10.0,
        "r_boost_cap": 2.0,
        "scale_cap": 1.6,
        "x_cap_multiplier": 2.0,
        "evt_u_quantile": 0.9,
        "n_paths": int(n_paths),
    }
    rng = np.random.default_rng(123)
    best: Dict[str, Any] = {"score": 1e18}
    # optional evidence from gov monthly CSV to drive weekly news signal during tuning
    evidence: Dict[str, Any] = {}
    if gov_monthly_csv:
        try:
            base = Path(__file__).resolve().parents[2]
            csv_path = base / "processed_data" / "his_outpatient_weekly_epi_counts.csv"
            dt_index, _series = load_his_outpatient_series(str(csv_path), disease)
            dates = [d.strftime("%Y-%m-%d") for d in dt_index]
            # compute horizon (weeks between train_until and end)
            try:
                split_idx = max(i for i, d in enumerate(dates) if d <= train_until)
                end_idx = max(i for i, d in enumerate(dates) if d <= end)
            except Exception:
                split_idx = 0; end_idx = 0
            horizon = max(1, end_idx - split_idx)
            evidence = build_evidence_pack_from_gov_monthly_csv(
                gov_monthly_csv, asof=train_until, future_weeks=horizon,
                future_decay=[1.0,0.95,0.9,0.85,0.8,0.7,0.6,0.5][:horizon]
            )
        except Exception:
            evidence = {}

    if n_trials and n_trials > 0:
        # random search
        for _ in range(int(n_trials)):
            p = dict(base_params)
            if mode == "chain":
                p["quality"] = float(rng.uniform(0.62, 0.75))
                p["amplitude_multiplier"] = float(rng.uniform(2.0, 2.4))
                p["ratio_cap_quantile"] = float(rng.uniform(0.985, 0.995))
                res = _eval_chain_metrics(disease, train_until, chain_k, p, gov_monthly_csv, chain_particles=chain_particles, posthoc_cov=posthoc_cov)
                score = objective(res)
            else:
                p["quality"] = float(rng.uniform(0.5, 0.95))
                p["amplitude_multiplier"] = float(rng.uniform(1.2, 2.8))
                p["ratio_cap_quantile"] = float(rng.uniform(0.95, 0.999))
                if evidence:
                    p["evidence"] = evidence
                res = run_sim(p, cfg)
                score = objective(res.get("metrics", {}))
            if score < best["score"]:
                best = {"params": p, "metrics": (res.get("metrics", {}) if isinstance(res, dict) and "metrics" in res else res), "score": score}
    else:
        if mode == "chain":
            qualities = np.linspace(0.62, 0.75, 6)
            amps = np.linspace(2.0, 2.4, 5)
            ratio_qs = np.linspace(0.985, 0.995, 6)
            for q in qualities:
                for am in amps:
                    for rq in ratio_qs:
                        p = dict(base_params)
                        p["quality"] = float(q)
                        p["amplitude_multiplier"] = float(am)
                        p["ratio_cap_quantile"] = float(rq)
                        res = _eval_chain_metrics(disease, train_until, chain_k, p, gov_monthly_csv, chain_particles=chain_particles, posthoc_cov=posthoc_cov)
                        score = objective(res)
                        if score < best["score"]:
                            best = {"params": p, "metrics": res, "score": score}
        else:
            qualities = np.linspace(0.5, 0.9, 7)
            amps = np.linspace(1.2, 2.6, 6)
            ratio_qs = np.linspace(0.96, 0.995, 6)
            for q in qualities:
                for am in amps:
                    for rq in ratio_qs:
                        p = dict(base_params)
                        p["quality"] = float(q)
                        p["amplitude_multiplier"] = float(am)
                        p["ratio_cap_quantile"] = float(rq)
                        if evidence:
                            p["evidence"] = evidence
                        res = run_sim(p, cfg)
                        score = objective(res.get("metrics", {}))
                        if score < best["score"]:
                            best = {"params": p, "metrics": res.get("metrics", {}), "score": score}
    return best


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--disease", default="手足口病")
    ap.add_argument("--train_until", default="2022-12-31")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--n_trials", type=int, default=80)
    ap.add_argument("--n_paths", type=int, default=800)
    ap.add_argument("--gov_monthly_csv", default="", help="정부 월간 CSV를 주별 신호로 사용")
    ap.add_argument("--mode", default="base", choices=["base","chain"], help="튜닝 모드")
    ap.add_argument("--chain", type=int, default=12)
    ap.add_argument("--chain_particles", type=int, default=500)
    ap.add_argument("--posthoc_cov", type=float, default=0.95)
    args = ap.parse_args()

    base = Path(__file__).resolve().parents[2]
    out_dir = base / "reports" / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)

    best = tune(
        args.disease, args.train_until, args.end,
        n_trials=args.n_trials, n_paths=args.n_paths,
        gov_monthly_csv=args.gov_monthly_csv,
        mode=args.mode, chain_k=args.chain,
        chain_particles=args.chain_particles, posthoc_cov=args.posthoc_cov,
    )
    out = out_dir / f"tuned_{args.disease}.json"
    out.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(out))


if __name__ == "__main__":
    main()


