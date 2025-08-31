from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class Episode:
    start_idx: int
    end_idx: int
    values: np.ndarray

    def shape_factors(self) -> np.ndarray:
        if len(self.values) < 2:
            return np.array([1.0])
        base = max(1e-9, float(self.values[0]))
        return np.asarray(self.values, dtype=float) / base


def extract_growth_episodes(
    series: np.ndarray,
    pct_threshold: float = 0.12,
    min_len: int = 2,
    relax_drop: float = -0.05,
) -> List[Episode]:
    y = np.asarray(series, dtype=float)
    if y.ndim != 1 or y.size < 3:
        return []

    growth = (y[1:] - y[:-1]) / np.maximum(1.0, y[:-1])
    episodes: List[Episode] = []
    i = 0
    while i < len(growth):
        if growth[i] >= pct_threshold:
            start = i
            j = i + 1
            while j < len(growth) and growth[j] > relax_drop:
                j += 1
            s = start
            e = j
            if e - s + 1 >= min_len:
                vals = y[s : e + 1]
                episodes.append(Episode(start_idx=s, end_idx=e, values=vals))
            i = j + 1
        else:
            i += 1
    return episodes


def _episode_similarity_score(episode: Episode, recent_deltas: np.ndarray) -> float:
    shape = episode.shape_factors()
    if len(shape) < 2:
        return 1e9
    deltas = np.diff(shape)
    m = min(len(deltas), len(recent_deltas))
    if m == 0:
        return 1e9
    return float(np.linalg.norm(deltas[:m] - recent_deltas[:m]))


def generate_paths_conditional(
    series: np.ndarray,
    horizon: int,
    n_paths: int,
    episodes: Optional[List[Episode]] = None,
    recent_window: int = 3,
    news_signal: float | np.ndarray = 0.0,
    quality: float = 1.0,
    random_state: Optional[int] = None,
    recent_baseline_window: int = 8,
    amplitude_quantile: float = 0.9,
    amplitude_multiplier: float = 1.8,
    ratio_cap_quantile: float = 0.98,
    warmup_weeks: int = 1,
    start_value_override: Optional[float] = None,
    use_delta_quantile: bool = True,
    delta_quantile: float = 0.05,
    nb_dispersion_k: Optional[float] = 8.0,
    # 안정화 캡 파라미터(가드레일)
    r_boost_cap: float = 2.0,
    scale_cap: float = 1.6,
    x_cap_multiplier: float = 2.0,
    seir_infection_curve: Optional[np.ndarray] = None,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    y = np.asarray(series, dtype=float)
    y0 = float(y[-1])

    if episodes is None:
        episodes = extract_growth_episodes(y)

    if not episodes:
        # 하락/안정 시나리오: 에피소드가 없으면 최근 추세 기반으로 단순 노이즈 경로 생성
        w = int(max(1, min(recent_baseline_window, len(y))))
        recent_mean = float(np.mean(y[-w:]))
        recent_std = float(np.std(y[-w:])) if len(y[-w:]) > 1 else 0.0
        # 품질 기반 노이즈 스케일 (하락 시에는 변동성을 더 줄임)
        noise_scale = max(0.05, 0.15 * (1.0 - quality))
        
        # 최근 4주 성장률 평균을 기반으로 완만한 하락/안정 추세 반영
        growth_w = min(4, len(y) -1)
        if growth_w > 0:
            recent_growth = (y[-1] - y[-(growth_w+1)]) / growth_w / np.maximum(1.0, y[-(growth_w+1)])
        else:
            recent_growth = 0.0
        
        drift = np.clip(recent_growth, -0.15, 0.05) # 하락폭 제한

        paths = np.zeros((n_paths, horizon))
        start_val = max(y0, 0.5 * recent_mean) # 시작값이 0으로 붕괴하는 것 방지
        
        for n in range(n_paths):
            path_n = [start_val]
            for h in range(horizon):
                noise = rng.normal(loc=0.0, scale=noise_scale)
                next_val = path_n[-1] * (1 + drift) + noise * recent_std
                path_n.append(max(0, next_val))
            paths[n, :] = path_n[1:]

        return paths

    tail = y[-(recent_window + 1) :]
    if len(tail) < recent_window + 1:
        recent_deltas = np.array([0.0])
    else:
        rec_shape = tail / max(1e-9, tail[0])
        recent_deltas = np.diff(rec_shape)

    scores = np.array([_episode_similarity_score(ep, recent_deltas) for ep in episodes])
    inv = 1.0 / np.maximum(1e-6, scores)
    probs = inv / inv.sum()

    if isinstance(news_signal, np.ndarray):
        s_vec = np.clip(news_signal.astype(float), 0.0, 1.0)
    else:
        s_vec = None
    base_scale = 1.0 + 1.0 * (float(news_signal) if not isinstance(news_signal, np.ndarray) else 0.0)
    # 품질이 낮을수록 변동성 확대
    noise_scale = 0.08 + 0.22 * (1.0 - float(np.clip(quality, 0.0, 1.0)))

    # 최근 기준선(학습 마지막 값이 0에 가까운 문제 완화)
    w = int(max(1, min(recent_baseline_window, len(y))))
    baseline = float(np.median(y[-w:]))
    if baseline <= 0.0:
        baseline = float(np.mean(y[-w:])) if np.any(y[-w:] > 0) else 1.0

    # 에피소드 진폭 분포에서 상위 분위로 스케일 강화(보수성 완화)
    amp_list = []
    for ep in episodes:
        f = ep.shape_factors()
        if f.size > 0:
            amp_list.append(float(np.max(f)))
    amp_scale = float(np.quantile(amp_list, amplitude_quantile)) if amp_list else 1.0
    amp_scale *= float(max(1.0, amplitude_multiplier))

    # 전주 비율의 전역 상한(폭주 방지) - 원시 시계열에서 경험적 상한 추정
    if use_delta_quantile:
        delta_floor = float(np.quantile(y[y>0], delta_quantile)) if np.any(y>0) else 1.0
    else:
        delta_floor = 1e-6
    ratios_all: list[float] = []
    for t in range(1, len(y)):
        denom = max(delta_floor, float(y[t-1]))
        ratios_all.append(float(y[t] / denom))
    r_cap_global = float(np.quantile(ratios_all, ratio_cap_quantile)) if ratios_all else 4.0

    paths = np.zeros((n_paths, horizon), dtype=float)
    for n in range(n_paths):
        ep = rng.choice(episodes, p=probs)
        factors = ep.shape_factors()
        x_old = np.linspace(0.0, 1.0, num=len(factors))
        x_new = np.linspace(0.0, 1.0, num=horizon + 1)
        interp = np.interp(x_new, x_old, factors, left=factors[0], right=factors[-1])
        # 전주 대비 성장률 비율 r_h = f(h)/f(h-1)
        denom_vec = np.maximum(delta_floor, interp[:-1]) if use_delta_quantile else np.maximum(1e-9, interp[:-1])
        ratios = interp[1:] / denom_vec
        ratios = np.clip(ratios, 0.5, r_cap_global)
        ratios_boost = 1.0 + (ratios - 1.0) * amp_scale
        # 방어: 하한을 0이 아닌 0.6으로 클립하여 r_eff가 0으로 붕괴하는 상황 방지
        ratios_boost = np.clip(ratios_boost, 0.6, float(r_boost_cap))

        # --- SEIR Hybrid Logic ---
        if seir_infection_curve is not None and seir_infection_curve.size >= horizon:
            seir_curve_h = seir_infection_curve[:horizon]
            # Normalize to get ratios
            seir_ratios = seir_curve_h[1:] / np.maximum(1e-9, seir_curve_h[:-1])
            seir_ratios = np.insert(seir_ratios, 0, seir_ratios[0]) # Pad first element
            
            # Blend episode-based ratios with SEIR-based ratios
            # Simple average blending for now, could be a parameter later
            ratios_boost = (ratios_boost + seir_ratios) / 2.0
            ratios_boost = np.clip(ratios_boost, 0.6, float(r_boost_cap))
        # --- End SEIR Hybrid Logic ---

        # 스케일(뉴스) 벡터 (상한 적용)
        if s_vec is not None and len(s_vec) >= horizon:
            scale_vec = 1.0 + 1.0 * s_vec[:horizon]
        else:
            scale_vec = np.full(horizon, base_scale, dtype=float)
        # 안전 상한
        scale_vec = np.clip(scale_vec, 0.0, float(scale_cap))

        # 시작값: 직전 주 값 y0 (사용자 요구: 그때의 기준으로 시작)
        x0 = float(max(1e-6, y0))
        if start_value_override is not None:
            x0 = float(max(x0, start_value_override))
        x = x0
        # x 상한: 과거 99% 분위 × 배수
        try:
            x_cap_value = float(np.quantile(y, 0.99)) * float(max(1.0, x_cap_multiplier))
            if not np.isfinite(x_cap_value) or x_cap_value <= 0:
                x_cap_value = 1e6
        except Exception:
            x_cap_value = 1e6
        for h in range(horizon):
            # 워밍업: 첫 k주 동안 급격한 점프 완화
            gamma = 1.0 if h >= warmup_weeks else (0.3 + 0.7 * (h + 1) / max(1, warmup_weeks))
            r_eff = 1.0 + gamma * (ratios_boost[h] - 1.0)
            # 로그정규 단계 노이즈(클리핑)
            step_noise = float(np.clip(rng.normal(loc=0.0, scale=noise_scale), -3.0 * noise_scale, 3.0 * noise_scale))
            x = x * r_eff * scale_vec[h] * float(np.exp(step_noise))
            x = float(np.clip(x, 0.0, x_cap_value))
            if nb_dispersion_k is not None and nb_dispersion_k > 0:
                # NB 샘플링: Gamma-Poisson 혼합으로 과산포 반영
                lam = float(np.clip(x, 0.0, 1e6))
                k = float(nb_dispersion_k)
                theta = lam / k if k > 0 else max(lam, 1.0)
                lam_sample = float(np.clip(rng.gamma(shape=k, scale=theta) if k > 0 else lam, 0.0, 1e6))
                cnt = rng.poisson(lam=lam_sample)
                paths[n, h] = float(cnt)
            else:
                paths[n, h] = max(0.0, x)

    return paths

