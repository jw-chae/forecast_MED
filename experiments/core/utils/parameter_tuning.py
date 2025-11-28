from typing import Dict, List, Any, Callable, Tuple, Optional
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from ..epidemic_models import run_seir_simulation
from .metrics import smape


class BayesianParameterTuner:
    """Lightweight Bayesian optimization (Gaussian-process EI) for smooth objectives."""

    def __init__(
        self,
        bounds: Dict[str, Tuple[float, float]],
        *,
        n_initial: int = 5,
        n_iter: int = 15,
        n_candidates: int = 256,
        xi: float = 0.01,
        random_state: Optional[int] = None,
    ) -> None:
        if not bounds:
            raise ValueError("bounds must not be empty")
        self.param_names = list(bounds.keys())
        self.bounds = np.asarray([bounds[k] for k in self.param_names], dtype=float)
        self.n_initial = n_initial
        self.n_iter = n_iter
        self.n_candidates = n_candidates
        self.xi = xi
        self.rng = np.random.default_rng(random_state)

    def _sample(self) -> np.ndarray:
        lows = self.bounds[:, 0]
        highs = self.bounds[:, 1]
        return self.rng.uniform(lows, highs)

    def _expected_improvement(self, mu: np.ndarray, sigma: np.ndarray, best: float) -> np.ndarray:
        sigma = np.maximum(1e-9, sigma)
        improvement = best - mu - self.xi
        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        ei[sigma <= 1e-9] = 0.0
        return np.maximum(0.0, ei)

    def optimize(self, objective: Callable[[np.ndarray], float]) -> Dict[str, Any]:
        X: List[np.ndarray] = []
        y: List[float] = []
        trace: List[Dict[str, float]] = []

        def _observe(x: np.ndarray) -> float:
            val = float(objective(x))
            X.append(x)
            y.append(val)
            trace.append({"value": val, **{name: float(v) for name, v in zip(self.param_names, x)}})
            return val

        for _ in range(self.n_initial):
            _observe(self._sample())

        best_idx = int(np.argmin(y))
        best_val = float(y[best_idx])
        best_x = X[best_idx]

        kernel = Matern(length_scale=np.ones(len(self.param_names)), nu=2.5) + WhiteKernel(noise_level=1e-5)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, random_state=self.rng.integers(0, 10_000))

        for _ in range(self.n_iter):
            gp.fit(np.vstack(X), np.asarray(y))
            candidates = self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.n_candidates, len(self.param_names)))
            mu, sigma = gp.predict(candidates, return_std=True)
            ei = self._expected_improvement(mu, sigma, best_val)
            if not np.any(ei > 0):
                candidate = self._sample()
            else:
                candidate = candidates[int(np.argmax(ei))]
            val = _observe(candidate)
            if val < best_val:
                best_val = val
                best_x = candidate

        return {
            "best_params": {name: float(value) for name, value in zip(self.param_names, best_x)},
            "best_score": float(best_val),
            "trace": trace,
        }


def bayesian_optimize_seir_params(
    history: List[float],
    *,
    population: int,
    initial_exposed: int,
    initial_recovered: int,
    max_history_weeks: int = 24,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Tunes SEIR hyper-parameters (beta/incubation/infectious/initial I) via Bayesian optimization."""

    hist = np.asarray(history[-max_history_weeks:], dtype=float)
    hist = hist[np.isfinite(hist)]
    if hist.size < 4:
        raise ValueError("Not enough historical points for Bayesian tuning")

    bounds = {
        "seir_beta": (0.15, 1.6),
        "seir_incubation_days": (2.0, 7.0),
        "seir_infectious_days": (3.0, 14.0),
        "initial_infectious_override": (1.0, max(5.0, float(hist[-1]) * 2.0)),
    }

    def _objective(x: np.ndarray) -> float:
        beta, incubation_days, infectious_days, initial_I = x
        days = int(len(hist) * 7)
        try:
            daily = run_seir_simulation(
                population=population,
                initial_exposed=int(initial_exposed),
                initial_infectious=max(1, int(initial_I)),
                initial_recovered=int(initial_recovered),
                beta=float(beta),
                incubation_days=float(incubation_days),
                infectious_days=float(infectious_days),
                days=days,
            )
            if daily.size < 7:
                return 1e6
            weekly = np.nanmean(daily.reshape(-1, 7), axis=1)
            k = min(len(weekly), len(hist))
            if k == 0:
                return 1e6
            loss = smape(hist[-k:], weekly[:k])
            if not np.isfinite(loss):
                return 1e6
            return float(loss)
        except Exception:
            return 1e6

    tuner = BayesianParameterTuner(bounds, n_initial=5, n_iter=12, random_state=random_state)
    result = tuner.optimize(_objective)
    params = result["best_params"].copy()
    params["initial_infectious_override"] = float(max(1.0, params["initial_infectious_override"]))

    return {
        "params": params,
        "objective": result["best_score"],
        "trace": result["trace"],
    }


def adjust_parameters_based_on_error(
    params: Dict[str, Any], 
    actual_value: float, 
    predicted_value: float, 
    error_threshold: float = 0.2
) -> Dict[str, Any]:
    """
    실제 값과 예측 값의 오차를 기반으로 파라미터를 자동 조정합니다.
    
    Args:
        params: 현재 파라미터 딕셔너리
        actual_value: 실제 값
        predicted_value: 예측 값
        error_threshold: 오차 임계값 (기본 20%)
        
    Returns:
        조정된 파라미터 딕셔너리
    """
    # 오차율 계산
    if predicted_value > 0:
        error_rate = abs(actual_value - predicted_value) / predicted_value
    else:
        error_rate = 1.0 if actual_value > 0 else 0.0
    
    # 큰 오차의 경우 더 강한 조정
    if error_rate > 0.3:  # 30% 이상 오차
        adjustment_multiplier = 1.5  # 더 강한 조정
    elif error_rate > 0.2:  # 20% 이상 오차
        adjustment_multiplier = 1.3
    else:
        adjustment_multiplier = 1.1
    
    # 오차가 임계값 이하이면 파라미터 조정하지 않음
    if error_rate <= error_threshold:
        return params
    
    # 오차 방향에 따라 파라미터 조정
    adjusted_params = params.copy()
    
    # 실제 값이 예측 값보다 큰 경우 (과소 예측)
    if actual_value > predicted_value:
        adjustment_factor = min(1.3, 1.0 + error_rate * adjustment_multiplier)
        adjusted_params["amplitude_multiplier"] = float(
            min(2.8, adjusted_params.get("amplitude_multiplier", 1.8) * adjustment_factor)
        )
        # r_boost_cap도 함께 조정
        adjusted_params["r_boost_cap"] = float(
            min(3.0, adjusted_params.get("r_boost_cap", 2.0) * adjustment_factor * 1.1)
        )
    # 실제 값이 예측 값보다 작은 경우 (과대 예측)
    elif actual_value < predicted_value:
        adjustment_factor = max(0.7, 1.0 - error_rate * adjustment_multiplier)
        adjusted_params["amplitude_multiplier"] = float(
            max(1.2, adjusted_params.get("amplitude_multiplier", 1.8) * adjustment_factor)
        )
        # r_boost_cap도 함께 조정
        adjusted_params["r_boost_cap"] = float(
            max(1.2, adjusted_params.get("r_boost_cap", 2.0) * adjustment_factor * 0.9)
        )
        # quality도 조정 (불확실성 증가)
        adjusted_params["quality"] = float(
            max(0.5, min(0.95, adjusted_params.get("quality", 0.72) * 0.95))
        )
    
    return adjusted_params


def detect_sudden_change(actual_data_history: List[Dict[str, Any]]) -> bool:
    """급격한 변화 감지"""
    if len(actual_data_history) < 3:
        return False
    
    recent_values = [data["actual_value"] for data in actual_data_history[-3:]]
    
    # 최근 3주간의 변화율 계산
    changes = []
    for i in range(1, len(recent_values)):
        if recent_values[i-1] > 0:
            change_rate = (recent_values[i] - recent_values[i-1]) / recent_values[i-1]
            changes.append(change_rate)
    
    # 평균 변화율이 -30% 이하이면 급격한 하락 판단
    if changes and len(changes) >= 2:
        avg_change = sum(changes) / len(changes)
        return avg_change <= -0.3
    
    return False


def adjust_parameters_for_sudden_change(params: Dict[str, Any]) -> Dict[str, Any]:
    """급격한 변화에 대한 파라미터 조정"""
    adjusted_params = params.copy()
    
    # 급격한 하락의 경우 파라미터를 더 보수적으로 조정
    adjusted_params["amplitude_multiplier"] = float(
        max(1.2, adjusted_params.get("amplitude_multiplier", 1.8) * 0.7)
    )
    adjusted_params["r_boost_cap"] = float(
        max(1.2, adjusted_params.get("r_boost_cap", 2.0) * 0.8)
    )
    adjusted_params["quality"] = float(
        max(0.5, min(0.95, adjusted_params.get("quality", 0.72) * 0.9))
    )
    
    return adjusted_params


def calculate_prediction_error(actual_values: List[float], predicted_values: List[float]) -> Dict[str, float]:
    """
    실제 값과 예측 값 사이의 오차를 계산합니다.
    
    Args:
        actual_values: 실제 값 리스트
        predicted_values: 예측 값 리스트
        
    Returns:
        오차 메트릭 딕셔너리
    """
    if not actual_values or not predicted_values or len(actual_values) != len(predicted_values):
        return {}
    
    import numpy as np
    
    actual = np.array(actual_values)
    predicted = np.array(predicted_values)
    
    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(actual - predicted))
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((actual - predicted) / np.where(actual != 0, actual, 1))) * 100
    
    # RMSE (Root Mean Square Error)
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    
    return {
        "mae": float(mae),
        "mape": float(mape),
        "rmse": float(rmse)
    }


def adaptive_parameter_tuning(
    params: Dict[str, Any],
    recent_errors: List[Dict[str, float]],
    window_size: int = 4
) -> Dict[str, Any]:
    """
    최근 오차 기록을 기반으로 파라미터를 적응적으로 조정합니다.
    
    Args:
        params: 현재 파라미터 딕셔너리
        recent_errors: 최근 오차 기록 리스트
        window_size: 평균을 계산할 오차 기록 수
        
    Returns:
        조정된 파라미터 딕셔너리
    """
    if not recent_errors or len(recent_errors) < 2:
        return params
    
    # 최근 window_size개의 오차만 사용
    recent = recent_errors[-window_size:] if len(recent_errors) > window_size else recent_errors
    
    # MAPE 평균 계산
    avg_mape = np.mean([error.get("mape", 0) for error in recent])
    
    # 오차 추세 분석
    if len(recent) >= 2:
        mape_trend = recent[-1].get("mape", 0) - recent[0].get("mape", 0)
    else:
        mape_trend = 0
    
    adjusted_params = params.copy()
    
    # 오차가 증가하는 추세이면 파라미터를 더 공격적으로 조정
    if mape_trend > 5:  # MAPE가 5% 이상 증가
        # 예측력을 높이기 위해 파라미터 조정
        adjusted_params["amplitude_multiplier"] = float(
            min(2.8, adjusted_params.get("amplitude_multiplier", 1.8) * 1.1)
        )
        adjusted_params["r_boost_cap"] = float(
            min(3.0, adjusted_params.get("r_boost_cap", 2.0) * 1.1)
        )
        adjusted_params["quality"] = float(
            max(0.5, min(0.95, adjusted_params.get("quality", 0.72) * 0.98))
        )
    elif mape_trend < -5:  # MAPE가 5% 이상 감소
        # 안정적인 예측을 위해 파라미터를 더 보수적으로 조정
        adjusted_params["amplitude_multiplier"] = float(
            max(1.2, adjusted_params.get("amplitude_multiplier", 1.8) * 0.95)
        )
        adjusted_params["r_boost_cap"] = float(
            max(1.2, adjusted_params.get("r_boost_cap", 2.0) * 0.95)
        )
        adjusted_params["quality"] = float(
            max(0.5, min(0.95, adjusted_params.get("quality", 0.72) * 1.02))
        )
    
    return adjusted_params