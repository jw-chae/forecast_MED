from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
import math


@dataclass
class FusionResult:
    mean: float
    variance: float
    std: float
    ci95: tuple[float, float]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean,
            "variance": self.variance,
            "std": self.std,
            "ci95": [self.ci95[0], self.ci95[1]],
        }


def precision_weighted_fusion(
    yhat_mean: float,
    yhat_var: float,
    y_obs: float,
    data_quality: float,
    manual_bias_mean: float,
    manual_bias_sd: float,
    kappa_quality: float = 0.5,
    news_signal: Optional[float] = None,
    news_scale: float = 10.0,
    news_var: float = 25.0,
) -> FusionResult:
    y_obs = max(0.0, float(y_obs))
    yhat_mean = max(0.0, float(yhat_mean))
    yhat_var = max(1e-9, float(yhat_var))

    q = min(1.0, max(0.0, float(data_quality)))
    mu_e = float(manual_bias_mean)
    sd_e = max(1e-9, float(manual_bias_sd))

    z = y_obs * (1.0 + mu_e)
    var_obs = (y_obs * sd_e) ** 2 + kappa_quality * (1.0 - q) * (y_obs ** 2)
    var_obs = max(1e-9, float(var_obs))

    use_news = news_signal is not None and not math.isnan(float(news_signal))
    if use_news:
        z_news = float(news_signal) * news_scale
        var_news = max(1e-9, float(news_var))
    else:
        z_news = 0.0
        var_news = None

    precision = (1.0 / yhat_var) + (1.0 / var_obs)
    numerator = (yhat_mean / yhat_var) + (z / var_obs)
    if var_news is not None:
        precision += 1.0 / var_news
        numerator += z_news / var_news

    variance = 1.0 / precision
    mean = numerator * variance
    std = math.sqrt(variance)
    lo = max(0.0, mean - 1.96 * std)
    hi = mean + 1.96 * std

    return FusionResult(mean=mean, variance=variance, std=std, ci95=(lo, hi))

