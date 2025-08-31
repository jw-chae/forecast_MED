from __future__ import annotations
import numpy as np
from scipy.integrate import odeint
from typing import Tuple

def seir_model(
    y: Tuple[float, float, float, float],
    t: np.ndarray,
    beta: float,
    sigma: float,
    gamma: float,
    N: int,
):
    """SEIR 모델의 미분 방정식 시스템"""
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

def run_seir_simulation(
    population: int,
    initial_exposed: int,
    initial_infectious: int,
    initial_recovered: int,
    beta: float,
    incubation_days: float,
    infectious_days: float,
    days: int,
) -> np.ndarray:
    """주어진 파라미터로 SEIR 시뮬레이션을 실행하고 감염자 수를 반환합니다."""
    
    if incubation_days <= 0 or infectious_days <= 0:
        raise ValueError("Incubation and infectious days must be positive.")

    sigma = 1.0 / incubation_days
    gamma = 1.0 / infectious_days
    
    S0 = population - initial_exposed - initial_infectious - initial_recovered
    E0 = initial_exposed
    I0 = initial_infectious
    R0 = initial_recovered
    
    t = np.linspace(0, days - 1, days)
    
    solution = odeint(
        seir_model,
        (S0, E0, I0, R0),
        t,
        args=(beta, sigma, gamma, population),
    )
    
    # 감염자 수(I) 반환
    return solution[:, 2]
