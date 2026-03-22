"""
Scaling law for speculative denoising acceptance rates.

Models the acceptance rate α as a function of:
- d: draft model size (parameters)
- T: target model size (parameters)
- t: normalized timestep in [0, 1]
- g: guidance scale

Core model:
    α(d, T, t) = [1 − C · (d/T)^(−β)] · h(t)

where h(t) is a timestep-dependent modulation capturing the fact that
early (noisy) steps are easier to predict than late (fine-detail) steps.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


@dataclass
class ScalingLawParams:
    C: float = 0.5
    beta: float = 0.5
    h_params: Dict[str, float] = field(default_factory=dict)
    r_squared: float = 0.0
    rmse: float = 0.0


def alpha_base(d_over_T: np.ndarray, C: float, beta: float) -> np.ndarray:
    """Base acceptance rate as function of model size ratio."""
    return np.clip(1.0 - C * np.power(d_over_T, -beta), 0.0, 1.0)


def h_linear(t: np.ndarray, a: float, b: float) -> np.ndarray:
    """Linear timestep modulation: h(t) = a·t + b."""
    return np.clip(a * t + b, 0.0, 1.0)


def h_cosine(t: np.ndarray, amplitude: float, phase: float) -> np.ndarray:
    """Cosine timestep modulation: h(t) = 1 - amplitude·cos(π·t + phase)."""
    return np.clip(1.0 - amplitude * np.cos(np.pi * t + phase), 0.0, 1.0)


def h_piecewise(t: np.ndarray, breakpoints: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Piecewise linear modulation defined by breakpoint-value pairs."""
    return np.clip(np.interp(t, breakpoints, values), 0.0, 1.0)


def full_model(X: np.ndarray, C: float, beta: float, h_a: float, h_b: float) -> np.ndarray:
    """
    Combined model: α(d/T, t) = alpha_base(d/T) · h_linear(t).
    X[:, 0] = d/T, X[:, 1] = t (normalized)
    """
    d_over_T = X[:, 0]
    t = X[:, 1]
    base = alpha_base(d_over_T, C, beta)
    mod = h_linear(t, h_a, h_b)
    return base * mod


def fit_scaling_law(
    d_sizes: np.ndarray,
    T_sizes: np.ndarray,
    timesteps: np.ndarray,
    acceptance_rates: np.ndarray,
    h_type: str = "linear",
) -> ScalingLawParams:
    """
    Fit the scaling law to measured acceptance rate data.

    Args:
        d_sizes: draft model sizes in billions (N,)
        T_sizes: target model sizes in billions (N,)
        timesteps: normalized timesteps [0,1] (N,)
        acceptance_rates: measured acceptance rates (N,)
        h_type: timestep modulation type ("linear" or "cosine")

    Returns:
        Fitted ScalingLawParams
    """
    d_over_T = d_sizes / T_sizes
    X = np.column_stack([d_over_T, timesteps])
    y = acceptance_rates

    valid = np.isfinite(y) & (y >= 0) & (y <= 1)
    X, y = X[valid], y[valid]

    if len(y) < 4:
        logger.warning("Insufficient data points (%d) for fitting", len(y))
        return ScalingLawParams()

    try:
        if h_type == "linear":
            popt, _ = curve_fit(
                full_model, X, y,
                p0=[0.5, 0.5, 0.3, 0.7],
                bounds=([0.01, 0.01, -2.0, 0.0], [5.0, 3.0, 2.0, 2.0]),
                maxfev=10000,
            )
            C, beta, h_a, h_b = popt
            h_params = {"a": float(h_a), "b": float(h_b)}

        elif h_type == "cosine":
            def cosine_model(X, C, beta, amp, phase):
                d_over_T = X[:, 0]
                t = X[:, 1]
                base = alpha_base(d_over_T, C, beta)
                mod = h_cosine(t, amp, phase)
                return base * mod

            popt, _ = curve_fit(
                cosine_model, X, y,
                p0=[0.5, 0.5, 0.3, 0.0],
                bounds=([0.01, 0.01, 0.0, -np.pi], [5.0, 3.0, 1.0, np.pi]),
                maxfev=10000,
            )
            C, beta, amp, phase = popt
            h_params = {"amplitude": float(amp), "phase": float(phase)}
        else:
            raise ValueError(f"Unknown h_type: {h_type}")

        y_pred = full_model(X, C, beta, h_params.get("a", 0.3), h_params.get("b", 0.7)) if h_type == "linear" else cosine_model(X, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1.0 - ss_res / (ss_tot + 1e-8)
        rmse = np.sqrt(ss_res / len(y))

        params = ScalingLawParams(
            C=float(C), beta=float(beta),
            h_params=h_params,
            r_squared=float(r_squared),
            rmse=float(rmse),
        )
        logger.info(
            "Fit: C=%.4f, β=%.4f, R²=%.4f, RMSE=%.4f",
            C, beta, r_squared, rmse,
        )
        return params

    except Exception as e:
        logger.error("Fitting failed: %s", e)
        return ScalingLawParams()


def optimal_gamma(
    alpha: float,
    draft_cost_ratio: float,
    max_gamma: int = 15,
) -> int:
    """
    Compute optimal draft length γ* that maximizes expected throughput.

    E[accepted + 1] / cost(γ) where cost = γ · c_draft + 1 · c_target.
    draft_cost_ratio = c_draft / c_target (typically d/T).

    Expected accepted steps = Σ_{k=0}^{γ-1} α^k = (1 - α^γ) / (1 - α).
    """
    if alpha <= 0 or alpha >= 1:
        return 1

    best_gamma = 1
    best_throughput = 0.0

    for g in range(1, max_gamma + 1):
        expected_accepted = (1.0 - alpha ** g) / (1.0 - alpha)
        cost = g * draft_cost_ratio + 1.0
        throughput = (expected_accepted + 1.0) / cost
        if throughput > best_throughput:
            best_throughput = throughput
            best_gamma = g

    return best_gamma


def predict_speedup(
    params: ScalingLawParams,
    d_size: float,
    T_size: float,
    num_steps: int = 50,
    draft_cost_ratio: Optional[float] = None,
) -> Dict[str, float]:
    """
    Predict wall-clock speedup using the fitted scaling law.

    Returns dict with predicted acceptance rate, optimal gamma, and speedup factor.
    """
    if draft_cost_ratio is None:
        draft_cost_ratio = d_size / T_size

    d_over_T = d_size / T_size
    t_mid = 0.5

    if "a" in params.h_params:
        alpha = alpha_base(np.array([d_over_T]), params.C, params.beta)[0]
        h_val = h_linear(np.array([t_mid]), params.h_params["a"], params.h_params["b"])[0]
        avg_alpha = alpha * h_val
    else:
        avg_alpha = alpha_base(np.array([d_over_T]), params.C, params.beta)[0]

    gamma_star = optimal_gamma(avg_alpha, draft_cost_ratio)

    baseline_cost = num_steps
    expected_accepted = (1.0 - avg_alpha ** gamma_star) / (1.0 - avg_alpha + 1e-8)
    spec_rounds = num_steps / (expected_accepted + 1.0)
    spec_cost = spec_rounds * (gamma_star * draft_cost_ratio + 1.0)

    speedup = baseline_cost / (spec_cost + 1e-8)

    return {
        "avg_acceptance_rate": float(avg_alpha),
        "optimal_gamma": gamma_star,
        "predicted_speedup": float(speedup),
        "draft_cost_ratio": draft_cost_ratio,
        "expected_accepted_per_round": float(expected_accepted),
    }
