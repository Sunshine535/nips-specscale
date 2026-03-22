#!/usr/bin/env python3
"""Fit the speculative decoding scaling law S(d,T) = c * (T/d)^beta.

Reads all JSON result files produced by benchmark_speculative.py,
aggregates acceptance rates and speedups, then fits:
  - Acceptance rate model:  alpha(d,T) = 1 - C * (d/T)^(-beta)
  - Speedup model:          S(d,T,gamma) = E[accepted] / (gamma * c_ratio + 1)
    where E[accepted] = (1 - alpha^(gamma+1)) / (1 - alpha)

Saves fitted parameters and publication-quality figures.

Usage:
    python scripts/fit_scaling_law.py --results_dir results/ --output_dir figures/
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger("fit_scaling_law")


# ======================================================================
# Data loading
# ======================================================================


def load_all_results(results_dir: str) -> pd.DataFrame:
    """Load every JSON result file into a DataFrame."""
    rows = []
    for p in Path(results_dir).glob("*.json"):
        try:
            with open(p) as f:
                d = json.load(f)
            rows.append(d)
        except Exception as e:
            logger.warning("Skipping %s: %s", p, e)
    if not rows:
        raise RuntimeError(f"No JSON files found in {results_dir}")
    df = pd.DataFrame(rows)
    logger.info("Loaded %d result files", len(df))
    return df


# ======================================================================
# Scaling-law models
# ======================================================================


def alpha_model(d_over_T: np.ndarray, C: float, beta: float) -> np.ndarray:
    """Predicted acceptance rate: alpha = 1 - C * (d/T)^(-beta)."""
    return np.clip(1.0 - C * np.power(d_over_T, -beta), 0.0, 1.0)


def expected_accepted(alpha: float, gamma: int) -> float:
    """E[N] = (1 - alpha^(gamma+1)) / (1 - alpha)  for alpha != 1."""
    if abs(alpha - 1.0) < 1e-12:
        return float(gamma + 1)
    return (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)


def speedup_model(
    params: Tuple[float, ...],
    alpha: np.ndarray,
    gamma: np.ndarray,
    c_ratio: np.ndarray,
) -> np.ndarray:
    """Predicted speedup given acceptance rate, gamma, and latency ratio."""
    en = (1.0 - np.power(alpha, gamma + 1)) / (1.0 - alpha + 1e-12)
    return en / (gamma * c_ratio + 1.0)


# ======================================================================
# Fitting routines
# ======================================================================


def fit_acceptance_rate(df: pd.DataFrame) -> Dict[str, dict]:
    """Fit alpha(d,T) = 1 - C*(d/T)^(-beta) per dataset (domain)."""
    results = {}
    domains = df["dataset"].unique() if "dataset" in df.columns else ["all"]

    for domain in domains:
        sub = df[df["dataset"] == domain] if domain != "all" else df
        if sub.empty:
            continue

        d_over_T = (sub["draft_size_B"] / sub["target_size_B"]).values
        alpha = sub["acceptance_rate_mean"].values

        mask = (d_over_T > 0) & (alpha > 0) & (alpha < 1)
        d_over_T = d_over_T[mask]
        alpha = alpha[mask]
        if len(d_over_T) < 2:
            logger.warning("Not enough data for domain=%s (n=%d)", domain, len(d_over_T))
            continue

        try:
            popt, pcov = curve_fit(
                alpha_model,
                d_over_T,
                alpha,
                p0=[0.5, 0.3],
                bounds=([0, 0], [5.0, 5.0]),
                maxfev=10000,
            )
            perr = np.sqrt(np.diag(pcov))
            predicted = alpha_model(d_over_T, *popt)
            ss_res = np.sum((alpha - predicted) ** 2)
            ss_tot = np.sum((alpha - alpha.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            results[domain] = {
                "C": float(popt[0]),
                "beta": float(popt[1]),
                "C_std": float(perr[0]),
                "beta_std": float(perr[1]),
                "R2": float(r2),
                "n_points": int(len(d_over_T)),
            }
            logger.info(
                "[%s] C=%.4f±%.4f  beta=%.4f±%.4f  R²=%.4f",
                domain,
                popt[0],
                perr[0],
                popt[1],
                perr[1],
                r2,
            )
        except RuntimeError as e:
            logger.error("Curve fit failed for domain=%s: %s", domain, e)

    return results


def fit_speedup_scaling(df: pd.DataFrame) -> Dict[str, float]:
    """Fit overall speedup: S = E[accepted] / (gamma * c_ratio + 1).

    We estimate c_ratio = t_draft / t_verify_per_token from the timing data.
    """
    if "draft_time_seconds" not in df.columns:
        logger.warning("No timing data — skipping speedup fit")
        return {}

    alpha = df["acceptance_rate_mean"].values
    gamma = df["gamma"].values.astype(float)
    speedup = df["wall_clock_speedup"].values

    mask = (alpha > 0) & (alpha < 1) & (speedup > 0)
    alpha, gamma, speedup = alpha[mask], gamma[mask], speedup[mask]

    if len(alpha) < 2:
        return {}

    def _model(X, c_ratio_est):
        a, g = X
        en = (1.0 - np.power(a, g + 1)) / (1.0 - a + 1e-12)
        return en / (g * c_ratio_est + 1.0)

    try:
        popt, pcov = curve_fit(
            _model,
            (alpha, gamma),
            speedup,
            p0=[0.1],
            bounds=([0], [10]),
        )
        predicted = _model((alpha, gamma), *popt)
        mape = np.mean(np.abs((speedup - predicted) / (speedup + 1e-8))) * 100

        return {
            "c_ratio": float(popt[0]),
            "c_ratio_std": float(np.sqrt(pcov[0, 0])),
            "MAPE_percent": float(mape),
            "n_points": int(len(alpha)),
        }
    except RuntimeError as e:
        logger.error("Speedup fit failed: %s", e)
        return {}


# ======================================================================
# Plotting
# ======================================================================

PLT_STYLE = {
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 150,
}


def plot_acceptance_vs_ratio(
    df: pd.DataFrame, fit_params: Dict[str, dict], out_dir: Path
):
    """Scatter plot of alpha vs d/T with fitted curves, faceted by domain."""
    plt.rcParams.update(PLT_STYLE)
    domains = sorted(fit_params.keys())
    n = len(domains)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)
    axes = axes[0]

    for ax, domain in zip(axes, domains):
        sub = df[df["dataset"] == domain] if domain != "all" else df
        d_over_T = sub["draft_size_B"] / sub["target_size_B"]
        alpha = sub["acceptance_rate_mean"]

        ax.scatter(d_over_T, alpha, alpha=0.7, s=40, edgecolors="k", linewidths=0.3)

        if domain in fit_params:
            C = fit_params[domain]["C"]
            beta = fit_params[domain]["beta"]
            r2 = fit_params[domain]["R2"]
            x_fit = np.linspace(d_over_T.min() * 0.8, d_over_T.max() * 1.2, 200)
            y_fit = alpha_model(x_fit, C, beta)
            ax.plot(x_fit, y_fit, "r-", linewidth=2)
            ax.set_title(
                f"{domain}\n"
                rf"$\alpha = 1 - {C:.3f}\,(d/T)^{{-{beta:.3f}}}$   $R^2={r2:.3f}$"
            )

        ax.set_xlabel("d / T  (draft / target size ratio)")
        ax.set_ylabel("Acceptance rate α")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = out_dir / "acceptance_vs_ratio.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_speedup_vs_gamma(df: pd.DataFrame, out_dir: Path):
    """Speedup vs gamma for each model pair."""
    plt.rcParams.update(PLT_STYLE)
    if "wall_clock_speedup" not in df.columns:
        return

    pairs = df.groupby(["draft_model", "target_model"])
    fig, ax = plt.subplots(figsize=(7, 5))

    for (draft, target), grp in pairs:
        grp_sorted = grp.sort_values("gamma")
        label = f"{draft.split('/')[-1]} → {target.split('/')[-1]}"
        ax.plot(
            grp_sorted["gamma"],
            grp_sorted["wall_clock_speedup"],
            "o-",
            label=label,
            markersize=5,
        )

    ax.set_xlabel("Speculation length γ")
    ax.set_ylabel("Wall-clock speedup")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_title("Speedup vs γ per model pair")

    fig.tight_layout()
    path = out_dir / "speedup_vs_gamma.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_3d_surface(df: pd.DataFrame, fit_params: Dict[str, dict], out_dir: Path):
    """3D surface: S(d, T) for a representative gamma and domain."""
    plt.rcParams.update(PLT_STYLE)
    domain = list(fit_params.keys())[0] if fit_params else None
    if domain is None:
        return

    C = fit_params[domain]["C"]
    beta = fit_params[domain]["beta"]

    d_range = np.linspace(0.5, 10, 50)
    T_range = np.linspace(5, 35, 50)
    D, T = np.meshgrid(d_range, T_range)

    mask = D < T
    ratio = np.where(mask, D / T, np.nan)
    alpha = np.where(mask, alpha_model(ratio, C, beta), np.nan)
    alpha = np.clip(alpha, 0.01, 0.999)

    gamma_val = 5
    en = (1.0 - np.power(alpha, gamma_val + 1)) / (1.0 - alpha + 1e-12)
    c_ratio = 0.1
    S = en / (gamma_val * c_ratio + 1.0)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(D, T, S, cmap="viridis", alpha=0.85, edgecolor="none")
    ax.set_xlabel("Draft size d (B)")
    ax.set_ylabel("Target size T (B)")
    ax.set_zlabel("Speedup S")
    ax.set_title(f"Scaling Law Surface  (γ={gamma_val}, domain={domain})")
    fig.colorbar(surf, shrink=0.5, pad=0.1)

    fig.tight_layout()
    path = out_dir / "scaling_law_surface.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_predicted_vs_actual(df: pd.DataFrame, fit_params: Dict[str, dict], out_dir: Path):
    """Predicted vs actual acceptance rate (parity plot)."""
    plt.rcParams.update(PLT_STYLE)
    fig, ax = plt.subplots(figsize=(5.5, 5))

    for domain, params in fit_params.items():
        sub = df[df["dataset"] == domain] if domain != "all" else df
        d_over_T = (sub["draft_size_B"] / sub["target_size_B"]).values
        actual = sub["acceptance_rate_mean"].values
        predicted = alpha_model(d_over_T, params["C"], params["beta"])
        ax.scatter(actual, predicted, s=30, alpha=0.7, label=domain)

    lims = [0, 1]
    ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Actual acceptance rate")
    ax.set_ylabel("Predicted acceptance rate")
    ax.set_title("Predicted vs Actual α")
    ax.legend()
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    fig.tight_layout()
    path = out_dir / "predicted_vs_actual.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


# ======================================================================
# Main
# ======================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fit speculative decoding scaling law")
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--output_dir", type=str, default="figures")
    return p


def main():
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    df = load_all_results(args.results_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fit acceptance rate scaling law per domain
    accept_params = fit_acceptance_rate(df)

    # 2. Fit speedup model
    speedup_params = fit_speedup_scaling(df)

    # 3. Save fitted parameters
    all_params = {
        "acceptance_rate_params": accept_params,
        "speedup_params": speedup_params,
    }
    params_path = out_dir / "fitted_params.json"
    with open(params_path, "w") as f:
        json.dump(all_params, f, indent=2)
    logger.info("Saved fitted parameters → %s", params_path)

    # 4. Generate figures
    plot_acceptance_vs_ratio(df, accept_params, out_dir)
    plot_speedup_vs_gamma(df, out_dir)
    plot_3d_surface(df, accept_params, out_dir)
    plot_predicted_vs_actual(df, accept_params, out_dir)

    logger.info("Done. All figures saved in %s", out_dir)


if __name__ == "__main__":
    main()
