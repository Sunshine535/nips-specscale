#!/usr/bin/env python3
"""
Scaling law analysis for SpecDenoise.

Reads acceptance rate measurements from results/acceptance_sweep/,
fits the parametric scaling law α(d/T, t), compares modulation types,
computes optimal γ* per model pair, and generates publication figures.

Usage:
    python scripts/run_scaling_law_fit.py
    python scripts/run_scaling_law_fit.py --input_dir results/acceptance_sweep --figure_format pdf
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.scaling_law import (
    ScalingLawParams,
    fit_scaling_law,
    optimal_gamma,
    predict_speedup,
    alpha_base,
    h_linear,
    h_cosine,
)
from src.dit_loader import MODEL_CONFIGS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("scaling_law_fit")


def parse_args():
    parser = argparse.ArgumentParser(description="Fit scaling law from acceptance sweep data")
    parser.add_argument("--input_dir", type=str, default="results/acceptance_sweep")
    parser.add_argument("--output_dir", type=str, default="results/scaling_law")
    parser.add_argument("--figure_format", type=str, default="pdf", choices=["pdf", "png", "svg"])
    parser.add_argument("--figure_dpi", type=int, default=300)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()


def load_sweep_results(input_dir: Path) -> List[dict]:
    """Load all JSON results from the acceptance sweep directory."""
    results = []
    if not input_dir.exists():
        logger.error("Input directory not found: %s", input_dir)
        return results

    for path in sorted(input_dir.glob("*.json")):
        if path.name == "sweep_summary.json":
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            if "error" not in data and "acceptance_rate" in data:
                results.append(data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Skipping %s: %s", path.name, e)
    logger.info("Loaded %d valid measurement files from %s", len(results), input_dir)
    return results


def extract_fitting_data(results: List[dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract arrays for scaling law fitting.

    For each result, expand per-timestep acceptance rates into individual data points.
    Returns (d_sizes, T_sizes, timesteps, acceptance_rates) all in shape (N,).
    """
    d_list, T_list, t_list, alpha_list = [], [], [], []

    for r in results:
        d_M = r.get("draft_params_M", 0)
        T_M = r.get("target_params_M", 0)
        if d_M == 0 or T_M == 0:
            continue

        per_t = r.get("acceptance_rate_per_timestep", [])
        if per_t:
            n_bins = len(per_t)
            for i, alpha_val in enumerate(per_t):
                t_norm = (i + 0.5) / n_bins
                d_list.append(d_M / 1000.0)
                T_list.append(T_M / 1000.0)
                t_list.append(t_norm)
                alpha_list.append(alpha_val)
        else:
            # Use overall acceptance rate at t=0.5
            d_list.append(d_M / 1000.0)
            T_list.append(T_M / 1000.0)
            t_list.append(0.5)
            alpha_list.append(r["acceptance_rate"])

    return (
        np.array(d_list),
        np.array(T_list),
        np.array(t_list),
        np.array(alpha_list),
    )


def fit_both_modulations(
    d_sizes: np.ndarray,
    T_sizes: np.ndarray,
    timesteps: np.ndarray,
    acceptance_rates: np.ndarray,
) -> Dict[str, ScalingLawParams]:
    """Fit scaling law with both linear and cosine h(t) modulations."""
    fits = {}
    for h_type in ["linear", "cosine"]:
        logger.info("Fitting with h_type=%s ...", h_type)
        params = fit_scaling_law(d_sizes, T_sizes, timesteps, acceptance_rates, h_type=h_type)
        fits[h_type] = params
        logger.info("  C=%.4f, β=%.4f, R²=%.4f, RMSE=%.4f",
                     params.C, params.beta, params.r_squared, params.rmse)
    return fits


def compute_optimal_gammas(
    fits: Dict[str, ScalingLawParams],
    model_pairs: List[dict],
) -> Dict[str, dict]:
    """Compute optimal γ* for each model pair using the best-fit scaling law."""
    best_h = max(fits, key=lambda k: fits[k].r_squared)
    params = fits[best_h]
    logger.info("Using best fit: h_type=%s (R²=%.4f)", best_h, params.r_squared)

    gamma_results = {}
    for pair in model_pairs:
        d_M = pair.get("draft_params_M", MODEL_CONFIGS.get(pair["draft"], {}).get("params_M", 33))
        T_M = pair.get("target_params_M", MODEL_CONFIGS.get(pair["target"], {}).get("params_M", 675))
        d_B = d_M / 1000.0
        T_B = T_M / 1000.0
        draft_cost_ratio = d_B / T_B

        pred = predict_speedup(params, d_B, T_B, num_steps=50, draft_cost_ratio=draft_cost_ratio)

        pair_name = f"{pair['draft']}→{pair['target']}"
        gamma_results[pair_name] = {
            "draft": pair["draft"],
            "target": pair["target"],
            "draft_params_M": d_M,
            "target_params_M": T_M,
            "draft_cost_ratio": draft_cost_ratio,
            **pred,
        }
        logger.info("  %s: γ*=%d, α=%.3f, speedup=%.2f×",
                     pair_name, pred["optimal_gamma"],
                     pred["avg_acceptance_rate"], pred["predicted_speedup"])

    return gamma_results


def generate_figures(
    results: List[dict],
    fits: Dict[str, ScalingLawParams],
    gamma_results: Dict[str, dict],
    d_sizes: np.ndarray,
    T_sizes: np.ndarray,
    timesteps: np.ndarray,
    acceptance_rates: np.ndarray,
    output_dir: Path,
    fmt: str = "pdf",
    dpi: int = 300,
):
    """Generate publication-quality figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        logger.error("matplotlib not available; skipping figure generation")
        return

    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 9,
        "figure.dpi": dpi,
    })
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Figure 1: α vs d/T with fit curve ----
    fig, ax = plt.subplots(figsize=(6, 4))
    d_over_T = d_sizes / T_sizes
    unique_ratios = np.unique(d_over_T)
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(unique_ratios), 3)))

    for i, ratio in enumerate(unique_ratios):
        mask = d_over_T == ratio
        ax.scatter(timesteps[mask], acceptance_rates[mask],
                   c=[colors[i]], alpha=0.4, s=15, label=f"d/T={ratio:.3f}")

    t_fit = np.linspace(0, 1, 200)
    for h_type, params in fits.items():
        d_ratio_mid = np.median(unique_ratios)
        base = alpha_base(np.array([d_ratio_mid]), params.C, params.beta)[0]
        if h_type == "linear":
            h_vals = h_linear(t_fit, params.h_params.get("a", 0.3), params.h_params.get("b", 0.7))
        else:
            h_vals = h_cosine(t_fit, params.h_params.get("amplitude", 0.3),
                              params.h_params.get("phase", 0.0))
        ax.plot(t_fit, base * h_vals, linewidth=2,
                label=f"Fit ({h_type}, R²={params.r_squared:.3f})")

    ax.set_xlabel("Normalized timestep $t$")
    ax.set_ylabel("Acceptance rate $\\alpha$")
    ax.set_title("Scaling Law Fit: $\\alpha(d/T, t)$")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"scaling_law_fit.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved scaling_law_fit.%s", fmt)

    # ---- Figure 2: Timestep modulation ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    # Aggregate per-timestep acceptance per model pair
    pair_timestep_data = {}
    for r in results:
        pair_key = f"{r['draft_model']}→{r['target_model']}"
        per_t = r.get("acceptance_rate_per_timestep", [])
        if per_t:
            if pair_key not in pair_timestep_data:
                pair_timestep_data[pair_key] = []
            pair_timestep_data[pair_key].append(per_t)

    pair_colors = plt.cm.tab10(np.linspace(0, 0.5, max(len(pair_timestep_data), 1)))
    for ax_idx, (h_type, params) in enumerate(fits.items()):
        ax = axes[ax_idx]
        for j, (pair_key, per_t_list) in enumerate(pair_timestep_data.items()):
            max_len = max(len(p) for p in per_t_list)
            avg_per_t = []
            for b in range(max_len):
                vals = [p[b] for p in per_t_list if b < len(p)]
                avg_per_t.append(np.mean(vals))
            t_centers = [(i + 0.5) / max_len for i in range(max_len)]
            ax.bar([t + j * 0.12 for t in t_centers], avg_per_t,
                   width=0.1, color=pair_colors[j], alpha=0.7, label=pair_key)

        t_curve = np.linspace(0, 1, 100)
        if h_type == "linear":
            h_vals = h_linear(t_curve, params.h_params.get("a", 0.3), params.h_params.get("b", 0.7))
        else:
            h_vals = h_cosine(t_curve, params.h_params.get("amplitude", 0.3),
                              params.h_params.get("phase", 0.0))
        ax.plot(t_curve, h_vals, "k--", linewidth=2, label=f"$h(t)$ fit")

        ax.set_xlabel("Normalized timestep $t$")
        ax.set_title(f"h(t) modulation ({h_type})")
        ax.legend(fontsize=8, loc="lower left")
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Acceptance rate / $h(t)$")
    fig.tight_layout()
    fig.savefig(output_dir / f"timestep_modulation.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved timestep_modulation.%s", fmt)

    # ---- Figure 3: Optimal γ* per model pair ----
    fig, ax = plt.subplots(figsize=(6, 4))
    pair_names = list(gamma_results.keys())
    gamma_stars = [gamma_results[p]["optimal_gamma"] for p in pair_names]
    speedups = [gamma_results[p]["predicted_speedup"] for p in pair_names]
    alphas = [gamma_results[p]["avg_acceptance_rate"] for p in pair_names]

    x_pos = np.arange(len(pair_names))
    bars = ax.bar(x_pos, gamma_stars, color=plt.cm.Paired(np.linspace(0.2, 0.8, len(pair_names))),
                  edgecolor="black", linewidth=0.5)

    for i, (g, s, a) in enumerate(zip(gamma_stars, speedups, alphas)):
        ax.text(i, g + 0.2, f"γ*={g}\n{s:.1f}×", ha="center", fontsize=9)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(pair_names, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Optimal $\\gamma^*$")
    ax.set_title("Optimal Draft Length per Model Pair")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / f"optimal_gamma.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved optimal_gamma.%s", fmt)

    # ---- Figure 4: Predicted vs measured speedup ----
    fig, ax = plt.subplots(figsize=(5, 5))

    measured_speedups = {}
    for r in results:
        pair_key = f"{r['draft_model']}→{r['target_model']}"
        gamma = r.get("gamma", 5)
        wall_time = r.get("avg_wall_time", 0)
        acc_rate = r.get("acceptance_rate", 0)
        if wall_time > 0 and acc_rate > 0:
            if pair_key not in measured_speedups:
                measured_speedups[pair_key] = []
            # Approximate measured speedup from NFE reduction
            baseline_nfe = r.get("num_inference_steps", 50)
            actual_nfe = r.get("avg_nfe", baseline_nfe)
            if actual_nfe > 0:
                measured_speedups[pair_key].append(baseline_nfe / actual_nfe)

    predicted_list = []
    measured_list = []
    labels = []
    for pair_key, pred in gamma_results.items():
        pred_s = pred["predicted_speedup"]
        if pair_key in measured_speedups and measured_speedups[pair_key]:
            meas_s = np.mean(measured_speedups[pair_key])
            predicted_list.append(pred_s)
            measured_list.append(meas_s)
            labels.append(pair_key)

    if predicted_list:
        ax.scatter(predicted_list, measured_list, s=80, zorder=5, edgecolors="black", linewidths=0.5)
        for i, lbl in enumerate(labels):
            ax.annotate(lbl, (predicted_list[i], measured_list[i]),
                        fontsize=8, xytext=(5, 5), textcoords="offset points")
        lims = [0, max(max(predicted_list), max(measured_list)) * 1.2]
        ax.plot(lims, lims, "k--", alpha=0.5, label="$y=x$")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
    else:
        ax.text(0.5, 0.5, "No matched data\n(run eval first)",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)

    ax.set_xlabel("Predicted speedup")
    ax.set_ylabel("Measured speedup (NFE ratio)")
    ax.set_title("Predicted vs Measured Speedup")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output_dir / f"predicted_vs_measured.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved predicted_vs_measured.%s", fmt)


def main():
    args = parse_args()
    input_dir = PROJECT_ROOT / args.input_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_sweep_results(input_dir)
    if not results:
        logger.error("No results found. Run acceptance sweep first.")
        sys.exit(1)

    d_sizes, T_sizes, timesteps, acceptance_rates = extract_fitting_data(results)
    logger.info("Extracted %d data points for fitting", len(d_sizes))

    if len(d_sizes) < 4:
        logger.error("Too few data points (%d) for fitting. Need ≥4.", len(d_sizes))
        sys.exit(1)

    fits = fit_both_modulations(d_sizes, T_sizes, timesteps, acceptance_rates)

    model_pairs = [
        {"draft": "DiT-S/2", "target": "DiT-XL/2", "draft_params_M": 33, "target_params_M": 675},
        {"draft": "DiT-B/2", "target": "DiT-XL/2", "draft_params_M": 131, "target_params_M": 675},
        {"draft": "DiT-S/2", "target": "DiT-B/2", "draft_params_M": 33, "target_params_M": 131},
    ]
    gamma_results = compute_optimal_gammas(fits, model_pairs)

    generate_figures(
        results, fits, gamma_results,
        d_sizes, T_sizes, timesteps, acceptance_rates,
        output_dir, fmt=args.figure_format, dpi=args.figure_dpi,
    )

    analysis = {
        "num_data_points": len(d_sizes),
        "num_measurement_files": len(results),
        "fits": {},
        "gamma_optimization": gamma_results,
        "best_h_type": max(fits, key=lambda k: fits[k].r_squared),
    }
    for h_type, params in fits.items():
        analysis["fits"][h_type] = {
            "C": params.C,
            "beta": params.beta,
            "h_params": params.h_params,
            "r_squared": params.r_squared,
            "rmse": params.rmse,
        }

    analysis_path = output_dir / "analysis_results.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    logger.info("Analysis saved: %s", analysis_path)

    logger.info("=" * 50)
    logger.info("Summary:")
    best_h = analysis["best_h_type"]
    best_fit = fits[best_h]
    logger.info("  Best fit: h_type=%s, R²=%.4f, RMSE=%.4f", best_h, best_fit.r_squared, best_fit.rmse)
    logger.info("  C=%.4f, β=%.4f", best_fit.C, best_fit.beta)
    for pair_name, gr in gamma_results.items():
        logger.info("  %s: γ*=%d, α=%.3f, predicted %.2f× speedup",
                     pair_name, gr["optimal_gamma"],
                     gr["avg_acceptance_rate"], gr["predicted_speedup"])


if __name__ == "__main__":
    main()
