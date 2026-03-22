#!/usr/bin/env python3
"""Complete scaling law analysis for speculative decoding.

Reads sweep results, fits:
  1. Per-dataset scaling law:  α(d/T) = 1 - C·(d/T)^(-β)
  2. Position-dependent acceptance from gamma-varying data
  3. Optimal γ* = argmax_γ E[tokens]/cost(γ)
  4. Theoretical throughput vs measured

Generates publication figures:
  - scaling_law_fit.pdf
  - position_decay.pdf
  - optimal_gamma.pdf
  - throughput_prediction.pdf
  - pareto_frontier.pdf

Usage:
    python scripts/run_scaling_law_analysis.py \
        --results_dir results/sweep/ \
        --output_dir results/analysis/
"""

import os
import sys
import json
import re
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize_scalar

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("scaling_law_analysis")

PLT_STYLE = {
    "font.size": 12,
    "font.family": "serif",
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 9,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 150,
}


# ======================================================================
# Data loading
# ======================================================================


def load_sweep_results(results_dir: str) -> pd.DataFrame:
    """Load all JSON result files recursively, tagging each with its seed."""
    rows: List[dict] = []
    for p in Path(results_dir).rglob("*.json"):
        try:
            with open(p) as f:
                d = json.load(f)
            parent = p.parent.name
            m = re.search(r"seed[_]?(\d+)", parent)
            d["seed"] = int(m.group(1)) if m else 0
            d["source_file"] = str(p)
            rows.append(d)
        except Exception as exc:
            logger.warning("Skipping %s: %s", p, exc)

    if not rows:
        raise RuntimeError(f"No JSON files found under {results_dir}")

    df = pd.DataFrame(rows)

    for col in ("draft_size_B", "target_size_B"):
        if col not in df.columns:
            df[col] = df["draft_model"].apply(_infer_size) if "draft" in col else df["target_model"].apply(_infer_size)

    logger.info(
        "Loaded %d files — %d pairs, gammas %s, datasets %s, seeds %s",
        len(df),
        df.groupby(["draft_model", "target_model"]).ngroups,
        sorted(df["gamma"].unique()),
        sorted(df["dataset"].unique()),
        sorted(df["seed"].unique()),
    )
    return df


def _infer_size(name: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)[Bb]", name)
    return float(m.group(1)) if m else 0.0


def aggregate_across_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """Average numeric columns across seeds for each configuration."""
    group_cols = ["draft_model", "target_model", "gamma", "dataset"]
    num_cols = df.select_dtypes(include=[np.number]).columns.difference(group_cols + ["seed"])
    agg_spec = {c: "mean" for c in num_cols}
    agg_spec["seed"] = "count"
    agg = df.groupby(group_cols, as_index=False).agg(agg_spec)
    agg.rename(columns={"seed": "n_seeds"}, inplace=True)
    return agg


# ======================================================================
# Scaling-law models
# ======================================================================


def alpha_model(d_over_T: np.ndarray, C: float, beta: float) -> np.ndarray:
    """α(d/T) = 1 - C·(d/T)^{-β}"""
    return np.clip(1.0 - C * np.power(d_over_T, -beta), 0.0, 1.0)


def expected_accepted(alpha: float, gamma: int) -> float:
    """E[N_acc] = α(1 - α^γ) / (1 - α)  (geometric per-position model)."""
    a = np.clip(alpha, 1e-10, 1.0 - 1e-10)
    return a * (1.0 - a ** gamma) / (1.0 - a)


def predicted_ar(alpha: float, gamma: int) -> float:
    """Predicted overall acceptance rate = E[N_acc] / γ."""
    return expected_accepted(alpha, gamma) / gamma


# ======================================================================
# Fitting routines
# ======================================================================


def fit_scaling_law(df: pd.DataFrame) -> Dict[str, dict]:
    """Fit α(d/T) = 1 - C·(d/T)^{-β} per dataset and overall.

    Uses median gamma as the reference to avoid confounding with γ.
    """
    agg = aggregate_across_seeds(df)

    ref_gamma = 5 if 5 in agg["gamma"].values else int(np.median(agg["gamma"].unique()))
    agg_ref = agg[agg["gamma"] == ref_gamma]

    pair_agg = agg_ref.groupby(["draft_model", "target_model", "dataset"], as_index=False).agg(
        alpha_mean=("acceptance_rate_mean", "mean"),
        d=("draft_size_B", "first"),
        T=("target_size_B", "first"),
    )
    pair_agg["d_over_T"] = pair_agg["d"] / pair_agg["T"]

    results: Dict[str, dict] = {}
    domains = list(pair_agg["dataset"].unique()) + ["all"]

    for domain in domains:
        sub = pair_agg if domain == "all" else pair_agg[pair_agg["dataset"] == domain]
        x = sub["d_over_T"].values
        y = sub["alpha_mean"].values

        mask = (x > 0) & (y > 0) & (y < 1)
        x, y = x[mask], y[mask]
        if len(x) < 2:
            logger.warning("Skipping domain=%s (n=%d < 2)", domain, len(x))
            continue

        try:
            popt, pcov = curve_fit(
                alpha_model, x, y,
                p0=[0.5, 0.3],
                bounds=([1e-6, 1e-6], [5.0, 5.0]),
                maxfev=20000,
            )
            perr = np.sqrt(np.diag(pcov))
            pred = alpha_model(x, *popt)
            ss_res = np.sum((y - pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            results[domain] = dict(
                C=float(popt[0]), beta=float(popt[1]),
                C_std=float(perr[0]), beta_std=float(perr[1]),
                R2=float(r2), n_points=int(len(x)), ref_gamma=ref_gamma,
            )
            logger.info(
                "[%s] C=%.4f±%.4f  β=%.4f±%.4f  R²=%.4f  (n=%d)",
                domain, popt[0], perr[0], popt[1], perr[1], r2, len(x),
            )
        except RuntimeError as exc:
            logger.error("Fit failed for domain=%s: %s", domain, exc)

    return results


def fit_position_dependent(df: pd.DataFrame) -> Dict[str, dict]:
    """Estimate base per-position acceptance α for each (draft,target) pair.

    Uses all gamma values: acceptance_rate(γ) ≈ α(1-α^γ) / (γ(1-α))
    and minimises MSE over γ to find α.
    """
    agg = aggregate_across_seeds(df)
    results: Dict[str, dict] = {}

    for (draft, target), grp in agg.groupby(["draft_model", "target_model"]):
        gam_ar = grp.groupby("gamma")["acceptance_rate_mean"].mean()
        gamma_vals = gam_ar.index.values.astype(float)
        ar_vals = gam_ar.values

        def _mse(log_a):
            a = np.exp(log_a)
            pred = np.array([predicted_ar(a, int(g)) for g in gamma_vals])
            return float(np.mean((ar_vals - pred) ** 2))

        res = minimize_scalar(_mse, bounds=(np.log(0.01), np.log(0.999)), method="bounded")
        alpha_fit = float(np.exp(res.x))

        pred_rates = np.array([predicted_ar(alpha_fit, int(g)) for g in gamma_vals])
        ss_res = np.sum((ar_vals - pred_rates) ** 2)
        ss_tot = np.sum((ar_vals - ar_vals.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        max_g = int(gamma_vals.max())
        pair_key = f"{draft.split('/')[-1]}__{target.split('/')[-1]}"
        results[pair_key] = dict(
            draft_model=draft, target_model=target,
            d=float(grp["draft_size_B"].iloc[0]),
            T=float(grp["target_size_B"].iloc[0]),
            base_alpha=alpha_fit,
            R2_gamma_fit=float(r2),
            gamma_values=gamma_vals.tolist(),
            observed_ar=ar_vals.tolist(),
            predicted_ar=pred_rates.tolist(),
            position_rates=[float(alpha_fit ** k) for k in range(1, max_g + 1)],
        )
        logger.info("[%s] base_α=%.4f  R²=%.4f", pair_key, alpha_fit, r2)

    return results


def compute_optimal_gamma(
    position_params: Dict[str, dict],
    max_gamma: int = 20,
) -> Dict[str, dict]:
    """Find γ* = argmax_γ (E[N]+1) / (γ·c_ratio + 1) for each pair.

    c_ratio = c_draft / c_target is approximated as d/T (FLOP-proportional).
    """
    results: Dict[str, dict] = {}
    for pair_key, p in position_params.items():
        alpha = p["base_alpha"]
        c_ratio = p["d"] / p["T"]

        best_g, best_tp = 1, 0.0
        per_gamma: Dict[int, dict] = {}
        for g in range(1, max_gamma + 1):
            e_acc = expected_accepted(alpha, g)
            e_tok = e_acc + 1.0
            cost = g * c_ratio + 1.0
            tp = e_tok / cost
            per_gamma[g] = dict(
                e_accepted=float(e_acc), e_tokens=float(e_tok),
                cost=float(cost), throughput_ratio=float(tp),
            )
            if tp > best_tp:
                best_tp, best_g = tp, g

        results[pair_key] = dict(
            optimal_gamma=best_g,
            max_throughput_ratio=float(best_tp),
            base_alpha=alpha,
            d_over_T=p["d"] / p["T"],
            c_ratio=c_ratio,
            per_gamma=per_gamma,
        )
        logger.info("[%s] γ*=%d  throughput_ratio=%.3f  α=%.4f", pair_key, best_g, best_tp, alpha)

    return results


def compute_throughput_predictions(
    df: pd.DataFrame,
    scaling_params: Dict[str, dict],
) -> pd.DataFrame:
    """Predict speedup from the fitted scaling law and compare to measured."""
    if "all" in scaling_params:
        C, beta = scaling_params["all"]["C"], scaling_params["all"]["beta"]
    elif scaling_params:
        v = next(iter(scaling_params.values()))
        C, beta = v["C"], v["beta"]
    else:
        return pd.DataFrame()

    agg = aggregate_across_seeds(df)
    records: List[dict] = []

    for _, row in agg.iterrows():
        d_T = row["draft_size_B"] / row["target_size_B"]
        g = int(row["gamma"])
        alpha_pred = float(alpha_model(np.array([d_T]), C, beta)[0])
        e_acc = expected_accepted(alpha_pred, g)
        c_ratio = d_T
        pred_speedup = (e_acc + 1) / (g * c_ratio + 1)

        records.append(dict(
            draft_model=row["draft_model"],
            target_model=row["target_model"],
            gamma=g,
            dataset=row.get("dataset", ""),
            d_over_T=d_T,
            alpha_predicted=alpha_pred,
            alpha_actual=row["acceptance_rate_mean"],
            speedup_predicted=pred_speedup,
            speedup_actual=row.get("wall_clock_speedup", np.nan),
        ))

    pred_df = pd.DataFrame(records)
    valid = pred_df.dropna(subset=["speedup_actual"])
    if len(valid) > 0:
        mape = np.mean(np.abs(
            (valid["speedup_actual"] - valid["speedup_predicted"])
            / (valid["speedup_actual"].abs() + 1e-8)
        )) * 100
        logger.info("Speedup prediction MAPE: %.2f%% (n=%d)", mape, len(valid))
    return pred_df


# ======================================================================
# Plotting
# ======================================================================


def plot_scaling_law_fit(
    df: pd.DataFrame,
    scaling_params: Dict[str, dict],
    out_dir: Path,
):
    """α vs d/T scatter with fitted power-law curves, per domain."""
    plt.rcParams.update(PLT_STYLE)

    domains = sorted(k for k in scaling_params if k != "all")
    if not domains:
        return

    agg = aggregate_across_seeds(df)
    ref_gamma = next(iter(scaling_params.values())).get("ref_gamma", 5)
    agg_ref = agg[agg["gamma"] == ref_gamma] if ref_gamma in agg["gamma"].values else agg

    pair_agg = agg_ref.groupby(["draft_model", "target_model", "dataset"], as_index=False).agg(
        alpha=("acceptance_rate_mean", "mean"),
        d=("draft_size_B", "first"),
        T=("target_size_B", "first"),
    )
    pair_agg["d_over_T"] = pair_agg["d"] / pair_agg["T"]

    cols = min(len(domains), 2)
    rows = (len(domains) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)

    for idx, domain in enumerate(domains):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        sub = pair_agg[pair_agg["dataset"] == domain]
        x, y = sub["d_over_T"].values, sub["alpha"].values

        ax.scatter(x, y, s=50, alpha=0.8, edgecolors="k", linewidths=0.4, zorder=5)

        if domain in scaling_params:
            p = scaling_params[domain]
            xf = np.linspace(max(x.min() * 0.5, 0.005), min(x.max() * 1.5, 1.0), 300)
            yf = alpha_model(xf, p["C"], p["beta"])
            ax.plot(xf, yf, "r-", linewidth=2, zorder=10)
            ax.text(
                0.05, 0.05,
                f"$C={p['C']:.3f}$\n$\\beta={p['beta']:.3f}$\n$R^2={p['R2']:.3f}$",
                transform=ax.transAxes, fontsize=9, va="bottom",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        ax.set_xlabel("$d/T$")
        ax.set_ylabel("Acceptance rate $\\alpha$")
        ax.set_title(domain)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    for idx in range(len(domains), rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    fig.suptitle(
        r"Scaling Law: $\alpha(d/T) = 1 - C \cdot (d/T)^{-\beta}$"
        f"  (γ={ref_gamma})",
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save_fig(fig, out_dir / "scaling_law_fit.pdf")


def plot_position_decay(
    position_params: Dict[str, dict],
    out_dir: Path,
):
    """Left: acceptance rate vs γ (observed + predicted).  Right: α^k decay."""
    plt.rcParams.update(PLT_STYLE)
    pairs = list(position_params.keys())
    if not pairs:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(pairs)))

    for i, pk in enumerate(sorted(pairs, key=lambda k: position_params[k]["d"] / position_params[k]["T"])):
        p = position_params[pk]
        label = pk.replace("Qwen3.5-", "").replace("__", "→")

        ax1.plot(p["gamma_values"], p["observed_ar"], "o", color=cmap[i], markersize=5)
        g_dense = np.linspace(min(p["gamma_values"]), max(p["gamma_values"]), 100)
        pred_dense = [predicted_ar(p["base_alpha"], g) for g in g_dense]
        ax1.plot(g_dense, pred_dense, "-", color=cmap[i], linewidth=1.5, label=label)

        positions = np.arange(1, len(p["position_rates"]) + 1)
        ax2.plot(positions, p["position_rates"], "o-", color=cmap[i],
                 markersize=4, linewidth=1.2, label=f"{label} (α={p['base_alpha']:.3f})")

    ax1.set_xlabel("Speculation length $\\gamma$")
    ax1.set_ylabel("Acceptance rate")
    ax1.set_title("Acceptance Rate vs γ  (dots=observed, lines=geometric fit)")
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Position $k$")
    ax2.set_ylabel("$\\alpha_k = \\alpha^k$")
    ax2.set_title("Per-Position Acceptance Decay")
    ax2.legend(fontsize=7, ncol=2)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout()
    _save_fig(fig, out_dir / "position_decay.pdf")


def plot_optimal_gamma(
    optimal_params: Dict[str, dict],
    out_dir: Path,
):
    """Bar chart of γ* per pair + throughput-ratio curves."""
    plt.rcParams.update(PLT_STYLE)
    if not optimal_params:
        return

    pairs = sorted(optimal_params.keys(), key=lambda k: optimal_params[k]["d_over_T"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    labels = [p.replace("Qwen3.5-", "").replace("__", "\n→ ") for p in pairs]
    gammas = [optimal_params[p]["optimal_gamma"] for p in pairs]
    alphas = [optimal_params[p]["base_alpha"] for p in pairs]

    bars = ax1.bar(range(len(pairs)), gammas, color="steelblue", alpha=0.85)
    for i, (bar, a) in enumerate(zip(bars, alphas)):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"α={a:.2f}", ha="center", fontsize=7)
    ax1.set_xticks(range(len(pairs)))
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Optimal $\\gamma^*$")
    ax1.set_title("Optimal Speculation Length per Pair")
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.grid(True, alpha=0.3, axis="y")

    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(pairs)))
    for i, pk in enumerate(pairs):
        pg = optimal_params[pk]["per_gamma"]
        gs = sorted(pg.keys(), key=int)
        tps = [pg[g]["throughput_ratio"] for g in gs]
        label = pk.replace("Qwen3.5-", "").replace("__", "→")
        ax2.plot([int(g) for g in gs], tps, "o-", color=cmap[i],
                 markersize=4, linewidth=1.2, label=label)
        opt_g = optimal_params[pk]["optimal_gamma"]
        ax2.plot(opt_g, pg[opt_g]["throughput_ratio"], "r*", markersize=10, zorder=10)

    ax2.set_xlabel("$\\gamma$")
    ax2.set_ylabel("Throughput ratio  ($E[N]/\\mathrm{cost}$)")
    ax2.set_title("Throughput vs Speculation Length (★ = optimal)")
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout()
    _save_fig(fig, out_dir / "optimal_gamma.pdf")


def plot_throughput_prediction(pred_df: pd.DataFrame, out_dir: Path):
    """Parity plots: predicted vs actual for α and speedup."""
    plt.rcParams.update(PLT_STYLE)
    if pred_df.empty:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    datasets = pred_df["dataset"].unique()
    colors = dict(zip(datasets, plt.cm.Set2(np.linspace(0, 0.8, len(datasets)))))

    for ds in datasets:
        sub = pred_df[pred_df["dataset"] == ds]
        ax1.scatter(sub["alpha_actual"], sub["alpha_predicted"],
                    s=25, alpha=0.6, color=colors[ds], label=ds,
                    edgecolors="k", linewidths=0.2)

    ax1.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax1.set_xlabel("Actual $\\alpha$")
    ax1.set_ylabel("Predicted $\\alpha$")
    ax1.set_title("Acceptance Rate: Predicted vs Actual")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect("equal")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    valid = pred_df.dropna(subset=["speedup_actual"])
    if len(valid) > 0:
        for ds in datasets:
            sub = valid[valid["dataset"] == ds]
            if sub.empty:
                continue
            ax2.scatter(sub["speedup_actual"], sub["speedup_predicted"],
                        s=25, alpha=0.6, color=colors[ds], label=ds,
                        edgecolors="k", linewidths=0.2)
        hi = max(valid["speedup_actual"].max(), valid["speedup_predicted"].max()) * 1.1
        ax2.plot([0, hi], [0, hi], "k--", linewidth=1, alpha=0.5)
        mape = np.mean(np.abs(
            (valid["speedup_actual"] - valid["speedup_predicted"])
            / (valid["speedup_actual"].abs() + 1e-8)
        )) * 100
        ax2.set_title(f"Speedup: Predicted vs Actual  (MAPE={mape:.1f}%)")
    else:
        ax2.set_title("Speedup: no actual data available")

    ax2.set_xlabel("Actual speedup")
    ax2.set_ylabel("Predicted speedup")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    _save_fig(fig, out_dir / "throughput_prediction.pdf")


def plot_pareto_frontier(
    df: pd.DataFrame,
    optimal_params: Dict[str, dict],
    out_dir: Path,
):
    """Speedup vs acceptance rate, coloured by d/T, with optimal points."""
    plt.rcParams.update(PLT_STYLE)
    agg = aggregate_across_seeds(df)

    agg["d_over_T"] = agg["draft_size_B"] / agg["target_size_B"]

    if "wall_clock_speedup" not in agg.columns or agg["wall_clock_speedup"].isna().all():
        for idx, row in agg.iterrows():
            a, g = row["acceptance_rate_mean"], int(row["gamma"])
            c = row["d_over_T"]
            e = expected_accepted(a, g)
            agg.at[idx, "wall_clock_speedup"] = (e + 1) / (g * c + 1)

    fig, ax = plt.subplots(figsize=(8, 6))

    sc = ax.scatter(
        agg["acceptance_rate_mean"], agg["wall_clock_speedup"],
        c=agg["d_over_T"], cmap="viridis", s=35, alpha=0.7,
        edgecolors="k", linewidths=0.3,
    )
    plt.colorbar(sc, ax=ax, label="$d / T$")

    for pk, p in optimal_params.items():
        opt_g = p["optimal_gamma"]
        a = p["base_alpha"]
        c = p["c_ratio"]
        e = expected_accepted(a, opt_g)
        sp = (e + 1) / (opt_g * c + 1)
        ax.plot(a, sp, "r*", markersize=12, zorder=10)

    ax.set_xlabel("Acceptance rate $\\alpha$")
    ax.set_ylabel("Speedup")
    ax.set_title("Speedup vs Acceptance Rate  (★ = optimal γ)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save_fig(fig, out_dir / "pareto_frontier.pdf")


def _save_fig(fig, path: Path):
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


# ======================================================================
# Main
# ======================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Scaling law analysis for speculative decoding",
    )
    p.add_argument("--results_dir", type=str, default="results/sweep",
                   help="Directory containing sweep JSON files")
    p.add_argument("--output_dir", type=str, default="results/analysis",
                   help="Directory for analysis outputs")
    p.add_argument("--figure_dir", type=str, default=None,
                   help="Figure output directory (default: <output_dir>/figures)")
    return p


def main():
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.figure_dir) if args.figure_dir else out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load data ----
    logger.info("=" * 60)
    logger.info("Phase 1: Loading sweep results from %s", args.results_dir)
    logger.info("=" * 60)
    df = load_sweep_results(args.results_dir)

    # ---- 2. Fit scaling law per dataset ----
    logger.info("=" * 60)
    logger.info("Phase 2: Fitting α(d/T) = 1 - C·(d/T)^(-β)")
    logger.info("=" * 60)
    scaling_params = fit_scaling_law(df)

    # ---- 3. Position-dependent acceptance ----
    logger.info("=" * 60)
    logger.info("Phase 3: Fitting position-dependent acceptance (geometric model)")
    logger.info("=" * 60)
    position_params = fit_position_dependent(df)

    # ---- 4. Optimal gamma ----
    logger.info("=" * 60)
    logger.info("Phase 4: Computing optimal γ* per pair")
    logger.info("=" * 60)
    optimal_params = compute_optimal_gamma(position_params)

    # ---- 5. Throughput predictions ----
    logger.info("=" * 60)
    logger.info("Phase 5: Theoretical vs measured throughput")
    logger.info("=" * 60)
    pred_df = compute_throughput_predictions(df, scaling_params)

    # ---- 6. Save results ----
    logger.info("=" * 60)
    logger.info("Saving results to %s", out_dir)
    logger.info("=" * 60)

    all_results = dict(
        scaling_law_params=scaling_params,
        position_dependent_params=position_params,
        optimal_gamma=optimal_params,
    )
    with open(out_dir / "analysis_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    if not pred_df.empty:
        pred_df.to_csv(out_dir / "throughput_predictions.csv", index=False)

    summary = aggregate_across_seeds(df)
    summary.to_csv(out_dir / "summary_table.csv", index=False)
    logger.info("Saved summary_table.csv (%d rows)", len(summary))

    # ---- 7. Figures ----
    logger.info("=" * 60)
    logger.info("Generating figures → %s", fig_dir)
    logger.info("=" * 60)

    plot_scaling_law_fit(df, scaling_params, fig_dir)
    plot_position_decay(position_params, fig_dir)
    plot_optimal_gamma(optimal_params, fig_dir)
    plot_throughput_prediction(pred_df, fig_dir)
    plot_pareto_frontier(df, optimal_params, fig_dir)

    logger.info("=" * 60)
    logger.info("Analysis complete.")
    logger.info("  Results : %s", out_dir)
    logger.info("  Figures : %s", fig_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
