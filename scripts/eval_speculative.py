#!/usr/bin/env python3
"""Detailed evaluation of a single draft–target pair.

Produces:
  - Per-position acceptance rates for gamma = 1..10
  - Geometric-decay hypothesis test (alpha_k vs alpha_1^k)
  - KL divergence estimation between draft and target
  - Summary JSON written to output_dir

Usage:
    python scripts/eval_speculative.py \
        --draft_model Qwen/Qwen3.5-0.8B \
        --target_model Qwen/Qwen3.5-27B \
        --dataset gsm8k \
        --num_samples 30 \
        --output_dir results/eval/
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict

import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.benchmark_speculative import load_models, load_datasets
from src.speculative_decode import SpeculativeDecoder

logger = logging.getLogger("eval_speculative")


# ======================================================================
# Position-dependent analysis
# ======================================================================


def evaluate_position_acceptance(
    decoder: SpeculativeDecoder,
    prompts: List[str],
    gamma_values: List[int],
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    num_trials: int = 5,
) -> Dict[str, dict]:
    """For each gamma, run multiple trials and record per-position acceptance."""
    tokenizer = decoder.tokenizer
    results = {}

    for gamma in gamma_values:
        logger.info("Evaluating gamma=%d across %d prompts × %d trials", gamma, len(prompts), num_trials)
        all_pos_rates: List[List[float]] = []
        all_accept: List[float] = []
        all_tpr: List[float] = []

        for prompt in prompts:
            ids = tokenizer(prompt, return_tensors="pt").input_ids
            stats = decoder.analyse_acceptance(
                ids,
                max_new_tokens=max_new_tokens,
                gamma=gamma,
                temperature=temperature,
                num_trials=num_trials,
            )
            all_accept.append(stats["mean_acceptance_rate"])
            all_tpr.append(stats["mean_tokens_per_round"])
            all_pos_rates.append(stats["position_acceptance_mean"])

        pos_arr = np.array(all_pos_rates)
        results[str(gamma)] = {
            "gamma": gamma,
            "acceptance_rate_mean": float(np.mean(all_accept)),
            "acceptance_rate_std": float(np.std(all_accept)),
            "tokens_per_round_mean": float(np.mean(all_tpr)),
            "position_acceptance_mean": pos_arr.mean(axis=0).tolist(),
            "position_acceptance_std": pos_arr.std(axis=0).tolist(),
        }

    return results


def test_geometric_decay(pos_results: Dict[str, dict]) -> Dict[str, dict]:
    """Test whether alpha_k ≈ alpha_1^k (geometric decay hypothesis).

    Returns R² and RMSE for each gamma value.
    """
    from sklearn.metrics import r2_score

    geo_results = {}
    for key, data in pos_results.items():
        gamma = data["gamma"]
        pos_means = np.array(data["position_acceptance_mean"][:gamma])
        if len(pos_means) < 2 or pos_means[0] <= 0:
            continue

        alpha_1 = pos_means[0]
        positions = np.arange(1, gamma + 1)
        predicted = alpha_1 ** positions

        actual = pos_means[: len(predicted)]
        if len(actual) != len(predicted):
            predicted = predicted[: len(actual)]

        r2 = float(r2_score(actual, predicted)) if len(actual) >= 2 else 0.0
        rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))

        geo_results[key] = {
            "gamma": gamma,
            "alpha_1": float(alpha_1),
            "R2_geometric": r2,
            "RMSE_geometric": rmse,
            "actual_rates": actual.tolist(),
            "predicted_geometric": predicted.tolist(),
        }
        logger.info(
            "Geometric decay test γ=%d: α₁=%.4f  R²=%.4f  RMSE=%.4f",
            gamma, alpha_1, r2, rmse,
        )

    return geo_results


# ======================================================================
# KL divergence
# ======================================================================


def evaluate_kl_divergence(
    decoder: SpeculativeDecoder,
    prompts: List[str],
    max_positions: int = 256,
) -> dict:
    """Average KL(target || draft) across prompts."""
    tokenizer = decoder.tokenizer
    all_kl_mean = []
    all_kl_std = []

    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt").input_ids
        kl = decoder.estimate_kl_divergence(ids, max_positions=max_positions)
        all_kl_mean.append(kl["kl_mean"])
        all_kl_std.append(kl["kl_std"])

    return {
        "kl_mean_across_prompts": float(np.mean(all_kl_mean)),
        "kl_std_across_prompts": float(np.std(all_kl_mean)),
        "kl_per_prompt_means": all_kl_mean,
    }


# ======================================================================
# Plotting
# ======================================================================


def plot_position_acceptance(
    pos_results: Dict[str, dict],
    geo_results: Dict[str, dict],
    out_dir: Path,
    pair_label: str,
):
    """Plot per-position acceptance rates + geometric prediction overlay."""
    plt.rcParams.update({"font.size": 12, "figure.dpi": 150})

    gamma_keys = sorted(pos_results.keys(), key=lambda k: int(k))
    n = len(gamma_keys)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows), squeeze=False)

    for idx, key in enumerate(gamma_keys):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        data = pos_results[key]
        gamma = data["gamma"]
        means = np.array(data["position_acceptance_mean"][:gamma])
        stds = np.array(data["position_acceptance_std"][:gamma])
        positions = np.arange(1, gamma + 1)

        ax.bar(positions, means, yerr=stds, capsize=3, alpha=0.7, label="Empirical")

        if key in geo_results:
            pred = geo_results[key]["predicted_geometric"]
            ax.plot(
                positions[: len(pred)],
                pred,
                "r--o",
                markersize=4,
                linewidth=1.5,
                label=f"Geometric (R²={geo_results[key]['R2_geometric']:.3f})",
            )

        ax.set_xlabel("Position k")
        ax.set_ylabel("Acceptance rate αₖ")
        ax.set_title(f"γ = {gamma}")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    fig.suptitle(f"Position-dependent acceptance: {pair_label}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = out_dir / "position_acceptance.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_kl_distribution(kl_info: dict, out_dir: Path, pair_label: str):
    """Histogram of per-prompt KL divergence."""
    plt.rcParams.update({"font.size": 12, "figure.dpi": 150})
    fig, ax = plt.subplots(figsize=(6, 4))

    vals = kl_info["kl_per_prompt_means"]
    ax.hist(vals, bins=20, edgecolor="black", alpha=0.7)
    ax.axvline(
        kl_info["kl_mean_across_prompts"],
        color="red",
        linestyle="--",
        label=f"Mean={kl_info['kl_mean_across_prompts']:.4f}",
    )
    ax.set_xlabel("KL(target ‖ draft)")
    ax.set_ylabel("Count")
    ax.set_title(f"KL Divergence Distribution: {pair_label}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = out_dir / "kl_divergence.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


# ======================================================================
# Main
# ======================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Detailed speculative decoding evaluation")
    p.add_argument("--draft_model", type=str, required=True)
    p.add_argument("--target_model", type=str, required=True)
    p.add_argument("--dataset", type=str, default="gsm8k")
    p.add_argument("--num_samples", type=int, default=30)
    p.add_argument("--gamma_values", type=int, nargs="+", default=[1, 2, 3, 4, 5, 7, 10])
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--num_trials", type=int, default=5)
    p.add_argument("--output_dir", type=str, default="results/eval")
    p.add_argument("--draft_gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p


def main():
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    draft_short = args.draft_model.split("/")[-1]
    target_short = args.target_model.split("/")[-1]
    pair_label = f"{draft_short} → {target_short}"

    out_dir = Path(args.output_dir) / f"{draft_short}__{target_short}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    draft_model, target_model, tokenizer = load_models(
        args.draft_model, args.target_model, draft_gpu=args.draft_gpu
    )
    decoder = SpeculativeDecoder(draft_model, target_model, tokenizer)

    # Load dataset
    datasets = load_datasets([args.dataset], args.num_samples)
    prompts = datasets.get(args.dataset, [])
    if not prompts:
        logger.error("No prompts loaded for dataset=%s", args.dataset)
        return

    # 1. Position-dependent acceptance rates
    logger.info("=== Position-dependent acceptance analysis ===")
    pos_results = evaluate_position_acceptance(
        decoder,
        prompts,
        gamma_values=args.gamma_values,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        num_trials=args.num_trials,
    )

    # 2. Geometric decay test
    logger.info("=== Geometric decay hypothesis test ===")
    try:
        geo_results = test_geometric_decay(pos_results)
    except ImportError:
        logger.warning("scikit-learn not installed — skipping geometric decay test")
        geo_results = {}

    # 3. KL divergence
    logger.info("=== KL divergence estimation ===")
    kl_info = evaluate_kl_divergence(decoder, prompts)

    # 4. Save all results
    combined = {
        "draft_model": args.draft_model,
        "target_model": args.target_model,
        "dataset": args.dataset,
        "num_samples": args.num_samples,
        "position_acceptance": pos_results,
        "geometric_decay_test": geo_results,
        "kl_divergence": kl_info,
    }
    json_path = out_dir / "eval_results.json"
    with open(json_path, "w") as f:
        json.dump(combined, f, indent=2)
    logger.info("Saved results → %s", json_path)

    # 5. Plots
    plot_position_acceptance(pos_results, geo_results, out_dir, pair_label)
    plot_kl_distribution(kl_info, out_dir, pair_label)

    logger.info("Evaluation complete for %s", pair_label)


if __name__ == "__main__":
    main()
