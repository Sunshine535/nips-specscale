#!/usr/bin/env python3
"""
Acceptance rate measurement sweep for SpecDenoise.

Sweeps over (draft, target) model pairs, γ values, timestep bins,
guidance scales, and random seeds to produce the data needed for
scaling-law fitting and figure generation.

Usage:
    python scripts/run_acceptance_sweep.py \
        --draft_model DiT-S/2 --target_model DiT-XL/2 \
        --gamma 5 --num_samples 100 --seed 0

    # Run the full sweep (all configs):
    python scripts/run_acceptance_sweep.py --full_sweep
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.speculative_denoise import NoiseSchedule, SpeculativeDenoiser
from src.dit_loader import load_dit_models, MODEL_CONFIGS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("acceptance_sweep")


def parse_args():
    parser = argparse.ArgumentParser(description="Acceptance rate sweep for SpecDenoise")
    parser.add_argument("--draft_model", type=str, default="DiT-S/2",
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--target_model", type=str, default="DiT-XL/2",
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--gamma", type=int, default=None,
                        help="Single γ value. If omitted, sweep all from config.")
    parser.add_argument("--guidance_scale", type=float, default=None,
                        help="Single guidance scale. If omitted, sweep all from config.")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None,
                        help="Single seed. If omitted, sweep all from config.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--noise_schedule", type=str, default="linear",
                        choices=["linear", "cosine", "shifted_cosine"])
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="results/acceptance_sweep")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--full_sweep", action="store_true",
                        help="Run the complete sweep over all model pairs, γ, guidance, seeds")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--draft_device", type=str, default="cuda:0")
    parser.add_argument("--target_device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--worker_id", type=int, default=0,
                        help="Worker index for multi-GPU parallel (0-based)")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Total number of parallel workers")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip configs whose result file already exists")
    return parser.parse_args()


def load_config(path: str) -> dict:
    cfg_path = PROJECT_ROOT / path
    if cfg_path.exists():
        with open(cfg_path) as f:
            return yaml.safe_load(f)
    logger.warning("Config %s not found, using defaults", cfg_path)
    return {}


def get_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16,
            "bfloat16": torch.bfloat16}[name]


def measure_acceptance_rate(
    denoiser: SpeculativeDenoiser,
    gamma: int,
    guidance_scale: float,
    num_samples: int,
    batch_size: int,
    num_inference_steps: int,
    temperature: float,
    seed: int,
    image_size: int,
    latent_channels: int = 4,
    patch_size: int = 2,
    num_classes: int = 1000,
) -> dict:
    """Run speculative denoising and collect acceptance statistics."""
    device = denoiser.target_device
    latent_size = image_size // 8  # VAE downsamples 8×
    shape = (batch_size, latent_channels, latent_size, latent_size)

    all_acceptance_rates = []
    all_acceptance_per_timestep = []
    all_nfe = []
    all_wall_times = []
    total_generated = 0

    gen = torch.Generator(device=device)

    n_batches = (num_samples + batch_size - 1) // batch_size
    logger.info("  Generating %d samples in %d batches (γ=%d, cfg=%.1f, seed=%d)",
                num_samples, n_batches, gamma, guidance_scale, seed)

    for batch_idx in range(n_batches):
        cur_bs = min(batch_size, num_samples - total_generated)
        cur_shape = (cur_bs,) + shape[1:]

        gen.manual_seed(seed * 100000 + batch_idx)
        class_labels = torch.randint(0, num_classes, (cur_bs,), device=device)

        try:
            output = denoiser.generate(
                shape=cur_shape,
                num_inference_steps=num_inference_steps,
                gamma=gamma,
                guidance_scale=guidance_scale,
                class_labels=class_labels,
                temperature=temperature,
                generator=gen,
            )
            all_acceptance_rates.append(output.acceptance_rate)
            all_acceptance_per_timestep.append(output.acceptance_rate_per_timestep)
            all_nfe.append(output.num_function_evals)
            all_wall_times.append(output.wall_time_seconds)

        except torch.cuda.OutOfMemoryError:
            logger.warning("  OOM at batch %d (bs=%d, γ=%d). Skipping.", batch_idx, cur_bs, gamma)
            torch.cuda.empty_cache()
            continue

        total_generated += cur_bs
        if (batch_idx + 1) % max(1, n_batches // 5) == 0:
            logger.info("    Batch %d/%d done (%.1f%%)", batch_idx + 1, n_batches,
                        100 * (batch_idx + 1) / n_batches)

    if not all_acceptance_rates:
        return {"error": "all_batches_oom"}

    avg_acceptance = sum(all_acceptance_rates) / len(all_acceptance_rates)
    avg_nfe = sum(all_nfe) / len(all_nfe)
    avg_wall = sum(all_wall_times) / len(all_wall_times)

    # Average per-timestep acceptance across batches
    n_bins = max(len(a) for a in all_acceptance_per_timestep) if all_acceptance_per_timestep else 0
    avg_per_t = []
    for b in range(n_bins):
        vals = [a[b] for a in all_acceptance_per_timestep if b < len(a)]
        avg_per_t.append(sum(vals) / len(vals) if vals else 0.0)

    return {
        "acceptance_rate": avg_acceptance,
        "acceptance_rate_per_timestep": avg_per_t,
        "avg_nfe": avg_nfe,
        "avg_wall_time": avg_wall,
        "total_generated": total_generated,
        "num_batches": len(all_acceptance_rates),
    }


def run_single_config(
    draft_name: str,
    target_name: str,
    gamma: int,
    guidance_scale: float,
    seed: int,
    args,
) -> dict:
    """Run acceptance measurement for a single (model pair, γ, cfg, seed)."""
    dtype = get_dtype(args.dtype)
    pretrained = args.pretrained and not args.no_pretrained

    logger.info("Loading models: %s → %s (pretrained=%s)", draft_name, target_name, pretrained)
    try:
        draft_model, target_model, info = load_dit_models(
            draft_name=draft_name,
            target_name=target_name,
            image_size=args.image_size,
            draft_device=args.draft_device,
            target_device=args.target_device,
            dtype=dtype,
            pretrained=pretrained,
        )
    except Exception as e:
        logger.error("Failed to load models: %s", e)
        return {"error": str(e)}

    schedule = NoiseSchedule(
        num_timesteps=1000,
        schedule_type=args.noise_schedule,
    )

    denoiser = SpeculativeDenoiser(
        draft_model=draft_model,
        target_model=target_model,
        noise_schedule=schedule,
        draft_device=torch.device(args.draft_device),
        target_device=torch.device(args.target_device),
    )

    result = measure_acceptance_rate(
        denoiser=denoiser,
        gamma=gamma,
        guidance_scale=guidance_scale,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_inference_steps=args.num_inference_steps,
        temperature=args.temperature,
        seed=seed,
        image_size=args.image_size,
    )

    result.update({
        "draft_model": draft_name,
        "target_model": target_name,
        "draft_params_M": info.get("draft_params_M", 0),
        "target_params_M": info.get("target_params_M", 0),
        "gamma": gamma,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "noise_schedule": args.noise_schedule,
        "num_inference_steps": args.num_inference_steps,
        "temperature": args.temperature,
    })

    del draft_model, target_model, denoiser
    torch.cuda.empty_cache()

    return result


def save_result(result: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    if "error" in result and "draft_model" not in result:
        tag = f"error__{int(time.time())}"
    else:
        tag = (
            f"{result['draft_model'].replace('/', '_')}"
            f"__{result['target_model'].replace('/', '_')}"
            f"__g{result['gamma']}"
            f"__cfg{result['guidance_scale']}"
            f"__seed{result['seed']}"
        )
    path = output_dir / f"{tag}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved: %s", path)


def _result_file_exists(result_meta: dict, output_dir: Path) -> bool:
    """Check if a result file already exists for this config."""
    tag = (
        f"{result_meta['draft'].replace('/', '_')}"
        f"__{result_meta['target'].replace('/', '_')}"
        f"__g{result_meta['gamma']}"
        f"__cfg{result_meta['guidance_scale']}"
        f"__seed{result_meta['seed']}"
    )
    return (output_dir / f"{tag}.json").exists()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    sweep_cfg = cfg.get("acceptance_sweep", {})

    output_dir = PROJECT_ROOT / args.output_dir

    if args.full_sweep:
        pairs = cfg.get("models", {}).get("pairs", [
            {"draft": "DiT-S/2", "target": "DiT-XL/2"},
            {"draft": "DiT-B/2", "target": "DiT-XL/2"},
            {"draft": "DiT-S/2", "target": "DiT-B/2"},
        ])
        gammas = sweep_cfg.get("gamma_values", [2, 3, 5, 7, 10])
        cfgs = sweep_cfg.get("guidance_scales", [1.0, 4.0, 7.5])
        seeds = sweep_cfg.get("seeds", [0, 1, 2])
    else:
        pairs = [{"draft": args.draft_model, "target": args.target_model}]
        gammas = [args.gamma] if args.gamma else sweep_cfg.get("gamma_values", [2, 3, 5, 7, 10])
        cfgs = [args.guidance_scale] if args.guidance_scale else sweep_cfg.get("guidance_scales", [1.0, 4.0, 7.5])
        seeds = [args.seed] if args.seed is not None else sweep_cfg.get("seeds", [0, 1, 2])

    all_configs = []
    for pair in pairs:
        for gamma in gammas:
            for guidance_scale in cfgs:
                for seed in seeds:
                    all_configs.append({
                        "draft": pair["draft"], "target": pair["target"],
                        "gamma": gamma, "guidance_scale": guidance_scale, "seed": seed,
                    })

    total_configs = len(all_configs)

    # Multi-GPU: each worker takes a strided subset
    my_configs = [c for i, c in enumerate(all_configs) if i % args.num_workers == args.worker_id]

    logger.info("=" * 60)
    logger.info("Acceptance Rate Sweep (worker %d/%d)", args.worker_id, args.num_workers)
    logger.info("  Total configs: %d | This worker: %d", total_configs, len(my_configs))
    logger.info("  γ values: %s", gammas)
    logger.info("  Guidance scales: %s", cfgs)
    logger.info("  Seeds: %s", seeds)
    logger.info("  Samples per config: %d", args.num_samples)
    logger.info("  Device: draft=%s target=%s", args.draft_device, args.target_device)
    logger.info("  Skip existing: %s", args.skip_existing)
    logger.info("=" * 60)

    all_results = []
    done = 0
    skipped = 0
    sweep_start = time.time()

    for c in my_configs:
        done += 1

        if args.skip_existing and _result_file_exists(c, output_dir):
            skipped += 1
            logger.info("[%d/%d] SKIP (exists) %s→%s γ=%d cfg=%.1f seed=%d",
                        done, len(my_configs), c["draft"], c["target"],
                        c["gamma"], c["guidance_scale"], c["seed"])
            continue

        logger.info(
            "[%d/%d] %s→%s  γ=%d  cfg=%.1f  seed=%d",
            done, len(my_configs),
            c["draft"], c["target"],
            c["gamma"], c["guidance_scale"], c["seed"],
        )

        result = run_single_config(
            draft_name=c["draft"],
            target_name=c["target"],
            gamma=c["gamma"],
            guidance_scale=c["guidance_scale"],
            seed=c["seed"],
            args=args,
        )

        save_result(result, output_dir)
        all_results.append(result)

        elapsed = time.time() - sweep_start
        actual_done = done - skipped
        if actual_done > 0:
            remaining = len(my_configs) - done
            eta = elapsed / actual_done * remaining
        else:
            eta = 0
        logger.info(
            "  → α=%.3f | Elapsed: %.0fs | ETA: %.0fs",
            result.get("acceptance_rate", 0), elapsed, eta,
        )

    summary_path = output_dir / f"sweep_summary_worker{args.worker_id}.json"
    summary = {
        "worker_id": args.worker_id,
        "num_workers": args.num_workers,
        "total_configs": total_configs,
        "assigned": len(my_configs),
        "completed": len(all_results),
        "skipped": skipped,
        "errors": sum(1 for r in all_results if "error" in r),
        "total_time_seconds": time.time() - sweep_start,
        "results": all_results,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Worker %d complete. Summary: %s", args.worker_id, summary_path)

    if all_results:
        rates = [r.get("acceptance_rate", 0) for r in all_results if "error" not in r]
        if rates:
            logger.info("Acceptance rate: mean=%.3f, min=%.3f, max=%.3f",
                        sum(rates) / len(rates), min(rates), max(rates))


if __name__ == "__main__":
    main()
