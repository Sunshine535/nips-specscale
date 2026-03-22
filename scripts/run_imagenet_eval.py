#!/usr/bin/env python3
"""
End-to-end ImageNet 256×256 class-conditional evaluation for SpecDenoise.

Generates 50K images with baseline solvers and SpecDenoise configurations,
computes FID-50K, Inception Score, wall-clock time, NFE, and GPU memory.

Usage:
    python scripts/run_imagenet_eval.py --method specdenoise --gamma 5 --num_images 50000
    python scripts/run_imagenet_eval.py --method baseline --solver ddim --num_steps 50
    python scripts/run_imagenet_eval.py --full_eval
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
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
logger = logging.getLogger("imagenet_eval")


def parse_args():
    parser = argparse.ArgumentParser(description="ImageNet class-conditional evaluation")
    parser.add_argument("--method", type=str, default="specdenoise",
                        choices=["baseline", "specdenoise"])
    parser.add_argument("--draft_model", type=str, default="DiT-S/2")
    parser.add_argument("--target_model", type=str, default="DiT-XL/2")
    parser.add_argument("--solver", type=str, default="ddim",
                        choices=["ddim", "dpmsolver++"])
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--gamma", type=int, default=5)
    parser.add_argument("--adaptive_gamma", action="store_true")
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--noise_schedule", type=str, default="linear")
    parser.add_argument("--num_images", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/imagenet_eval")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--full_eval", action="store_true",
                        help="Run all baselines + SpecDenoise configs")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--draft_device", type=str, default="cuda:0")
    parser.add_argument("--target_device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--fid_reference", type=str, default="",
                        help="Path to FID reference stats (npz or directory)")
    parser.add_argument("--save_samples", action="store_true",
                        help="Save generated sample tensors to disk")
    parser.add_argument("--compute_fid", action="store_true", default=True)
    parser.add_argument("--no_fid", action="store_true")
    return parser.parse_args()


def get_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16,
            "bfloat16": torch.bfloat16}[name]


def load_config(path: str) -> dict:
    cfg_path = PROJECT_ROOT / path
    if cfg_path.exists():
        with open(cfg_path) as f:
            return yaml.safe_load(f)
    return {}


def compute_fid_score(
    generated_samples: np.ndarray,
    reference: str,
    batch_size: int = 64,
) -> Optional[float]:
    """Compute FID score using clean-fid or torchmetrics."""
    # Try clean-fid first
    try:
        from cleanfid import fid as cleanfid
        import tempfile
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(len(generated_samples)):
                img = generated_samples[i]
                if img.max() <= 1.0:
                    img = (img * 255).clip(0, 255).astype(np.uint8)
                if img.shape[0] in (1, 3, 4):
                    img = img.transpose(1, 2, 0)
                if img.shape[-1] == 1:
                    img = img.squeeze(-1)
                Image.fromarray(img).save(os.path.join(tmpdir, f"{i:06d}.png"))

            if reference and os.path.isdir(reference):
                score = cleanfid.compute_fid(tmpdir, reference, batch_size=batch_size)
            else:
                score = cleanfid.compute_fid(tmpdir, dataset_name="imagenet_256",
                                              dataset_split="custom", batch_size=batch_size)
        logger.info("FID (clean-fid): %.2f", score)
        return score

    except ImportError:
        logger.info("clean-fid not available, trying torchmetrics...")

    except Exception as e:
        logger.warning("clean-fid failed: %s, trying torchmetrics...", e)

    # Fallback to torchmetrics
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance

        fid_metric = FrechetInceptionDistance(feature=2048, normalize=True)
        fid_metric = fid_metric.to("cuda" if torch.cuda.is_available() else "cpu")

        gen_tensor = torch.from_numpy(generated_samples).float()
        if gen_tensor.max() > 1.0:
            gen_tensor = gen_tensor / 255.0
        if gen_tensor.shape[1] not in (1, 3):
            gen_tensor = gen_tensor.permute(0, 3, 1, 2)

        for i in range(0, len(gen_tensor), batch_size):
            batch = gen_tensor[i:i + batch_size].to(fid_metric.device)
            if batch.shape[1] == 1:
                batch = batch.repeat(1, 3, 1, 1)
            fid_metric.update(batch, real=False)

        logger.warning("torchmetrics FID requires real image stats. "
                       "Returning partial metric (generated-only update).")
        return None

    except ImportError:
        logger.error("Neither clean-fid nor torchmetrics available for FID computation")
        return None


def compute_inception_score(
    generated_samples: np.ndarray,
    batch_size: int = 64,
    splits: int = 10,
) -> Optional[Tuple[float, float]]:
    """Compute Inception Score."""
    try:
        from torchmetrics.image.inception import InceptionScore

        is_metric = InceptionScore(normalize=True, splits=splits)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        is_metric = is_metric.to(device)

        gen_tensor = torch.from_numpy(generated_samples).float()
        if gen_tensor.max() > 1.0:
            gen_tensor = gen_tensor / 255.0
        if gen_tensor.shape[1] not in (1, 3):
            gen_tensor = gen_tensor.permute(0, 3, 1, 2)

        for i in range(0, len(gen_tensor), batch_size):
            batch = gen_tensor[i:i + batch_size].to(device)
            if batch.shape[1] == 1:
                batch = batch.repeat(1, 3, 1, 1)
            is_metric.update(batch)

        mean, std = is_metric.compute()
        logger.info("Inception Score: %.2f ± %.2f", mean.item(), std.item())
        return mean.item(), std.item()

    except ImportError:
        logger.warning("torchmetrics not available for IS computation")
        return None


def get_gpu_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def generate_baseline(
    target_model,
    schedule: NoiseSchedule,
    num_images: int,
    batch_size: int,
    num_steps: int,
    guidance_scale: float,
    image_size: int,
    num_classes: int,
    seed: int,
    device: torch.device,
) -> Dict:
    """Generate images with baseline DDIM (target model only)."""
    latent_size = image_size // 8
    shape = (batch_size, 4, latent_size, latent_size)

    dummy_draft = target_model  # unused for baseline
    denoiser = SpeculativeDenoiser(
        draft_model=dummy_draft,
        target_model=target_model,
        noise_schedule=schedule,
        draft_device=device,
        target_device=device,
    )

    all_samples = []
    total_nfe = 0
    total_wall = 0.0
    gen = torch.Generator(device=device)

    n_batches = (num_images + batch_size - 1) // batch_size
    logger.info("Baseline generation: %d images in %d batches (steps=%d)",
                num_images, n_batches, num_steps)

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    for batch_idx in range(n_batches):
        cur_bs = min(batch_size, num_images - len(all_samples) * batch_size)
        if cur_bs <= 0:
            break
        cur_shape = (cur_bs, shape[1], shape[2], shape[3])

        gen.manual_seed(seed * 100000 + batch_idx)
        class_labels = torch.randint(0, num_classes, (cur_bs,), device=device)

        try:
            samples, wall, nfe = denoiser.generate_baseline(
                shape=cur_shape,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                class_labels=class_labels,
                generator=gen,
            )
            all_samples.append(samples.cpu().numpy())
            total_nfe += nfe
            total_wall += wall
        except torch.cuda.OutOfMemoryError:
            logger.warning("OOM at batch %d, reducing effective count", batch_idx)
            torch.cuda.empty_cache()
            continue

        if (batch_idx + 1) % max(1, n_batches // 10) == 0:
            logger.info("  Batch %d/%d (%.0f%%)", batch_idx + 1, n_batches,
                        100 * (batch_idx + 1) / n_batches)

    peak_mem = get_gpu_memory_mb()
    if all_samples:
        all_samples_np = np.concatenate(all_samples, axis=0)[:num_images]
    else:
        all_samples_np = np.array([])

    return {
        "samples": all_samples_np,
        "total_nfe": total_nfe,
        "total_wall_time": total_wall,
        "avg_wall_per_image": total_wall / max(len(all_samples_np), 1),
        "peak_gpu_memory_mb": peak_mem,
        "num_generated": len(all_samples_np),
    }


def generate_specdenoise(
    draft_model,
    target_model,
    schedule: NoiseSchedule,
    num_images: int,
    batch_size: int,
    num_steps: int,
    gamma: int,
    adaptive_gamma: bool,
    guidance_scale: float,
    temperature: float,
    image_size: int,
    num_classes: int,
    seed: int,
    draft_device: torch.device,
    target_device: torch.device,
) -> Dict:
    """Generate images with SpecDenoise."""
    latent_size = image_size // 8
    shape = (batch_size, 4, latent_size, latent_size)

    denoiser = SpeculativeDenoiser(
        draft_model=draft_model,
        target_model=target_model,
        noise_schedule=schedule,
        draft_device=draft_device,
        target_device=target_device,
    )

    all_samples = []
    total_nfe = 0
    total_wall = 0.0
    total_accepted = 0
    total_draft_steps = 0
    gen = torch.Generator(device=target_device)

    n_batches = (num_images + batch_size - 1) // batch_size
    method_name = f"SpecDenoise(γ={'adaptive' if adaptive_gamma else gamma})"
    logger.info("%s: %d images in %d batches", method_name, num_images, n_batches)

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    for batch_idx in range(n_batches):
        cur_bs = min(batch_size, num_images - len(all_samples) * batch_size)
        if cur_bs <= 0:
            break
        cur_shape = (cur_bs, shape[1], shape[2], shape[3])

        gen.manual_seed(seed * 100000 + batch_idx)
        class_labels = torch.randint(0, num_classes, (cur_bs,), device=target_device)

        try:
            output = denoiser.generate(
                shape=cur_shape,
                num_inference_steps=num_steps,
                gamma=gamma,
                guidance_scale=guidance_scale,
                class_labels=class_labels,
                temperature=temperature,
                adaptive_gamma=adaptive_gamma,
                generator=gen,
            )
            all_samples.append(output.samples.cpu().numpy())
            total_nfe += output.num_function_evals
            total_wall += output.wall_time_seconds
            total_accepted += output.total_accepted_steps
            total_draft_steps += output.total_draft_steps
        except torch.cuda.OutOfMemoryError:
            logger.warning("OOM at batch %d, skipping", batch_idx)
            torch.cuda.empty_cache()
            continue

        if (batch_idx + 1) % max(1, n_batches // 10) == 0:
            logger.info("  Batch %d/%d (%.0f%%) α=%.3f",
                        batch_idx + 1, n_batches,
                        100 * (batch_idx + 1) / n_batches,
                        total_accepted / max(total_draft_steps, 1))

    peak_mem = get_gpu_memory_mb()
    if all_samples:
        all_samples_np = np.concatenate(all_samples, axis=0)[:num_images]
    else:
        all_samples_np = np.array([])

    return {
        "samples": all_samples_np,
        "total_nfe": total_nfe,
        "total_wall_time": total_wall,
        "avg_wall_per_image": total_wall / max(len(all_samples_np), 1),
        "peak_gpu_memory_mb": peak_mem,
        "num_generated": len(all_samples_np),
        "acceptance_rate": total_accepted / max(total_draft_steps, 1),
        "total_accepted": total_accepted,
        "total_draft_steps": total_draft_steps,
    }


def run_evaluation(
    run_name: str,
    gen_result: Dict,
    fid_reference: str,
    compute_fid_flag: bool,
    output_dir: Path,
    save_samples: bool,
) -> dict:
    """Compute metrics and save results for a single generation run."""
    samples = gen_result.pop("samples")
    metrics = dict(gen_result)
    metrics["run_name"] = run_name

    if len(samples) > 0 and compute_fid_flag:
        fid = compute_fid_score(samples, fid_reference)
        metrics["fid_50k"] = fid

        is_result = compute_inception_score(samples)
        if is_result:
            metrics["inception_score_mean"] = is_result[0]
            metrics["inception_score_std"] = is_result[1]

    if save_samples and len(samples) > 0:
        sample_path = output_dir / f"{run_name}_samples.npz"
        np.savez_compressed(sample_path, samples=samples[:1000])
        logger.info("Saved %d sample tensors to %s", min(1000, len(samples)), sample_path)

    result_path = output_dir / f"{run_name}_metrics.json"
    serializable = {k: v for k, v in metrics.items() if isinstance(v, (int, float, str, bool, type(None)))}
    with open(result_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info("Saved metrics: %s", result_path)

    return metrics


def run_full_eval(args, cfg: dict):
    """Run all baseline and SpecDenoise configurations."""
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = get_dtype(args.dtype)
    pretrained = args.pretrained and not args.no_pretrained
    device = torch.device(args.target_device)
    compute_fid_flag = args.compute_fid and not args.no_fid

    eval_cfg = cfg.get("imagenet_eval", {})

    # -- Baselines --
    baselines = eval_cfg.get("baselines", [
        {"name": "DDIM-50", "solver": "ddim", "num_steps": 50},
        {"name": "DDIM-25", "solver": "ddim", "num_steps": 25},
        {"name": "DPM-Solver++-20", "solver": "dpmsolver++", "num_steps": 20},
    ])

    logger.info("Loading target model for baselines...")
    _, target_model, info = load_dit_models(
        draft_name="DiT-S/2",
        target_name=args.target_model,
        image_size=args.image_size,
        draft_device=args.draft_device,
        target_device=args.target_device,
        dtype=dtype,
        pretrained=pretrained,
    )

    schedule = NoiseSchedule(num_timesteps=1000, schedule_type=args.noise_schedule)
    all_metrics = []

    for bl in baselines:
        logger.info("=" * 50)
        logger.info("Baseline: %s", bl["name"])
        gen = generate_baseline(
            target_model=target_model,
            schedule=schedule,
            num_images=args.num_images,
            batch_size=args.batch_size,
            num_steps=bl["num_steps"],
            guidance_scale=args.guidance_scale,
            image_size=args.image_size,
            num_classes=args.num_classes,
            seed=args.seed,
            device=device,
        )
        m = run_evaluation(bl["name"], gen, args.fid_reference, compute_fid_flag,
                           output_dir, args.save_samples)
        m["method"] = "baseline"
        m["solver"] = bl["solver"]
        m["num_steps"] = bl["num_steps"]
        all_metrics.append(m)

    # -- SpecDenoise variants --
    spec_configs = eval_cfg.get("specdenoise", [
        {"name": "SpecDenoise-fixed-γ5", "gamma": 5, "adaptive": False},
        {"name": "SpecDenoise-adaptive", "gamma": 10, "adaptive": True},
    ])

    logger.info("Loading draft model for SpecDenoise...")
    draft_model, target_model, info = load_dit_models(
        draft_name=args.draft_model,
        target_name=args.target_model,
        image_size=args.image_size,
        draft_device=args.draft_device,
        target_device=args.target_device,
        dtype=dtype,
        pretrained=pretrained,
    )

    for sc in spec_configs:
        logger.info("=" * 50)
        logger.info("SpecDenoise: %s", sc["name"])
        gen = generate_specdenoise(
            draft_model=draft_model,
            target_model=target_model,
            schedule=schedule,
            num_images=args.num_images,
            batch_size=args.batch_size,
            num_steps=args.num_steps,
            gamma=sc["gamma"],
            adaptive_gamma=sc["adaptive"],
            guidance_scale=args.guidance_scale,
            temperature=args.temperature,
            image_size=args.image_size,
            num_classes=args.num_classes,
            seed=args.seed,
            draft_device=torch.device(args.draft_device),
            target_device=torch.device(args.target_device),
        )
        m = run_evaluation(sc["name"], gen, args.fid_reference, compute_fid_flag,
                           output_dir, args.save_samples)
        m["method"] = "specdenoise"
        m["gamma"] = sc["gamma"]
        m["adaptive"] = sc["adaptive"]
        m["draft_model"] = args.draft_model
        m["target_model"] = args.target_model
        all_metrics.append(m)

    # -- Summary table --
    summary_path = output_dir / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info("Summary saved: %s", summary_path)

    print_comparison_table(all_metrics)
    return all_metrics


def run_single_eval(args, cfg: dict):
    """Run a single evaluation configuration."""
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = get_dtype(args.dtype)
    pretrained = args.pretrained and not args.no_pretrained
    compute_fid_flag = args.compute_fid and not args.no_fid

    schedule = NoiseSchedule(num_timesteps=1000, schedule_type=args.noise_schedule)

    if args.method == "baseline":
        _, target_model, info = load_dit_models(
            draft_name="DiT-S/2", target_name=args.target_model,
            image_size=args.image_size,
            draft_device=args.draft_device, target_device=args.target_device,
            dtype=dtype, pretrained=pretrained,
        )
        gen = generate_baseline(
            target_model=target_model, schedule=schedule,
            num_images=args.num_images, batch_size=args.batch_size,
            num_steps=args.num_steps, guidance_scale=args.guidance_scale,
            image_size=args.image_size, num_classes=args.num_classes,
            seed=args.seed, device=torch.device(args.target_device),
        )
        name = f"baseline_{args.solver}_{args.num_steps}"
    else:
        draft_model, target_model, info = load_dit_models(
            draft_name=args.draft_model, target_name=args.target_model,
            image_size=args.image_size,
            draft_device=args.draft_device, target_device=args.target_device,
            dtype=dtype, pretrained=pretrained,
        )
        gen = generate_specdenoise(
            draft_model=draft_model, target_model=target_model, schedule=schedule,
            num_images=args.num_images, batch_size=args.batch_size,
            num_steps=args.num_steps, gamma=args.gamma,
            adaptive_gamma=args.adaptive_gamma,
            guidance_scale=args.guidance_scale, temperature=args.temperature,
            image_size=args.image_size, num_classes=args.num_classes,
            seed=args.seed,
            draft_device=torch.device(args.draft_device),
            target_device=torch.device(args.target_device),
        )
        name = f"specdenoise_g{args.gamma}_{'adaptive' if args.adaptive_gamma else 'fixed'}"

    m = run_evaluation(name, gen, args.fid_reference, compute_fid_flag,
                       output_dir, args.save_samples)
    return m


def print_comparison_table(metrics_list: List[dict]):
    """Print a formatted comparison table."""
    logger.info("=" * 80)
    logger.info("%-30s %8s %8s %10s %10s %8s", "Method", "FID↓", "IS↑", "Wall(s)", "NFE", "Mem(MB)")
    logger.info("-" * 80)
    for m in metrics_list:
        fid = m.get("fid_50k", None)
        is_val = m.get("inception_score_mean", None)
        fid_str = f"{fid:.2f}" if fid is not None else "N/A"
        is_str = f"{is_val:.2f}" if is_val is not None else "N/A"
        logger.info(
            "%-30s %8s %8s %10.1f %10d %8.0f",
            m.get("run_name", "?"),
            fid_str, is_str,
            m.get("total_wall_time", 0),
            m.get("total_nfe", 0),
            m.get("peak_gpu_memory_mb", 0),
        )
    logger.info("=" * 80)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.full_eval:
        run_full_eval(args, cfg)
    else:
        run_single_eval(args, cfg)


if __name__ == "__main__":
    main()
