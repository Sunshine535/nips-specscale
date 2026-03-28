#!/usr/bin/env python3
"""Benchmark speculative decoding across datasets and model pairs.

Usage:
    python scripts/benchmark_speculative.py \
        --draft_model Qwen/Qwen3.5-0.8B \
        --target_model Qwen/Qwen3.5-27B \
        --gamma 5 \
        --datasets gsm8k math humaneval mmlu \
        --num_samples 50 \
        --output_dir results/
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.speculative_decode import SpeculativeDecoder

logger = logging.getLogger("spec_benchmark")


# ======================================================================
# Model loading
# ======================================================================

def load_models(
    draft_name: str,
    target_name: str,
    draft_gpu: int = 0,
    dtype: torch.dtype = torch.bfloat16,
):
    """Load draft and target models on separate GPUs.

    Draft model → single GPU.
    Target model → distributed across remaining GPUs via device_map="auto".
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading tokenizer from %s", target_name)
    tokenizer = AutoTokenizer.from_pretrained(target_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_gpus = torch.cuda.device_count()
    logger.info("Detected %d GPUs", num_gpus)

    logger.info("Loading draft model %s on GPU %d", draft_name, draft_gpu)
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_name,
        torch_dtype=dtype,
        device_map={"": draft_gpu},
        trust_remote_code=True,
    )

    if num_gpus > 1:
        max_memory = {
            i: "0GiB" if i == draft_gpu else "75GiB" for i in range(num_gpus)
        }
    else:
        max_memory = None

    logger.info("Loading target model %s (auto device_map)", target_name)
    target_model = AutoModelForCausalLM.from_pretrained(
        target_name,
        torch_dtype=dtype,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
    )

    return draft_model, target_model, tokenizer


# ======================================================================
# Dataset loading — local JSONL first, then HF Hub, then synthetic
# ======================================================================

_DATA_CACHE_DIR = Path(os.environ.get(
    "SPECSCALE_DATA_DIR",
    Path(__file__).resolve().parent.parent / "data" / "prompts",
))

DATASET_LOADERS: Dict[str, Any] = {}


def _try_local(name: str, num_samples: int) -> "List[str] | None":
    """Load pre-cached prompts from local JSONL."""
    jsonl = _DATA_CACHE_DIR / f"{name}.jsonl"
    if not jsonl.exists():
        return None
    prompts = []
    with open(jsonl) as f:
        for line in f:
            if len(prompts) >= num_samples:
                break
            obj = json.loads(line)
            prompts.append(obj["prompt"])
    logger.info("Loaded %d prompts for %s from %s", len(prompts), name, jsonl)
    return prompts if prompts else None


def _load_gsm8k(num_samples: int) -> List[str]:
    local = _try_local("gsm8k", num_samples)
    if local:
        return local

    from datasets import load_dataset

    for repo in ["openai/gsm8k", "gsm8k"]:
        try:
            ds = load_dataset(repo, "main", split="test")
            break
        except Exception as e:
            logger.debug("gsm8k from %s failed: %s", repo, e)
            continue
    else:
        logger.warning("gsm8k unavailable, generating synthetic math prompts")
        return [
            f"Solve the following math problem step by step.\n\n"
            f"Question: A store has {i*3} apples and sells {i} each day. "
            f"How many days until they run out?\n\nAnswer:"
            for i in range(1, num_samples + 1)
        ]
    prompts = []
    for ex in ds.select(range(min(num_samples, len(ds)))):
        prompts.append(
            f"Solve the following math problem step by step.\n\n"
            f"Question: {ex['question']}\n\nAnswer:"
        )
    return prompts


def _load_math(num_samples: int) -> List[str]:
    local = _try_local("math", num_samples)
    if local:
        return local

    from datasets import load_dataset

    for repo_cfg in [
        ("HuggingFaceH4/MATH-500", None),
        ("hendrycks/competition_math", None),
        ("math_dataset", "algebra__linear_1d"),
    ]:
        try:
            repo, cfg = repo_cfg
            ds = load_dataset(repo, cfg, split="test")
            break
        except Exception as e:
            logger.debug("math from %s/%s failed: %s", repo, cfg, e)
            continue
    else:
        logger.warning("math dataset unavailable, generating synthetic prompts")
        return [
            f"Solve the following competition math problem.\n\n"
            f"Problem: Find the value of x if {i}x + {i*2} = {i*5}.\n\nSolution:"
            for i in range(1, num_samples + 1)
        ]
    prompts = []
    for ex in ds.select(range(min(num_samples, len(ds)))):
        problem = ex.get("problem", ex.get("question", ""))
        prompts.append(
            f"Solve the following competition math problem.\n\n"
            f"Problem: {problem}\n\nSolution:"
        )
    return prompts


def _load_humaneval(num_samples: int) -> List[str]:
    local = _try_local("humaneval", num_samples)
    if local:
        return local

    from datasets import load_dataset

    for repo in ["openai/openai_humaneval", "openai_humaneval"]:
        try:
            ds = load_dataset(repo, split="test")
            break
        except Exception as e:
            logger.debug("humaneval from %s failed: %s", repo, e)
            continue
    else:
        logger.warning("humaneval unavailable, generating synthetic code prompts")
        return [
            f"def solution_{i}(n: int) -> int:\n"
            f"    \"\"\"Return the {i}-th element of the Fibonacci sequence.\"\"\"\n"
            for i in range(num_samples)
        ]
    prompts = []
    for ex in ds.select(range(min(num_samples, len(ds)))):
        prompts.append(ex["prompt"])
    return prompts


def _load_mmlu(num_samples: int) -> List[str]:
    local = _try_local("mmlu", num_samples)
    if local:
        return local

    from datasets import load_dataset

    for repo_cfg in [
        ("cais/mmlu", "all"),
        ("hails/mmlu_no_train", "all"),
        ("tasksource/mmlu", "abstract_algebra"),
    ]:
        try:
            repo, cfg = repo_cfg
            ds = load_dataset(repo, cfg, split="test")
            break
        except Exception as e:
            logger.debug("mmlu from %s/%s failed: %s", repo, cfg, e)
            continue
    else:
        logger.warning("mmlu unavailable, generating synthetic MCQ prompts")
        topics = ["physics", "chemistry", "biology", "history", "geography",
                  "math", "literature", "economics", "philosophy", "law"]
        return [
            f"Answer the following multiple-choice question.\n\n"
            f"Question: Which of the following is a key concept in {topics[i % len(topics)]}?\n"
            f"A. Concept Alpha\nB. Concept Beta\nC. Concept Gamma\nD. Concept Delta\n\nAnswer:"
            for i in range(num_samples)
        ]
    prompts = []
    for ex in ds.select(range(min(num_samples, len(ds)))):
        q = ex.get("question", ex.get("input", ""))
        choices_raw = ex.get("choices", [])
        if isinstance(choices_raw, list) and choices_raw:
            letters = "ABCDEFGH"
            options = "\n".join(
                f"{letters[i]}. {c}" for i, c in enumerate(choices_raw)
            )
        else:
            options = ""
        prompts.append(
            f"Answer the following multiple-choice question.\n\n"
            f"Question: {q}\n{options}\n\nAnswer:"
        )
    return prompts


DATASET_LOADERS = {
    "gsm8k": _load_gsm8k,
    "math": _load_math,
    "humaneval": _load_humaneval,
    "humaneval_plus": _load_humaneval,
    "mmlu": _load_mmlu,
}


def load_datasets(names: List[str], num_samples: int) -> Dict[str, List[str]]:
    result = {}
    for name in names:
        loader = DATASET_LOADERS.get(name)
        if loader is None:
            logger.warning("Unknown dataset %s — skipping", name)
            continue
        logger.info("Loading dataset %s (%d samples)", name, num_samples)
        try:
            result[name] = loader(num_samples)
        except Exception as e:
            logger.error("Failed to load %s: %s", name, e)
    return result


# ======================================================================
# Benchmark runner
# ======================================================================


def run_single_benchmark(
    decoder: SpeculativeDecoder,
    prompts: List[str],
    gamma: int,
    max_new_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    """Run speculative decoding + autoregressive baseline on a set of prompts."""
    tokenizer = decoder.tokenizer

    spec_accept_rates = []
    spec_tpr = []
    spec_throughputs = []
    spec_wall_times = []

    baseline_throughputs = []
    baseline_wall_times = []

    for prompt in tqdm(prompts, desc="spec_decode", leave=False):
        ids = tokenizer(prompt, return_tensors="pt").input_ids
        out = decoder.generate(
            ids, max_new_tokens=max_new_tokens, gamma=gamma, temperature=temperature
        )
        spec_accept_rates.append(out.acceptance_rate)
        spec_tpr.append(out.tokens_per_round)
        spec_throughputs.append(out.throughput)
        spec_wall_times.append(out.wall_time_seconds)

    for prompt in tqdm(prompts, desc="baseline_ar", leave=False):
        ids = tokenizer(prompt, return_tensors="pt").input_ids
        _, wall = decoder.generate_autoregressive(
            ids, max_new_tokens=max_new_tokens, temperature=temperature
        )
        baseline_throughputs.append(max_new_tokens / wall)
        baseline_wall_times.append(wall)

    mean_spec_wall = float(np.mean(spec_wall_times))
    mean_base_wall = float(np.mean(baseline_wall_times))

    return {
        "gamma": gamma,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "num_prompts": len(prompts),
        "acceptance_rate_mean": float(np.mean(spec_accept_rates)),
        "acceptance_rate_std": float(np.std(spec_accept_rates)),
        "tokens_per_round_mean": float(np.mean(spec_tpr)),
        "tokens_per_round_std": float(np.std(spec_tpr)),
        "spec_throughput_mean": float(np.mean(spec_throughputs)),
        "spec_throughput_std": float(np.std(spec_throughputs)),
        "spec_wall_time_mean": mean_spec_wall,
        "baseline_throughput_mean": float(np.mean(baseline_throughputs)),
        "baseline_throughput_std": float(np.std(baseline_throughputs)),
        "baseline_wall_time_mean": mean_base_wall,
        "wall_clock_speedup": mean_base_wall / mean_spec_wall if mean_spec_wall > 0 else 0,
    }


# ======================================================================
# Main
# ======================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Speculative decoding benchmark")
    p.add_argument("--draft_model", type=str, required=True)
    p.add_argument("--target_model", type=str, required=True)
    p.add_argument("--gamma", type=int, nargs="+", default=[5])
    p.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["gsm8k", "math", "humaneval", "mmlu"],
    )
    p.add_argument("--num_samples", type=int, default=50)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--output_dir", type=str, default="results")
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

    draft_model, target_model, tokenizer = load_models(
        args.draft_model, args.target_model, draft_gpu=args.draft_gpu
    )
    decoder = SpeculativeDecoder(draft_model, target_model, tokenizer)

    datasets = load_datasets(args.datasets, args.num_samples)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    draft_short = args.draft_model.split("/")[-1]
    target_short = args.target_model.split("/")[-1]

    for ds_name, prompts in datasets.items():
        for gamma in args.gamma:
            fname = f"{draft_short}__{target_short}__g{gamma}__{ds_name}.json"
            out_path = out_dir / fname
            if out_path.exists():
                logger.info("SKIP (exists) %s", out_path)
                continue

            logger.info(
                "Benchmarking %s → %s | gamma=%d | dataset=%s",
                draft_short,
                target_short,
                gamma,
                ds_name,
            )
            results = run_single_benchmark(
                decoder,
                prompts,
                gamma=gamma,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            results["draft_model"] = args.draft_model
            results["target_model"] = args.target_model
            results["dataset"] = ds_name
            results["draft_size_B"] = _infer_size(args.draft_model)
            results["target_size_B"] = _infer_size(args.target_model)

            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info("Saved → %s", out_path)


def _infer_size(model_name: str) -> float:
    """Best-effort extraction of model size in billions from the name."""
    import re

    m = re.search(r"(\d+(?:\.\d+)?)[Bb]", model_name)
    if m:
        return float(m.group(1))
    return 0.0


if __name__ == "__main__":
    main()
