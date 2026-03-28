#!/usr/bin/env python3
"""Download LLM and DiT model weights from Hugging Face."""
import argparse
import gc
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("download_models")


def download_llm_models(model_ids):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    for name in model_ids:
        log.info("LLM: %s", name)
        try:
            AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            m = AutoModelForCausalLM.from_pretrained(
                name, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
            )
            n = sum(p.numel() for p in m.parameters()) / 1e9
            log.info("  OK (%.2fB params)", n)
            del m
            gc.collect()
        except Exception as e:
            log.error("  FAILED: %s", e)


def download_dit_models(model_ids):
    try:
        from diffusers import DiTPipeline
        import torch
    except ImportError:
        log.warning("diffusers not installed, skipping DiT download")
        return

    for repo in model_ids:
        log.info("DiT: %s", repo)
        try:
            pipe = DiTPipeline.from_pretrained(repo, torch_dtype=torch.bfloat16)
            n = sum(p.numel() for p in pipe.transformer.parameters()) / 1e6
            log.info("  OK (%.1fM transformer params)", n)
            del pipe
            gc.collect()
        except Exception as e:
            log.warning("  FAILED: %s", e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_models", nargs="+", default=[])
    parser.add_argument("--dit_models", nargs="+", default=[])
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()


    if args.llm_models:
        log.info("=== Downloading LLM models ===")
        download_llm_models(args.llm_models)

    if args.dit_models:
        log.info("=== Downloading DiT models ===")
        download_dit_models(args.dit_models)

    log.info("Download complete.")


if __name__ == "__main__":
    main()
