# SpecScale: Universal Scaling Laws for Speculative Inference Across Modalities

---

## Quick Start

```bash
# 1. Clone and enter project
git clone https://github.com/Sunshine535/nips-specscale.git
cd nips-specscale

# 2. Install dependencies
bash setup.sh

# 3. Run all experiments
bash run.sh

# 4. (Optional) Run in background for long experiments
nohup bash run.sh > run.log 2>&1 &
tail -f run.log
```

### Check Completion

```bash
cat results/.pipeline_done   # Shows PIPELINE_COMPLETE when all phases finish
ls results/.phase_markers/   # See which individual phases completed
```

### Save and Send Results

```bash
# Option A: Push to GitHub
git add results/ logs/
git commit -m "Experiment results"
git push origin main

# Option B: Package as tarball
bash collect_results.sh
# Output: results_archive/nips-specscale_results_YYYYMMDD_HHMMSS.tar.gz
```

### Resume After Interruption

Re-run `bash run.sh` — completed phases are automatically skipped.
To force re-run all phases: `FORCE_RERUN=1 bash run.sh`

## Project Structure

```
nips-specscale/
├── README.md
├── LICENSE
├── setup.sh                                  # One-click environment setup
├── requirements.txt
├── configs/
│   ├── llm_pairs.yaml                        # LLM draft/target model pairs
│   ├── dit_config.yaml                       # DiT model configurations
│   └── default.yaml                          # Default experiment settings
├── scripts/
│   ├── gpu_utils.sh                          # Auto GPU detection utilities
│   ├── run_all_experiments.sh                # Master pipeline (Phase 0–8)
│   ├── download_models.py                    # Phase 0: Download LLM + DiT models
│   ├── benchmark_speculative.py              # Phase 1: LLM speculative decoding sweep
│   ├── run_scaling_law_llm.py                # Phase 2: LLM scaling law fitting
│   ├── run_acceptance_sweep_dit.py           # Phase 3: DiT acceptance sweep
│   ├── run_scaling_law_dit.py                # Phase 4: DiT scaling law fitting
│   ├── run_unified_comparison.py             # Phase 5: Cross-modal comparison
│   ├── run_imagenet_eval.py                  # Phase 6: ImageNet FID evaluation
│   └── generate_paper_figures_tables.py      # Phase 8: Paper figures + tables
├── src/
│   ├── speculative_decode.py                 # Core speculative decoding
│   ├── speculative_denoise.py                # Speculative denoising for DiT
│   ├── dit_loader.py                         # DiT model loading utilities
│   └── scaling_law_diffusion.py              # Scaling law analysis
├── results/                                  # Experiment outputs
└── logs/                                     # Training logs
```

## Experiments

| Phase | Description | Details |
|-------|------------|---------|
| 0 | Model download | Qwen3.5 full series + DiT models |
| 1 | LLM speculative decoding sweep | Draft/target pairs across model sizes |
| 2 | LLM scaling law | Fit S(d,T,γ) for language models |
| 3 | DiT acceptance sweep | Speculative denoising across DiT sizes |
| 4 | DiT scaling law | Fit scaling formula for diffusion |
| 5 | Cross-modal comparison | Unified S(d,T,γ) narrative |
| 6 | ImageNet FID evaluation | Generation quality assessment |
| 7 | Ablation studies | Component analysis |
| 8 | Paper figures & tables | Publication-ready outputs |

## Models

| Component | Model | Role |
|-----------|-------|------|
| LLM series | Qwen/Qwen3.5-{0.8B, 4B, 9B, 14B, 27B} | Draft/target LLM pairs |
| DiT series | facebook/DiT-{S,B,L,XL}/2-256 | Draft/target diffusion pairs |

## Timeline & GPU Budget

| Phase | Est. GPU-hours |
|-------|---------------|
| LLM speculative sweep | ~200 |
| LLM scaling law | ~50 |
| DiT acceptance sweep | ~300 |
| DiT scaling law | ~50 |
| ImageNet FID eval | ~100 |
| Ablations + figures | ~50 |
| **Total** | **~750** |

## Citation

```bibtex
@inproceedings{specscale2026neurips,
  title={SpecScale: Universal Scaling Laws for Speculative Inference Across Modalities},
  author={Anonymous},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
