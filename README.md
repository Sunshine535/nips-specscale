# SpecScale: Universal Scaling Laws for Speculative Inference Across Modalities

**SpecScale** is a research codebase that unifies two lines of work—**SpecDraft** (speculative decoding for autoregressive large language models) and **SpecDiff** (speculative denoising for diffusion transformers, DiT)—under a single empirical and theoretical question: *do speculative inference mechanisms obey universal scaling laws when we vary draft/target capacity and speculation depth?*

The project tests the hypothesis that **acceptance-rate structure** in speculative inference can be described by a shared scaling form across **discrete autoregressive** and **continuous diffusion** generative stacks, after accounting for modality-specific modulation.

---

## Narrative (paper arc)

1. **Theory-first framing (SpecDraft).**  
   We posit a scaling law for the mean acceptance rate (or equivalent verification statistic) as a function of draft size \(d\), target size \(T\), and speculation depth \(\gamma\), denoted **\(S(d, T, \gamma)\)**. In the autoregressive setting, \(S\) is modeled as a **power law in the model-size ratio** \(d/T\) (with \(\gamma\)-dependent exponents and prefactors), capturing how often draft proposals survive target verification as capacities are varied.

2. **LLM validation.**  
   We measure \(S(d,T,\gamma)\) on **eleven draft/target pairs** spanning a wide range of size ratios, **seven \(\gamma\) values**, **four benchmark suites**, and **three random seeds**, using tree-consistent speculative decoding implementations and standardized prompts.

3. **Diffusion extension (SpecDiff).**  
   We lift the same measurement protocol to **class-conditional DiT** pipelines, where speculation operates in **continuous denoising time** rather than token index. Empirically, the **same scaling-law skeleton** holds, but with an additional **timestep-dependent modulation \(h(t)\)** that accounts for varying verification difficulty across noise levels.

4. **Universality claim.**  
   Joint analysis shows that **speculative inference scaling is modality-agnostic at the functional level**: AR and DiT runs collapse onto a shared parametric family once \(h(t)\) (or its discrete analogue) is absorbed into the effective statistics being fit.

---

## Benchmarks and metrics

### Large language models

| Benchmark | Role in this repo |
|-----------|-------------------|
| **MT-Bench** | Multi-turn chat quality under speculative decoding (conversation-level acceptance and task success). |
| **HumanEval+** | Code completion with extended tests; measures acceptance vs. functional correctness. |
| **GSM8K** | Grade-school math reasoning; stress-tests long reasoning chains at fixed \(\gamma\). |
| **MATH** | Competition-level math; higher difficulty tail for acceptance stability. |
| **MMLU** | Broad knowledge/multiple-choice; coverage across domains. |

> **Pipeline note.** Phase 1 of `scripts/run_all_experiments.sh` runs the **four-dataset core grid** (`gsm8k`, `math`, `humaneval_plus`, `mmlu`) for the paper’s main LLM scaling-law table. **MT-Bench** is included in the paper’s benchmark list and should be invoked via the same driver (`scripts/benchmark_speculative.py`) with a `mt_bench` dataset hook when the optional conversation data path is configured.

Primary measured quantities include **mean acceptance rate**, **tokens accepted per verification round**, **wall-clock speedup vs. autoregressive baseline**, and **task accuracy** where applicable.

### Diffusion transformers (ImageNet 256×256)

| Metric | Description |
|--------|-------------|
| **FID-50K** | Fréchet Inception Distance on 50k generated samples vs. reference ImageNet statistics (`clean-fid` / `torchmetrics`). |
| **Inception Score (IS)** | Sharpness and diversity proxy reported alongside FID for class-conditional DiT. |

Phase 6 runs **`scripts/run_imagenet_eval.py`**, which should materialize `fid_50k`, IS summaries, and timing logs under `results/imagenet_eval/`.

---

## Repository layout (expected)

```text
nips-specscale/
  README.md
  requirements.txt
  setup.sh                    # recommended: create .venv + install deps
  configs/
    default.yaml              # DiT paths, scheduler, default guidance
  scripts/
    run_all_experiments.sh    # master orchestration (Phases 0–8)
    benchmark_speculative.py  # LLM measurement driver
    run_scaling_law_llm.py    # fit S(d,T,γ) on LLM JSONs
    run_acceptance_sweep_dit.py  # DiT grid: 3×5×3×3 (pairs×γ×guidance×seeds)
    run_scaling_law_dit.py    # DiT fit + h(t) modulation
    run_unified_comparison.py # cross-modality collapse / universality plots
    run_imagenet_eval.py      # FID-50K + IS
    run_ablations.sh          # γ schedules, temperatures, noise schedules
    generate_paper_figures_tables.py
    download_models.py        # optional explicit prefetch wrapper
  src/                        # shared speculative decoders / DiT wrappers
  results/                    # created by experiments (git-ignored)
```

---

## Quick start

```bash
git clone <this-repo> nips-specscale
cd nips-specscale

# Python 3.10+ recommended
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
# Install a CUDA build of PyTorch matching your driver from https://pytorch.org

bash scripts/run_all_experiments.sh --quick
```

**Smoke test.** `--quick` shrinks sample counts, uses **two** LLM pairs, **two** \(\gamma\) values, **two** datasets, **one** seed, and a **reduced DiT sweep** (`--sweep_mode quick_subset` inside `run_acceptance_sweep_dit.py`).

**Full paper reproduction.**

```bash
bash scripts/run_all_experiments.sh
```

**Resume / single phase.**

```bash
bash scripts/run_all_experiments.sh --from-phase 4
bash scripts/run_all_experiments.sh --only-phase 6
```

**Dry run (print-only).**

```bash
bash scripts/run_all_experiments.sh --dry-run --from-phase 1
```

---

## Hardware expectations

- **LLM phases:** multi-GPU machines are strongly preferred when target models approach **27B parameters** (e.g., one GPU for draft, remaining GPUs for target sharding via `device_map="auto"` inside `benchmark_speculative.py`).
- **DiT phases:** **one high-memory GPU** (24GB+ consumer, or 40GB/80GB datacenter) per job is typical for DiT-XL at 256×256 with mixed precision.
- `scripts/run_all_experiments.sh` sources **`gpu_utils.sh`** (from `github_repos/_shared/` in the monorepo, or a copied `scripts/gpu_utils.sh` in a standalone checkout) and exports `NUM_GPUS`, `GPU_MEM_MIB`, and `GPU_CLASS` for downstream scripts.

---

## Master pipeline (Phases 0–8)

| Phase | Script(s) | Purpose |
|-------|-----------|---------|
| 0 | `download_models.py` (optional) or inline prefetch | Cache **Qwen3.5** LLM weights and **facebook/DiT-*-2-256** checkpoints. |
| 1 | `benchmark_speculative.py` | **11×7×4×3** LLM grid: pairs × \(\gamma\) × datasets × seeds. |
| 2 | `run_scaling_law_llm.py` | Nonlinear fit of **\(S(d,T,\gamma)\)**; residuals vs. ratio \(d/T\). |
| 3 | `run_acceptance_sweep_dit.py --full_sweep` | **3×5×3×3** DiT acceptance measurements (pairs × \(\gamma\) × guidance × seeds). |
| 4 | `run_scaling_law_dit.py --fit_ht_modulation` | Same law family + **\(h(t)\)** timestep modulation. |
| 5 | `run_unified_comparison.py` | **Cross-modality** parameter alignment and universality figure. |
| 6 | `run_imagenet_eval.py` | **FID-50K** and **IS** for speculative vs. baseline denoising. |
| 7 | `run_ablations.sh` | **\(\gamma\) schedules**, **temperature**, **noise schedule** robustness. |
| 8 | `generate_paper_figures_tables.py` | Publication PDFs/CSVs under `results/final/`. |

---

## Dependencies

See `requirements.txt`. Notable pinned minimums:

- `diffusers>=0.30.0` — DiT pipelines and schedulers.
- `torchmetrics>=1.3.0` and `clean-fid>=0.1.35` — FID/IS evaluation.
- `Pillow>=10.0.0`, `scikit-learn>=1.5.0` — preprocessing / lightweight fitting utilities.

---

## Citation (placeholder)

```bibtex
@misc{specscale2026,
  title={Universal Scaling Laws for Speculative Inference Across Modalities},
  author={TBD},
  year={2026},
  note={SpecScale: SpecDraft + SpecDiff unified codebase}
}
```

---

## License

MIT License — see `LICENSE` if present in the distribution you cloned.

---

## Acknowledgments

This repository **merges and extends** the **SpecDraft** (autoregressive speculative decoding scaling laws) and **SpecDiff** (DiT speculative denoising) codebases. DiT checkpoints are provided by the **facebook/DiT-*-2-256** releases on Hugging Face; LLM experiments default to the **Qwen3.5** open-weights family.
