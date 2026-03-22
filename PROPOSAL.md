# SpecScale: Universal Scaling Laws for Speculative Inference Across Modalities

## One-Sentence Summary

We derive and empirically validate universal scaling laws for speculative inference, showing that acceptance rate follows a power-law in model size ratio with shared exponent structure across autoregressive LLMs and diffusion transformers.

## Problem

Speculative decoding accelerates LLM inference by using a small draft model to propose tokens that a large target model verifies in parallel. Speculative denoising extends this to diffusion models. However:

1. **No predictive theory**: Practitioners must run expensive sweeps to find optimal draft model size and speculation length (γ) for each deployment.
2. **Modality isolation**: LLM and diffusion communities develop speculative methods independently, missing potential shared structure.
3. **No optimal configuration formula**: The relationship between draft/target model sizes, γ, and actual speedup lacks a closed-form characterization.

## Core Hypothesis

**Acceptance rate follows a universal power law**: α(r) = a · r^b, where r = d/T is the draft-to-target parameter ratio, with the exponent b being approximately constant across modalities. For diffusion models, a timestep-dependent modulation h(t) captures the varying difficulty of denoising at different noise levels.

## Approach

### Track 1: LLM Scaling Laws (SpecDraft)

**Measurement infrastructure**: 11 Qwen model pairs (0.8B-32B) × 7 γ values × 4 datasets × 3 seeds = 924 measurement points.

**Scaling law fitting**:
- Acceptance rate: α(d, T, γ, k) = a · (d/T)^b · γ^c · exp(-d_k · k)
- Position-dependent decay: α_k = α_base · exp(-λ · k)
- Speedup formula: S = E[accepted + 1] / (γ · c_draft + c_verify)
- Optimal γ*: closed-form from derivative of S w.r.t. γ

### Track 2: Diffusion Scaling Laws (SpecDiff)

**Extension to continuous domain**: In diffusion models, the draft predicts multiple denoising steps, and the target verifies by comparing score function divergence.

**Key additions**:
- Timestep modulation: α(r, t) = α_base(r) · h(t), where h(t) models varying acceptance across noise levels
- Two h(t) models: linear (1 - slope·t) and cosine (cos(πt/2))
- Score divergence metric replaces token probability ratio

**Measurement**: 3 DiT pairs × 5 γ × 3 guidance scales × 3 seeds × 6 timestep bins = 810 measurements + FID-50K evaluation on ImageNet 256×256.

### Track 3: Universality Analysis

Cross-modality comparison:
- Fit exponents independently for LLM and DiT
- Test whether exponent b is statistically similar
- Unified visualization overlaying both domains
- Implications for new modalities (audio, video, 3D)

## Experiments

| Phase | Details |
|-------|---------|
| LLM sweep | 924 measurement points across 11 model pairs |
| LLM scaling law | Power-law fit, position decay, optimal γ derivation |
| DiT acceptance sweep | 810 measurements across 3 DiT pairs |
| DiT scaling law | Timestep-modulated fit (linear + cosine h(t)) |
| Cross-modality | Unified analysis, universality test |
| ImageNet eval | FID-50K, Inception Score for DiT |
| Ablations | γ schedules, temperatures, noise schedules, solvers |

## Benchmarks

**LLM**: GSM8K, MATH, HumanEval, MMLU (+ optional MT-Bench)
**DiT**: ImageNet 256×256 class-conditional generation (FID-50K, IS)

## Expected Contributions

1. First unified scaling law for speculative inference across modalities
2. Closed-form optimal γ* formula for deployment without sweeps
3. Timestep-modulated scaling law for diffusion speculative denoising
4. Empirical validation on 1700+ measurement points
5. Practical deployment guidelines: given model pair → predict speedup

## Novelty Argument

While speculative decoding (Leviathan 2023) and various extensions exist, no prior work:
- Derives predictive scaling laws for acceptance rate
- Shows universality across discrete (LLM) and continuous (diffusion) domains
- Provides closed-form optimal draft length without empirical sweeps

## NeurIPS Justification

Theory-first paper with strong empirical validation. The universal scaling law has immediate practical impact (deployment optimization) and theoretical significance (connecting discrete and continuous generative inference).
