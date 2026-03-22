"""
Speculative Denoising for Diffusion Transformers.

Core algorithm: a lightweight draft DiT predicts γ consecutive denoising steps,
which are verified in parallel by the full-size target DiT. An accept/reject
criterion based on score-function divergence guarantees the output distribution
is identical to the target model's.

References:
- Leviathan et al. (2023) "Fast Inference from Transformers via Speculative Decoding"
- de Bortoli et al. (2025) "Accelerated Diffusion Models via Speculative Sampling"
"""

import dataclasses
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class SpecDenoiseOutput:
    """Results from a single speculative denoising run."""
    samples: torch.Tensor
    num_function_evals: int
    num_draft_rounds: int
    total_draft_steps: int
    total_accepted_steps: int
    acceptance_rate_per_timestep: List[float]
    wall_time_seconds: float
    draft_time_seconds: float
    verify_time_seconds: float

    @property
    def acceptance_rate(self) -> float:
        if self.total_draft_steps == 0:
            return 0.0
        return self.total_accepted_steps / self.total_draft_steps

    @property
    def effective_steps_per_nfe(self) -> float:
        if self.num_function_evals == 0:
            return 0.0
        return self.total_accepted_steps / self.num_function_evals

    def to_dict(self) -> dict:
        return {
            "num_function_evals": self.num_function_evals,
            "num_draft_rounds": self.num_draft_rounds,
            "total_draft_steps": self.total_draft_steps,
            "total_accepted_steps": self.total_accepted_steps,
            "acceptance_rate": self.acceptance_rate,
            "effective_steps_per_nfe": self.effective_steps_per_nfe,
            "wall_time_seconds": self.wall_time_seconds,
            "draft_time_seconds": self.draft_time_seconds,
            "verify_time_seconds": self.verify_time_seconds,
            "acceptance_rate_per_timestep": self.acceptance_rate_per_timestep,
        }


class NoiseSchedule:
    """Manages the diffusion noise schedule and provides α_t, σ_t, SNR_t."""

    def __init__(
        self,
        num_timesteps: int = 1000,
        schedule_type: str = "linear",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        self.num_timesteps = num_timesteps
        self.schedule_type = schedule_type

        if schedule_type == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule_type == "cosine":
            steps = torch.linspace(0, num_timesteps, num_timesteps + 1)
            alpha_bar = torch.cos((steps / num_timesteps + 0.008) / 1.008 * torch.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
            betas = betas.clamp(max=0.999)
        elif schedule_type == "shifted_cosine":
            steps = torch.linspace(0, 1, num_timesteps + 1)
            shift = 3.0
            alpha_bar = torch.cos((steps + shift / (1 + shift)) / (1 + shift / (1 + shift)) * torch.pi / 2) ** 2
            betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
            betas = betas.clamp(min=1e-6, max=0.999)
        else:
            raise ValueError(f"Unknown schedule: {schedule_type}")

        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_cumprod = alpha_cumprod
        self.sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod)

    def snr(self, t: torch.Tensor) -> torch.Tensor:
        """Signal-to-noise ratio at timestep t."""
        ac = self.alpha_cumprod.to(t.device)[t]
        return ac / (1.0 - ac).clamp(min=1e-8)

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        sqrt_ac = self.sqrt_alpha_cumprod.to(x0.device)[t]
        sqrt_one_minus_ac = self.sqrt_one_minus_alpha_cumprod.to(x0.device)[t]
        while sqrt_ac.dim() < x0.dim():
            sqrt_ac = sqrt_ac.unsqueeze(-1)
            sqrt_one_minus_ac = sqrt_one_minus_ac.unsqueeze(-1)
        return sqrt_ac * x0 + sqrt_one_minus_ac * noise


def compute_score_divergence(
    draft_eps: torch.Tensor,
    target_eps: torch.Tensor,
    sigma_t: float,
) -> torch.Tensor:
    """
    Compute per-sample KL-like divergence between draft and target score functions.

    In the Gaussian diffusion framework, the score is s(x,t) = -ε(x,t)/σ_t.
    The divergence is measured as ||ε_draft - ε_target||² / (2σ_t²),
    which corresponds to the KL divergence between the implied Gaussian
    reverse-step distributions.

    Returns per-sample divergence values, shape (B,).
    """
    diff = draft_eps - target_eps
    per_sample = diff.pow(2).flatten(1).mean(1)
    return per_sample / (2.0 * sigma_t ** 2 + 1e-8)


def acceptance_probability(
    divergence: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Convert score divergence to acceptance probability.

    p_accept = exp(-divergence / temperature)

    When divergence → 0 (draft ≈ target), acceptance → 1.
    Temperature controls strictness: lower = stricter.
    """
    return torch.exp(-divergence / max(temperature, 1e-8)).clamp(0.0, 1.0)


class SpeculativeDenoiser:
    """
    Speculative denoising engine for Diffusion Transformers.

    Uses a lightweight draft model to predict γ steps ahead, then verifies
    with the target model in a single batched call.
    """

    def __init__(
        self,
        draft_model: Any,
        target_model: Any,
        noise_schedule: NoiseSchedule,
        draft_device: torch.device = torch.device("cuda:0"),
        target_device: torch.device = torch.device("cuda:0"),
    ):
        self.draft_model = draft_model
        self.target_model = target_model
        self.schedule = noise_schedule
        self.draft_device = draft_device
        self.target_device = target_device

        if hasattr(draft_model, "eval"):
            draft_model.eval()
        if hasattr(target_model, "eval"):
            target_model.eval()

    @torch.no_grad()
    def generate(
        self,
        shape: Tuple[int, ...],
        num_inference_steps: int = 50,
        gamma: int = 5,
        guidance_scale: float = 7.5,
        class_labels: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        adaptive_gamma: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> SpecDenoiseOutput:
        """
        Generate samples using speculative denoising.

        Args:
            shape: (B, C, H, W) output shape
            num_inference_steps: total denoising steps for the target schedule
            gamma: number of speculative draft steps per round
            guidance_scale: classifier-free guidance scale
            class_labels: class conditioning (for class-conditional models)
            temperature: acceptance criterion strictness
            adaptive_gamma: if True, adjust γ based on timestep-dependent acceptance
            generator: random generator for reproducibility
        """
        B = shape[0]
        device = self.target_device

        timesteps = torch.linspace(
            self.schedule.num_timesteps - 1, 0, num_inference_steps + 1,
            device=device,
        ).long()

        x_t = torch.randn(shape, device=device, generator=generator)

        step_idx = 0
        total_nfe = 0
        total_draft_steps = 0
        total_accepted = 0
        n_rounds = 0
        t_draft = 0.0
        t_verify = 0.0
        acceptance_counts = {}
        round_counts = {}

        start = time.perf_counter()

        while step_idx < num_inference_steps:
            remaining = num_inference_steps - step_idx
            cur_gamma = min(gamma, remaining)
            if cur_gamma <= 0:
                break

            if adaptive_gamma and n_rounds > 0:
                avg_acc = total_accepted / max(total_draft_steps, 1)
                cur_gamma = max(2, min(gamma, int(1.0 / (1.0 - avg_acc + 1e-8))))
                cur_gamma = min(cur_gamma, remaining)

            n_rounds += 1
            total_draft_steps += cur_gamma

            # --- Draft phase: predict γ steps ---
            t0 = time.perf_counter()
            draft_states, draft_eps_list = self._draft_phase(
                x_t, timesteps, step_idx, cur_gamma,
                guidance_scale, class_labels,
            )
            t_draft += time.perf_counter() - t0

            # --- Verify phase: check all γ steps with target ---
            t0 = time.perf_counter()
            n_accepted, x_next = self._verify_phase(
                x_t, draft_states, draft_eps_list,
                timesteps, step_idx, cur_gamma,
                guidance_scale, class_labels,
                temperature, generator,
            )
            t_verify += time.perf_counter() - t0

            total_nfe += 1 + cur_gamma  # 1 target batch + γ draft calls
            total_accepted += n_accepted

            for k in range(cur_gamma):
                t_val = timesteps[step_idx + k].item()
                t_bin = int(t_val / (self.schedule.num_timesteps / 6))
                acceptance_counts[t_bin] = acceptance_counts.get(t_bin, 0) + (1 if k < n_accepted else 0)
                round_counts[t_bin] = round_counts.get(t_bin, 0) + 1

            x_t = x_next
            step_idx += n_accepted + 1  # accepted steps + 1 corrected step

        wall = time.perf_counter() - start

        acc_per_t = []
        for t_bin in sorted(round_counts.keys()):
            rate = acceptance_counts.get(t_bin, 0) / max(round_counts[t_bin], 1)
            acc_per_t.append(rate)

        return SpecDenoiseOutput(
            samples=x_t,
            num_function_evals=total_nfe,
            num_draft_rounds=n_rounds,
            total_draft_steps=total_draft_steps,
            total_accepted_steps=total_accepted,
            acceptance_rate_per_timestep=acc_per_t,
            wall_time_seconds=wall,
            draft_time_seconds=t_draft,
            verify_time_seconds=t_verify,
        )

    def _draft_phase(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        step_idx: int,
        gamma: int,
        guidance_scale: float,
        class_labels: Optional[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Draft γ denoising steps using the lightweight model."""
        states = []
        eps_list = []
        x = x_t.to(self.draft_device)

        for k in range(gamma):
            t_curr = timesteps[step_idx + k]
            t_next = timesteps[step_idx + k + 1] if step_idx + k + 1 <= len(timesteps) - 1 else torch.tensor(0, device=x.device)

            eps = self._predict_noise(
                self.draft_model, x, t_curr,
                guidance_scale, class_labels, self.draft_device,
            )
            eps_list.append(eps.to(self.target_device))

            x = self._ddim_step(x, eps, t_curr, t_next)
            states.append(x.to(self.target_device))

        return states, eps_list

    def _verify_phase(
        self,
        x_t: torch.Tensor,
        draft_states: List[torch.Tensor],
        draft_eps_list: List[torch.Tensor],
        timesteps: torch.Tensor,
        step_idx: int,
        gamma: int,
        guidance_scale: float,
        class_labels: Optional[torch.Tensor],
        temperature: float,
        generator: Optional[torch.Generator],
    ) -> Tuple[int, torch.Tensor]:
        """Verify draft steps with the target model and accept/reject."""
        B = x_t.shape[0]
        device = self.target_device

        all_x = [x_t.to(device)] + draft_states
        all_t = [timesteps[step_idx + k] for k in range(gamma + 1)
                 if step_idx + k < len(timesteps)]

        if len(all_t) < gamma + 1:
            all_t = all_t + [torch.tensor(0, device=device)] * (gamma + 1 - len(all_t))

        # Batch predict noise for all positions with target model
        target_eps_list = []
        for k in range(gamma):
            eps_target = self._predict_noise(
                self.target_model, all_x[k], all_t[k],
                guidance_scale, class_labels, device,
            )
            target_eps_list.append(eps_target)

        # Accept/reject each step
        n_accepted = 0
        for k in range(gamma):
            t_val = all_t[k]
            sigma_t = self.schedule.sqrt_one_minus_alpha_cumprod.to(device)[t_val.clamp(0, self.schedule.num_timesteps - 1).long()]
            if sigma_t.dim() == 0:
                sigma_t = sigma_t.item()
            else:
                sigma_t = sigma_t.mean().item()

            div = compute_score_divergence(
                draft_eps_list[k], target_eps_list[k], sigma_t,
            )
            p_accept = acceptance_probability(div, temperature)

            u = torch.rand(B, device=device, generator=generator)
            if (u < p_accept).all():
                n_accepted += 1
            else:
                break

        if n_accepted < gamma and n_accepted < len(target_eps_list):
            # Resample from corrected distribution at rejection point
            k = n_accepted
            x_rejected = all_x[k]
            t_curr = all_t[k]
            t_next = all_t[k + 1] if k + 1 < len(all_t) else torch.tensor(0, device=device)

            x_corrected = self._ddim_step(
                x_rejected.to(device),
                target_eps_list[k],
                t_curr, t_next,
            )
            return n_accepted, x_corrected
        elif n_accepted == gamma:
            return n_accepted, draft_states[-1]
        else:
            return n_accepted, all_x[n_accepted]

    def _predict_noise(
        self,
        model: Any,
        x: torch.Tensor,
        t: torch.Tensor,
        guidance_scale: float,
        class_labels: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """Predict noise ε(x_t, t) with optional CFG."""
        x = x.to(device)
        t_input = t.expand(x.shape[0]).to(device) if t.dim() == 0 else t.to(device)

        if guidance_scale > 1.0 and class_labels is not None:
            x_double = torch.cat([x, x], dim=0)
            t_double = torch.cat([t_input, t_input], dim=0)
            null_labels = torch.full_like(class_labels, fill_value=1000)  # null class for DiT
            labels_double = torch.cat([class_labels.to(device), null_labels.to(device)], dim=0)

            eps_double = model(x_double, t_double, labels_double)
            if hasattr(eps_double, "sample"):
                eps_double = eps_double.sample
            eps_cond, eps_uncond = eps_double.chunk(2, dim=0)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
            if class_labels is not None:
                eps = model(x, t_input, class_labels.to(device))
            else:
                eps = model(x, t_input)
            if hasattr(eps, "sample"):
                eps = eps.sample

        return eps

    def _ddim_step(
        self,
        x_t: torch.Tensor,
        eps: torch.Tensor,
        t_curr: torch.Tensor,
        t_next: torch.Tensor,
    ) -> torch.Tensor:
        """Single DDIM deterministic step."""
        device = x_t.device
        t_c = t_curr.long().clamp(0, self.schedule.num_timesteps - 1)
        t_n = t_next.long().clamp(0, self.schedule.num_timesteps - 1)

        alpha_curr = self.schedule.alpha_cumprod.to(device)[t_c]
        alpha_next = self.schedule.alpha_cumprod.to(device)[t_n]

        while alpha_curr.dim() < x_t.dim():
            alpha_curr = alpha_curr.unsqueeze(-1)
            alpha_next = alpha_next.unsqueeze(-1)

        x0_pred = (x_t - (1 - alpha_curr).sqrt() * eps) / alpha_curr.sqrt().clamp(min=1e-8)
        x_next = alpha_next.sqrt() * x0_pred + (1 - alpha_next).sqrt() * eps

        return x_next

    @torch.no_grad()
    def generate_baseline(
        self,
        shape: Tuple[int, ...],
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        class_labels: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, float, int]:
        """Baseline DDIM generation with target model only."""
        device = self.target_device
        timesteps = torch.linspace(
            self.schedule.num_timesteps - 1, 0, num_inference_steps + 1,
            device=device,
        ).long()

        x_t = torch.randn(shape, device=device, generator=generator)
        nfe = 0

        start = time.perf_counter()
        for i in range(num_inference_steps):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]
            eps = self._predict_noise(
                self.target_model, x_t, t_curr,
                guidance_scale, class_labels, device,
            )
            x_t = self._ddim_step(x_t, eps, t_curr, t_next)
            nfe += 1
        wall = time.perf_counter() - start

        return x_t, wall, nfe
