"""Core speculative decoding with rejection sampling.

Implements the algorithm from Leviathan et al. (2023) "Fast Inference from
Transformers via Speculative Decoding" and Chen et al. (2023) "Accelerating
Large Language Model Decoding with Speculative Sampling".
"""

import time
import logging
import dataclasses
from typing import Optional, List, Tuple, Any

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class SpeculativeOutput:
    """Aggregated results from a single speculative decoding run."""

    generated_ids: torch.Tensor
    num_generated_tokens: int
    num_draft_rounds: int
    total_draft_tokens: int
    total_accepted_tokens: int
    acceptance_counts_by_position: List[int]
    draft_rounds_by_position: List[int]
    wall_time_seconds: float
    draft_time_seconds: float
    verify_time_seconds: float

    @property
    def acceptance_rate(self) -> float:
        if self.total_draft_tokens == 0:
            return 0.0
        return self.total_accepted_tokens / self.total_draft_tokens

    @property
    def tokens_per_round(self) -> float:
        if self.num_draft_rounds == 0:
            return 0.0
        return self.num_generated_tokens / self.num_draft_rounds

    @property
    def throughput(self) -> float:
        if self.wall_time_seconds == 0:
            return 0.0
        return self.num_generated_tokens / self.wall_time_seconds

    @property
    def position_acceptance_rates(self) -> List[float]:
        rates = []
        for acc, total in zip(
            self.acceptance_counts_by_position, self.draft_rounds_by_position
        ):
            rates.append(acc / total if total > 0 else 0.0)
        return rates

    def to_dict(self) -> dict:
        return {
            "num_generated_tokens": self.num_generated_tokens,
            "num_draft_rounds": self.num_draft_rounds,
            "total_draft_tokens": self.total_draft_tokens,
            "total_accepted_tokens": self.total_accepted_tokens,
            "acceptance_rate": self.acceptance_rate,
            "tokens_per_round": self.tokens_per_round,
            "throughput_tokens_per_sec": self.throughput,
            "wall_time_seconds": self.wall_time_seconds,
            "draft_time_seconds": self.draft_time_seconds,
            "verify_time_seconds": self.verify_time_seconds,
            "acceptance_counts_by_position": self.acceptance_counts_by_position,
            "draft_rounds_by_position": self.draft_rounds_by_position,
            "position_acceptance_rates": self.position_acceptance_rates,
        }


def _trim_kv_cache(past: Any, keep_length: int) -> Any:
    """Trim a KV cache to only retain the first `keep_length` positions."""
    if past is None:
        return None
    if isinstance(past, tuple):
        return tuple(
            tuple(t[:, :, :keep_length, :] for t in layer) for layer in past
        )
    if hasattr(past, "key_cache") and hasattr(past, "value_cache"):
        for i in range(len(past.key_cache)):
            past.key_cache[i] = past.key_cache[i][:, :, :keep_length, :]
            past.value_cache[i] = past.value_cache[i][:, :, :keep_length, :]
        return past
    raise TypeError(f"Unsupported KV cache type: {type(past)}")


class SpeculativeDecoder:
    """Speculative decoding engine with KV-cached draft and target models."""

    def __init__(
        self,
        draft_model: PreTrainedModel,
        target_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
    ):
        self.draft_model = draft_model
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.draft_model.eval()
        self.target_model.eval()
        self.draft_device = next(draft_model.parameters()).device
        self.target_device = next(target_model.parameters()).device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        gamma: int = 5,
        temperature: float = 1.0,
    ) -> SpeculativeOutput:
        """Run speculative decoding and return generation + statistics."""
        assert input_ids.shape[0] == 1, "Only batch_size=1 is supported"

        prefix_len = input_ids.shape[1]
        acc_by_pos = [0] * gamma
        rounds_by_pos = [0] * gamma
        total_draft = 0
        total_accepted = 0
        n_rounds = 0
        t_draft_total = 0.0
        t_verify_total = 0.0

        # --- warm-up: push prefix through both models to fill KV caches ---
        draft_out = self.draft_model(
            input_ids.to(self.draft_device), use_cache=True
        )
        draft_kv = draft_out.past_key_values
        draft_next_logits = draft_out.logits[:, -1, :]

        target_out = self.target_model(
            input_ids.to(self.target_device), use_cache=True
        )
        target_kv = target_out.past_key_values
        target_next_logits = target_out.logits[:, -1, :]

        all_token_ids = input_ids.cpu().clone()
        kv_len = prefix_len

        start = time.perf_counter()

        while all_token_ids.shape[1] - prefix_len < max_new_tokens:
            remaining = max_new_tokens - (all_token_ids.shape[1] - prefix_len)
            cur_gamma = min(gamma, remaining)
            if cur_gamma <= 0:
                break

            n_rounds += 1
            for k in range(cur_gamma):
                rounds_by_pos[k] += 1
            total_draft += cur_gamma

            # ---- draft phase ----
            t0 = time.perf_counter()
            draft_tokens, draft_probs, draft_kv = self._draft_phase(
                draft_next_logits, draft_kv, cur_gamma, temperature
            )
            t_draft_total += time.perf_counter() - t0

            # ---- verify phase ----
            t0 = time.perf_counter()
            verify_out = self.target_model(
                draft_tokens.view(1, -1).to(self.target_device),
                past_key_values=target_kv,
                use_cache=True,
            )
            target_kv_ext = verify_out.past_key_values
            verify_logits = verify_out.logits  # [1, cur_gamma, V]

            n_acc, accepted = self._rejection_sample(
                target_next_logits,
                verify_logits,
                draft_tokens,
                draft_probs,
                cur_gamma,
                temperature,
            )
            t_verify_total += time.perf_counter() - t0

            total_accepted += n_acc
            for k in range(n_acc):
                acc_by_pos[k] += 1

            all_token_ids = torch.cat(
                [all_token_ids, accepted.view(1, -1).cpu()], dim=1
            )

            # ---- update KV caches ----
            new_kv_len = kv_len + n_acc
            draft_kv = _trim_kv_cache(draft_kv, new_kv_len)
            target_kv = _trim_kv_cache(target_kv_ext, new_kv_len)

            last_tok = accepted[-1]

            d_out = self.draft_model(
                last_tok.view(1, 1).to(self.draft_device),
                past_key_values=draft_kv,
                use_cache=True,
            )
            draft_kv = d_out.past_key_values
            draft_next_logits = d_out.logits[:, -1, :]

            t_out = self.target_model(
                last_tok.view(1, 1).to(self.target_device),
                past_key_values=target_kv,
                use_cache=True,
            )
            target_kv = t_out.past_key_values
            target_next_logits = t_out.logits[:, -1, :]

            kv_len = new_kv_len + 1

        wall = time.perf_counter() - start

        final_ids = all_token_ids[:, : prefix_len + max_new_tokens]
        return SpeculativeOutput(
            generated_ids=final_ids,
            num_generated_tokens=final_ids.shape[1] - prefix_len,
            num_draft_rounds=n_rounds,
            total_draft_tokens=total_draft,
            total_accepted_tokens=total_accepted,
            acceptance_counts_by_position=acc_by_pos,
            draft_rounds_by_position=rounds_by_pos,
            wall_time_seconds=wall,
            draft_time_seconds=t_draft_total,
            verify_time_seconds=t_verify_total,
        )

    @torch.no_grad()
    def generate_autoregressive(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, float]:
        """Baseline autoregressive generation with the target model only."""
        generated = input_ids.to(self.target_device)
        past = None

        start = time.perf_counter()
        for _ in range(max_new_tokens):
            if past is None:
                out = self.target_model(generated, use_cache=True)
            else:
                out = self.target_model(
                    generated[:, -1:], past_key_values=past, use_cache=True
                )
            past = out.past_key_values
            logits = out.logits[:, -1, :]
            probs = F.softmax(logits / max(temperature, 1e-8), dim=-1)
            tok = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, tok], dim=1)
        wall = time.perf_counter() - start
        return generated.cpu(), wall

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _draft_phase(
        self,
        start_logits: torch.Tensor,
        kv: Any,
        gamma: int,
        temperature: float,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Any]:
        """Autoregressively draft *gamma* tokens using the draft model.

        Returns (draft_token_ids, draft_distributions, updated_kv).
        """
        tokens: List[torch.Tensor] = []
        probs_list: List[torch.Tensor] = []
        logits = start_logits
        current_kv = kv

        for _ in range(gamma):
            p = F.softmax(logits / max(temperature, 1e-8), dim=-1).squeeze(0)
            tok = torch.multinomial(p, num_samples=1).squeeze(-1)
            tokens.append(tok.cpu())
            probs_list.append(p.cpu())

            out = self.draft_model(
                tok.view(1, 1).to(self.draft_device),
                past_key_values=current_kv,
                use_cache=True,
            )
            current_kv = out.past_key_values
            logits = out.logits[:, -1, :]

        return torch.stack(tokens), probs_list, current_kv

    @staticmethod
    def _rejection_sample(
        target_next_logits: torch.Tensor,
        verify_logits: torch.Tensor,
        draft_tokens: torch.Tensor,
        draft_probs: List[torch.Tensor],
        gamma: int,
        temperature: float,
    ) -> Tuple[int, torch.Tensor]:
        """Rejection-sampling verification of draft tokens.

        Returns (n_accepted_from_draft, accepted_token_tensor).
        """
        device = verify_logits.device
        temp = max(temperature, 1e-8)
        accepted: List[torch.Tensor] = []
        n_accepted = 0

        for i in range(gamma):
            if i == 0:
                tgt_logits_i = target_next_logits.squeeze(0)
            else:
                tgt_logits_i = verify_logits[:, i - 1, :].squeeze(0)

            tp = F.softmax(tgt_logits_i.to(device) / temp, dim=-1)
            dp = draft_probs[i].to(device)
            tok_id = draft_tokens[i].item()

            p_t = tp[tok_id]
            p_d = dp[tok_id].clamp(min=1e-10)

            if torch.rand(1, device=device).item() < min(1.0, (p_t / p_d).item()):
                accepted.append(draft_tokens[i])
                n_accepted += 1
            else:
                adjusted = (tp - dp).clamp(min=0)
                s = adjusted.sum()
                if s > 0:
                    adjusted = adjusted / s
                else:
                    adjusted = tp
                new_tok = torch.multinomial(adjusted, num_samples=1).squeeze(-1)
                accepted.append(new_tok.cpu())
                break
        else:
            bonus_logits = verify_logits[:, gamma - 1, :].squeeze(0)
            bonus_p = F.softmax(bonus_logits.to(device) / temp, dim=-1)
            bonus = torch.multinomial(bonus_p, num_samples=1).squeeze(-1)
            accepted.append(bonus.cpu())

        return n_accepted, torch.stack(accepted)

    # ------------------------------------------------------------------
    # Detailed per-token analysis (used by eval_speculative.py)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def analyse_acceptance(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        gamma: int = 5,
        temperature: float = 1.0,
        num_trials: int = 20,
    ) -> dict:
        """Run many trials and return rich per-position statistics."""
        all_pos_rates: List[List[float]] = []
        all_accept_rates: List[float] = []
        all_tpr: List[float] = []

        for _ in range(num_trials):
            out = self.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                gamma=gamma,
                temperature=temperature,
            )
            all_accept_rates.append(out.acceptance_rate)
            all_tpr.append(out.tokens_per_round)
            all_pos_rates.append(out.position_acceptance_rates)

        import numpy as np

        pos_arr = np.array(all_pos_rates)
        return {
            "gamma": gamma,
            "temperature": temperature,
            "mean_acceptance_rate": float(np.mean(all_accept_rates)),
            "std_acceptance_rate": float(np.std(all_accept_rates)),
            "mean_tokens_per_round": float(np.mean(all_tpr)),
            "position_acceptance_mean": pos_arr.mean(axis=0).tolist(),
            "position_acceptance_std": pos_arr.std(axis=0).tolist(),
        }

    @torch.no_grad()
    def estimate_kl_divergence(
        self,
        input_ids: torch.Tensor,
        max_positions: int = 256,
    ) -> dict:
        """Estimate KL(target || draft) on the given prefix.

        Computes forward KL at each position in *input_ids* and averages.
        """
        seq_len = min(input_ids.shape[1], max_positions)
        chunk = input_ids[:, :seq_len]

        draft_logits = self.draft_model(
            chunk.to(self.draft_device)
        ).logits.float()
        target_logits = self.target_model(
            chunk.to(self.target_device)
        ).logits.float()

        draft_lp = F.log_softmax(draft_logits.cpu(), dim=-1)
        target_lp = F.log_softmax(target_logits.cpu(), dim=-1)
        target_p = target_lp.exp()

        kl_per_pos = (target_p * (target_lp - draft_lp)).sum(dim=-1).squeeze(0)
        return {
            "kl_mean": kl_per_pos.mean().item(),
            "kl_std": kl_per_pos.std().item(),
            "kl_per_position": kl_per_pos.tolist(),
        }
