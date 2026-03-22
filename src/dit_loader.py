"""
Utility to load DiT models of different sizes for speculative denoising experiments.

Supports:
- DiT-S/2 (33M) — draft
- DiT-B/2 (131M) — draft/target
- DiT-XL/2 (675M) — target
- Custom wrappers for diffusers-based models
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

MODEL_CONFIGS = {
    "DiT-S/2": {"depth": 12, "hidden_size": 384, "num_heads": 6, "patch_size": 2, "params_M": 33},
    "DiT-B/2": {"depth": 12, "hidden_size": 768, "num_heads": 12, "patch_size": 2, "params_M": 131},
    "DiT-L/2": {"depth": 24, "hidden_size": 1024, "num_heads": 16, "patch_size": 2, "params_M": 458},
    "DiT-XL/2": {"depth": 28, "hidden_size": 1152, "num_heads": 16, "patch_size": 2, "params_M": 675},
}


def load_dit_models(
    draft_name: str = "DiT-S/2",
    target_name: str = "DiT-XL/2",
    image_size: int = 256,
    num_classes: int = 1000,
    draft_device: str = "cuda:0",
    target_device: str = "cuda:0",
    dtype: torch.dtype = torch.float32,
    pretrained: bool = True,
) -> Tuple[nn.Module, nn.Module, dict]:
    """
    Load draft and target DiT models.

    Returns: (draft_model, target_model, info_dict)
    """
    info = {
        "draft_name": draft_name,
        "target_name": target_name,
        "draft_params_M": MODEL_CONFIGS.get(draft_name, {}).get("params_M", 0),
        "target_params_M": MODEL_CONFIGS.get(target_name, {}).get("params_M", 0),
    }

    try:
        from diffusers import DiTPipeline
        logger.info("Loading models via diffusers DiTPipeline")

        draft_model = _load_via_diffusers(draft_name, image_size, num_classes, draft_device, dtype, pretrained)
        target_model = _load_via_diffusers(target_name, image_size, num_classes, target_device, dtype, pretrained)

        return draft_model, target_model, info

    except ImportError:
        logger.info("diffusers not available, falling back to standalone DiT")

    draft_model = _build_standalone_dit(draft_name, image_size, num_classes, draft_device, dtype)
    target_model = _build_standalone_dit(target_name, image_size, num_classes, target_device, dtype)

    return draft_model, target_model, info


def _load_via_diffusers(
    model_name: str,
    image_size: int,
    num_classes: int,
    device: str,
    dtype: torch.dtype,
    pretrained: bool,
) -> nn.Module:
    """Load a DiT model using the diffusers library."""
    from diffusers import DiTPipeline

    hf_name_map = {
        "DiT-S/2": "facebook/DiT-S-256" if image_size == 256 else "facebook/DiT-S-512",
        "DiT-B/2": "facebook/DiT-B-256" if image_size == 256 else "facebook/DiT-B-512",
        "DiT-L/2": "facebook/DiT-L-256" if image_size == 256 else "facebook/DiT-L-512",
        "DiT-XL/2": "facebook/DiT-XL-2-256" if image_size == 256 else "facebook/DiT-XL-2-512",
    }

    hf_name = hf_name_map.get(model_name)
    if hf_name is None:
        raise ValueError(f"No HuggingFace mapping for {model_name}")

    if pretrained:
        pipe = DiTPipeline.from_pretrained(hf_name, torch_dtype=dtype)
        model = pipe.transformer.to(device)
    else:
        pipe = DiTPipeline.from_pretrained(hf_name, torch_dtype=dtype)
        model = pipe.transformer
        model.apply(_init_weights)
        model = model.to(device)

    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("Loaded %s: %.1fM params on %s", model_name, n_params, device)
    return model


def _build_standalone_dit(
    model_name: str,
    image_size: int,
    num_classes: int,
    device: str,
    dtype: torch.dtype,
) -> nn.Module:
    """Build a minimal DiT wrapper for experiments without pretrained weights."""
    cfg = MODEL_CONFIGS.get(model_name)
    if cfg is None:
        raise ValueError(f"Unknown model: {model_name}")

    model = MinimalDiT(
        input_size=image_size // cfg["patch_size"],
        patch_size=cfg["patch_size"],
        in_channels=4,  # latent channels for VAE
        hidden_size=cfg["hidden_size"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        num_classes=num_classes,
    )
    model = model.to(device=device, dtype=dtype)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("Built standalone %s: %.1fM params on %s", model_name, n_params, device)
    return model


def _init_weights(module: nn.Module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class MinimalDiT(nn.Module):
    """
    Minimal DiT implementation for standalone experiments.
    Input: (B, C, H, W) latent, (B,) timestep, (B,) class label
    Output: (B, C, H, W) predicted noise
    """

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        num_classes: int = 1000,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_patches = (input_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            in_channels, hidden_size,
            kernel_size=patch_size, stride=patch_size,
        )
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, hidden_size) * 0.02
        )

        self.time_embed = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.class_embed = nn.Embedding(num_classes + 1, hidden_size)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])

        self.final_norm = nn.LayerNorm(hidden_size)
        self.final_proj = nn.Linear(hidden_size, patch_size * patch_size * in_channels)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self.patch_size

        x = self.patch_embed(x)  # (B, hidden, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, hidden)
        x = x + self.pos_embed

        t_emb = self._timestep_embedding(t, self.hidden_size).to(x.dtype)
        t_emb = self.time_embed(t_emb)

        if y is not None:
            c_emb = self.class_embed(y.long())
            cond = t_emb + c_emb
        else:
            cond = t_emb

        for block in self.blocks:
            x = block(x, cond)

        x = self.final_norm(x)
        x = self.final_proj(x)  # (B, num_patches, p*p*C)

        h = w = int(self.num_patches ** 0.5)
        x = x.reshape(B, h, w, p, p, self.in_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, self.in_channels, H, W)
        return x

    @staticmethod
    def _timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-torch.arange(half, device=t.device).float() * (torch.log(torch.tensor(10000.0)) / half))
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class DiTBlock(nn.Module):
    """Transformer block with adaptive layer norm (adaLN-Zero)."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 6),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN(cond).unsqueeze(1).chunk(6, dim=-1)
        )

        h = self.norm1(x) * (1 + scale_msa) + shift_msa
        h, _ = self.attn(h, h, h)
        x = x + gate_msa * h

        h = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        x = x + gate_mlp * h

        return x
