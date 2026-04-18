"""
Core DiT components for TL field generation.

This version is designed to be robust with large ROIs by:
1) running diffusion in a compact latent space,
2) using patch tokens inside the transformer to avoid N^2 memory blowups.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint


def modulate(x, shift, scale):
    """FiLM-style modulation used in AdaLN blocks."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _get_1d_sincos_pos_embed(embed_dim, positions):
    """Create standard 1D sin-cos positional embedding."""
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even for sin-cos embedding, got {embed_dim}")

    half_dim = embed_dim // 2
    if half_dim == 0:
        return torch.zeros((positions.shape[0], 0), device=positions.device, dtype=torch.float32)

    omega = torch.arange(half_dim, device=positions.device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / half_dim))
    out = positions[:, None] * omega[None, :]
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


def get_2d_sincos_pos_embed(embed_dim, grid_shape, device):
    """Create 2D sin-cos positional embeddings with output dim == embed_dim."""
    if embed_dim % 4 != 0:
        raise ValueError(f"embed_dim must be divisible by 4, got {embed_dim}")

    h, w = grid_shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )

    grid_y = grid_y.reshape(-1)
    grid_x = grid_x.reshape(-1)

    emb_y = _get_1d_sincos_pos_embed(embed_dim // 2, grid_y)
    emb_x = _get_1d_sincos_pos_embed(embed_dim // 2, grid_x)
    return torch.cat([emb_x, emb_y], dim=1)


class TimestepEmbedder(nn.Module):
    """Sinusoidal diffusion timestep embedding followed by MLP."""

    def __init__(self, hidden_size, freq_emb_size=256):
        super().__init__()
        self.freq_emb_size = freq_emb_size
        self.mlp = nn.Sequential(
            nn.Linear(freq_emb_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.freq_emb_size))


class ConditionEmbedder(nn.Module):
    """Embed scalar conditioning (e.g. normalized angle)."""

    def __init__(self, cond_dim, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, cond):
        return self.mlp(cond)


class Attention(nn.Module):
    """Multi-head self-attention with chunked computation for ROCm stability."""

    def __init__(self, dim, num_heads=8, qkv_bias=True, attention_chunk_size=256):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.attention_chunk_size = int(attention_chunk_size)
        self._runtime_chunk_size = int(attention_chunk_size)
        self._min_chunk_size = 4

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    @staticmethod
    def _is_oom_error(err):
        msg = str(err).lower()
        return isinstance(err, torch.cuda.OutOfMemoryError) or "out of memory" in msg

    def _attention_core(self, q, k, v):
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_scores = attn_scores - attn_scores.amax(dim=-1, keepdim=True)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_probs, v)

    def _chunked_attention(self, q, k, v):
        bsz, n_heads, n_tokens, head_dim = q.shape
        configured = max(0, self.attention_chunk_size)
        runtime = max(0, self._runtime_chunk_size)
        chunk = configured if configured > 0 else n_tokens
        if runtime > 0:
            chunk = min(chunk, runtime)

        min_chunk = self._min_chunk_size
        while True:
            try:
                if chunk <= 0 or chunk >= n_tokens:
                    out = self._attention_core(q, k, v)
                else:
                    out = torch.empty((bsz, n_heads, n_tokens, head_dim), device=q.device, dtype=q.dtype)
                    for start in range(0, n_tokens, chunk):
                        end = min(start + chunk, n_tokens)
                        out[:, :, start:end, :] = self._attention_core(q[:, :, start:end, :], k, v)

                self._runtime_chunk_size = chunk
                return out
            except RuntimeError as err:
                if not self._is_oom_error(err) or chunk <= min_chunk:
                    raise

                if q.device.type == "cuda":
                    torch.cuda.empty_cache()

                chunk = max(min_chunk, chunk // 2)

    def reduce_runtime_chunk(self, factor=2):
        """Reduce runtime chunk size for OOM recovery."""
        factor = max(2, int(factor))
        current = max(self._min_chunk_size, int(self._runtime_chunk_size))
        self._runtime_chunk_size = max(self._min_chunk_size, current // factor)

    def get_runtime_chunk(self):
        return int(self._runtime_chunk_size)

    def forward(self, x):
        bsz, n_tokens, dim = x.shape
        qkv = self.qkv(x).reshape(bsz, n_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_out = self._chunked_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(bsz, n_tokens, dim)
        return self.proj(attn_out)


class MLPBlock(nn.Module):
    """Transformer MLP block."""

    def __init__(self, hidden_size, mlp_ratio=4.0):
        super().__init__()
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.fc1 = nn.Linear(hidden_size, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, hidden_size)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class DiTBlock(nn.Module):
    """DiT block with AdaLN-Zero conditioning."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attention_chunk_size=256):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            attention_chunk_size=attention_chunk_size,
        )

        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp = MLPBlock(hidden_size, mlp_ratio)

        self.ada_ln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

    def forward(self, x, c):
        shift_attn, scale_attn, shift_mlp, scale_mlp, gate_attn, gate_mlp = self.ada_ln(c).chunk(6, dim=1)

        x = x + gate_attn.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_attn, scale_attn))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DiffusionSchedule(nn.Module):
    """Linear variance schedule with cached cumulative products."""

    def __init__(self, num_steps=1000, beta_start=1e-4, beta_end=2e-2):
        super().__init__()
        self.num_steps = int(num_steps)

        betas = torch.linspace(beta_start, beta_end, self.num_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def q_sample(self, x0, t, noise=None):
        """Forward diffusion: q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t]

        while sqrt_alpha.ndim < x0.ndim:
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)

        xt = sqrt_alpha * x0 + sqrt_one_minus * noise
        return xt, noise


class LatentCodec(nn.Module):
    """Deterministic latent codec (downsample/upsample), no external VAE required."""

    def __init__(self, compression_ratio=8, latent_channels=1):
        super().__init__()
        self.compression_ratio = max(1, int(compression_ratio))
        self.latent_channels = max(1, int(latent_channels))

    def latent_shape(self, height, width):
        h_lat = math.ceil(height / self.compression_ratio)
        w_lat = math.ceil(width / self.compression_ratio)
        return h_lat, w_lat

    def encode(self, x, cond=None):
        """Encode image tensor (B, 1, H, W) to latent tensor (B, C_lat, H_lat, W_lat)."""
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"encode expects shape (B,1,H,W), got {tuple(x.shape)}")

        h_lat, w_lat = self.latent_shape(x.shape[2], x.shape[3])
        z = F.interpolate(x, size=(h_lat, w_lat), mode="area")
        if self.latent_channels > 1:
            z = z.repeat(1, self.latent_channels, 1, 1)
        return z

    def decode(self, z, output_shape, cond=None):
        """Decode latent tensor back to image tensor (B,1,H,W)."""
        if z.ndim != 4:
            raise ValueError(f"decode expects shape (B,C,H,W), got {tuple(z.shape)}")

        if z.shape[1] == 1:
            x = z
        else:
            x = z.mean(dim=1, keepdim=True)

        # Bicubic interpolation for smooth upsampling
        x = F.interpolate(x, size=output_shape, mode="bicubic", align_corners=False)
        
        # Apply multi-scale smoothing to reduce pixelation artifacts
        x_smooth = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), kernel_size=3, stride=1)
        x_smooth = x_smooth[:, :, :output_shape[0], :output_shape[1]]  # Crop to output size
        x = 0.85 * x + 0.15 * x_smooth  # Increased smoothing weight for less pixelated output
        
        return x.clamp(-1.0, 1.0)


class DiT(nn.Module):
    """Diffusion Transformer in latent space with patch tokenization."""

    def __init__(
        self,
        latent_channels=1,
        hidden_size=256,
        depth=6,
        num_heads=4,
        mlp_ratio=4.0,
        num_diffusion_steps=1000,
        beta_start=1e-4,
        beta_end=2e-2,
        cond_dim=1,
        patch_size=4,
        attention_chunk_size=256,
        gradient_checkpointing=False,
    ):
        super().__init__()

        self.latent_channels = int(latent_channels)
        self.hidden_size = int(hidden_size)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.patch_size = int(patch_size)
        self.attention_chunk_size = int(attention_chunk_size)
        self.gradient_checkpointing = bool(gradient_checkpointing)

        if self.patch_size < 1:
            raise ValueError(f"patch_size must be >= 1, got {self.patch_size}")

        self.t_embedder = TimestepEmbedder(self.hidden_size)
        self.cond_embedder = ConditionEmbedder(cond_dim, self.hidden_size)

        self.patch_embed = nn.Conv2d(
            in_channels=self.latent_channels,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio,
                    attention_chunk_size=self.attention_chunk_size,
                )
                for _ in range(self.depth)
            ]
        )

        self.norm_final = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.ada_ln_final = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, 2 * self.hidden_size),
        )
        self.output_proj = nn.Linear(
            self.hidden_size,
            self.patch_size * self.patch_size * self.latent_channels,
        )
        self.output_refine = nn.Conv2d(
            self.latent_channels,
            self.latent_channels,
            kernel_size=3,
            padding=1,
            groups=self.latent_channels,
            bias=True,
        )
        nn.init.zeros_(self.output_refine.weight)
        nn.init.zeros_(self.output_refine.bias)

        self.diffusion_schedule = DiffusionSchedule(
            num_steps=num_diffusion_steps,
            beta_start=beta_start,
            beta_end=beta_end,
        )

        self.register_buffer("_pos_embed", torch.empty(0), persistent=False)
        self._pos_shape = None

    def _get_pos_embed(self, h_tokens, w_tokens, device):
        shape = (h_tokens, w_tokens)
        if (
            self._pos_shape != shape
            or self._pos_embed.numel() == 0
            or self._pos_embed.device != device
        ):
            self._pos_embed = get_2d_sincos_pos_embed(self.hidden_size, shape, device)
            self._pos_shape = shape
        return self._pos_embed

    def reduce_attention_runtime_chunk(self, factor=2):
        """Reduce runtime attention chunk in all blocks (used by OOM recovery)."""
        for block in self.blocks:
            block.attn.reduce_runtime_chunk(factor=factor)

    def get_attention_runtime_chunk(self):
        if not self.blocks:
            return 0
        return self.blocks[0].attn.get_runtime_chunk()

    def forward(self, z, t, cond=None):
        """
        Args:
            z: (B, C_lat, H_lat, W_lat)
            t: (B,)
            cond: (B, cond_dim)
        Returns:
            noise prediction with same shape as z.
        """
        bsz, channels, height, width = z.shape
        if channels != self.latent_channels:
            raise ValueError(
                f"Expected {self.latent_channels} latent channels, got {channels}. "
                "Check config VAE_LATENT_CHANNELS and checkpoint compatibility."
            )

        pad_h = (self.patch_size - (height % self.patch_size)) % self.patch_size
        pad_w = (self.patch_size - (width % self.patch_size)) % self.patch_size
        z_pad = F.pad(z, (0, pad_w, 0, pad_h), mode="replicate")

        x = self.patch_embed(z_pad)
        _, _, h_tokens, w_tokens = x.shape
        n_tokens = h_tokens * w_tokens

        x = x.flatten(2).transpose(1, 2)
        pos = self._get_pos_embed(h_tokens, w_tokens, z.device)
        x = x + pos.unsqueeze(0)

        t_emb = self.t_embedder(t)
        if cond is not None:
            c = t_emb + self.cond_embedder(cond)
        else:
            c = t_emb

        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                x = torch_checkpoint(block, x, c, use_reentrant=False)
            else:
                x = block(x, c)

        shift, scale = self.ada_ln_final(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.output_proj(x)

        noise_pad = x.reshape(
            bsz,
            h_tokens,
            w_tokens,
            self.patch_size,
            self.patch_size,
            self.latent_channels,
        )
        noise_pad = noise_pad.permute(0, 5, 1, 3, 2, 4).reshape(
            bsz,
            self.latent_channels,
            h_tokens * self.patch_size,
            w_tokens * self.patch_size,
        )

        if noise_pad.shape[2] != height or noise_pad.shape[3] != width:
            noise = noise_pad[:, :, :height, :width]
        else:
            noise = noise_pad

        noise = noise + self.output_refine(noise)

        return noise


if __name__ == "__main__":
    model = DiT(
        latent_channels=1,
        hidden_size=128,
        depth=3,
        num_heads=2,
        mlp_ratio=4.0,
        num_diffusion_steps=1000,
        cond_dim=1,
        patch_size=4,
        attention_chunk_size=256,
        gradient_checkpointing=True,
    )

    z = torch.randn(2, 1, 88, 250)
    t = torch.randint(0, 1000, (2,))
    cond = torch.randn(2, 1)

    noise = model(z, t, cond)
    print(f"Input latent:  {tuple(z.shape)}")
    print(f"Output noise:  {tuple(noise.shape)}")
    print("DiT forward OK")