"""Optional VAE support with safe fallback to deterministic latent codec."""

import importlib.util
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def _freeze_module(module):
    module.eval()
    for param in module.parameters():
        param.requires_grad = False


class UNetAEFromCheckpoint(nn.Module):
    """Load ConditionalUNetAE from unet_ae_modular checkpoint."""

    def __init__(self, checkpoint_path):
        super().__init__()

        if not checkpoint_path or not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"UNet AE checkpoint not found: {checkpoint_path}"
            )

        unet_ae_model_path = os.path.join(
            os.path.dirname(__file__), "..", "unet_ae_modular", "model.py"
        )
        spec = importlib.util.spec_from_file_location("unet_ae_model", unet_ae_model_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not import UNet model from: {unet_ae_model_path}")

        unet_ae_model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(unet_ae_model_module)
        ConditionalUNetAE = unet_ae_model_module.ConditionalUNetAE

        self.unet_ae = ConditionalUNetAE()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        missing, unexpected = self.unet_ae.load_state_dict(state_dict, strict=False)
        print(f"[VAE] Loaded ConditionalUNetAE checkpoint from {checkpoint_path}")
        if missing:
            print(f"[VAE][WARN] Missing keys while loading UNet AE: {len(missing)}")
        if unexpected:
            print(f"[VAE][WARN] Unexpected keys while loading UNet AE: {len(unexpected)}")

    def encode(self, x, cond):
        """Encode image in [0, 1] to bottleneck latent."""
        _, _, _, bottleneck = self.unet_ae.encode(x, cond)
        return bottleneck

    def decode(self, z, cond, output_shape):
        """Decode latent using decoder path with zero skip placeholders."""
        bsz, _, h_lat, w_lat = z.shape
        device = z.device
        dtype = z.dtype

        e1_dummy = torch.zeros(bsz, 32, h_lat * 8, w_lat * 8, device=device, dtype=dtype)
        e2_dummy = torch.zeros(bsz, 64, h_lat * 4, w_lat * 4, device=device, dtype=dtype)
        e3_dummy = torch.zeros(bsz, 128, h_lat * 2, w_lat * 2, device=device, dtype=dtype)

        x_01 = self.unet_ae.decode(e1_dummy, e2_dummy, e3_dummy, z, cond)
        if x_01.shape[2:] != output_shape:
            x_01 = F.interpolate(x_01, size=output_shape, mode="bicubic", align_corners=False)
        return x_01.clamp(0.0, 1.0)

    def forward(self, x, cond):
        z = self.encode(x, cond)
        return self.decode(z, cond, (x.shape[2], x.shape[3]))

    def latent_shape(self, height, width):
        h_lat = max(1, int(height) // 8)
        w_lat = max(1, int(width) // 8)
        return h_lat, w_lat


class VAECodec(nn.Module):
    """Codec wrapper compatible with DiT pipeline (input/output in [-1, 1])."""

    def __init__(self, vae_model):
        super().__init__()
        self.vae = vae_model
        self.latent_channels = 256

    @staticmethod
    def _default_cond(batch_size, device, dtype):
        return torch.zeros(batch_size, 1, device=device, dtype=dtype)

    def encode(self, x, cond=None):
        if cond is None:
            cond = self._default_cond(x.shape[0], x.device, x.dtype)
        x_01 = torch.clamp((x + 1.0) * 0.5, 0.0, 1.0)
        return self.vae.encode(x_01, cond)

    def decode(self, z, output_shape, cond=None):
        if cond is None:
            cond = self._default_cond(z.shape[0], z.device, z.dtype)
        x_01 = self.vae.decode(z, cond, output_shape)
        x = x_01 * 2.0 - 1.0
        return x.clamp(-1.0, 1.0)

    def latent_shape(self, height, width):
        return self.vae.latent_shape(height, width)


def build_codec(
    device,
    use_vae=False,
    vae_checkpoint_path=None,
    fallback_compression_ratio=8,
    fallback_latent_channels=1,
    verbose=True,
):
    """Create VAE codec when possible, otherwise fallback to LatentCodec."""
    from Transformers_2.No_usar.Intento_Transformers.model import LatentCodec

    if use_vae:
        try:
            vae_model = UNetAEFromCheckpoint(vae_checkpoint_path).to(device)
            _freeze_module(vae_model)

            codec = VAECodec(vae_model).to(device)
            _freeze_module(codec)

            info = {
                "codec_type": "vae",
                "vae_checkpoint_path": vae_checkpoint_path,
                "latent_channels": int(codec.latent_channels),
                "compression_ratio": 8.0,
            }
            if verbose:
                print(f"[CODEC] Using VAE codec from: {vae_checkpoint_path}")
            return codec, info
        except Exception as exc:
            if verbose:
                print(f"[CODEC][WARN] VAE unavailable ({exc}); falling back to LatentCodec")

    codec = LatentCodec(
        compression_ratio=fallback_compression_ratio,
        latent_channels=fallback_latent_channels,
    ).to(device)
    codec.eval()

    info = {
        "codec_type": "latent",
        "vae_checkpoint_path": None,
        "latent_channels": int(fallback_latent_channels),
        "compression_ratio": float(fallback_compression_ratio),
    }
    if verbose:
        print(
            "[CODEC] Using deterministic LatentCodec "
            f"(ratio={fallback_compression_ratio}, channels={fallback_latent_channels})"
        )
    return codec, info


def build_codec_from_checkpoint(
    checkpoint,
    device,
    default_use_vae=False,
    default_vae_checkpoint_path=None,
    default_compression_ratio=8,
    default_latent_channels=1,
    verbose=True,
):
    """Rebuild codec from checkpoint metadata with backward compatibility."""

    state_dict = None
    for key in ("model_state_dict", "state_dict", "model", "ema", "ema_model_state_dict"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            state_dict = checkpoint[key]
            break

    codec_type = str(checkpoint.get("codec_type", "")).strip().lower()
    if "latent_channels" in checkpoint:
        latent_channels = checkpoint.get("latent_channels")
    elif state_dict is not None and "patch_embed.weight" in state_dict:
        latent_channels = int(state_dict["patch_embed.weight"].shape[1])
    else:
        latent_channels = default_latent_channels

    vae_checkpoint_path = checkpoint.get("vae_checkpoint_path", default_vae_checkpoint_path)

    if codec_type == "":
        inferred_vae = bool(vae_checkpoint_path) and int(latent_channels) >= 64
        use_vae = bool(default_use_vae) or inferred_vae
    else:
        use_vae = codec_type == "vae"

    compression_ratio = checkpoint.get("compression_ratio", default_compression_ratio)

    return build_codec(
        device=device,
        use_vae=use_vae,
        vae_checkpoint_path=vae_checkpoint_path,
        fallback_compression_ratio=int(round(float(compression_ratio))),
        fallback_latent_channels=int(latent_channels),
        verbose=verbose,
    )
