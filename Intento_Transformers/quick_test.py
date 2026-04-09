#!/usr/bin/env python3
"""Quick checkpoint sanity check (no training)."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import (
    ATTENTION_CHUNK_SIZE,
    DATA_FOLDER,
    DIT_MODEL_PATH,
    MLP_RATIO,
    MODEL_DEPTH,
    MODEL_HIDDEN_SIZE,
    MODEL_NUM_HEADS,
    MODEL_PATCH_SIZE,
    ROI_CORNER,
    ROI_HEIGHT,
    ROI_MODE,
    ROI_WIDTH,
    ROIS_PER_PLANE,
    VAE_COMPRESSION_RATIO,
    VAE_LATENT_CHANNELS,
)
from data_utils import Normalizer, load_all_rois
from generate import generate_samples_ddim
from model import DiT, LatentCodec


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not os.path.exists(DIT_MODEL_PATH):
        print(f"Checkpoint not found: {DIT_MODEL_PATH}")
        print("Run train.py first.")
        sys.exit(1)

    checkpoint = torch.load(DIT_MODEL_PATH, map_location=device)

    latent_channels = int(checkpoint.get("latent_channels", VAE_LATENT_CHANNELS))
    hidden_size = int(checkpoint.get("hidden_size", MODEL_HIDDEN_SIZE))
    depth = int(checkpoint.get("depth", MODEL_DEPTH))
    num_heads = int(checkpoint.get("num_heads", MODEL_NUM_HEADS))
    mlp_ratio = float(checkpoint.get("mlp_ratio", MLP_RATIO))
    patch_size = int(checkpoint.get("patch_size", MODEL_PATCH_SIZE))
    diffusion_steps = int(checkpoint.get("diffusion_steps", 1000))
    attention_chunk_size = int(checkpoint.get("attention_chunk_size", ATTENTION_CHUNK_SIZE))

    roi_h = int(checkpoint.get("roi_height", ROI_HEIGHT))
    roi_w = int(checkpoint.get("roi_width", ROI_WIDTH))
    compression_ratio = int(checkpoint.get("compression_ratio", VAE_COMPRESSION_RATIO))

    model = DiT(
        latent_channels=latent_channels,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        num_diffusion_steps=diffusion_steps,
        patch_size=patch_size,
        attention_chunk_size=attention_chunk_size,
    ).to(device)
    state_key = "ema_model_state_dict" if "ema_model_state_dict" in checkpoint else "model_state_dict"
    model.load_state_dict(checkpoint[state_key], strict=False)
    print(f"Using checkpoint weights: {state_key}")
    model.eval()

    codec = LatentCodec(
        compression_ratio=compression_ratio,
        latent_channels=latent_channels,
    ).to(device)

    if all(k in checkpoint for k in ("tl_min", "tl_max", "angle_min", "angle_max")):
        norm = Normalizer.from_checkpoint(checkpoint)
        print("Normalizer loaded from checkpoint")
    else:
        rois, angles, _ = load_all_rois(
            DATA_FOLDER,
            roi_h,
            roi_w,
            ROIS_PER_PLANE,
            roi_mode=ROI_MODE,
            roi_corner=ROI_CORNER,
        )
        norm = Normalizer(rois, angles)
        print("Normalizer rebuilt from data")

    test_angle = 0.5 * (norm.angle_min + norm.angle_max)
    print(f"Generating one sample at angle {test_angle:.3f}")

    gen_tl, gen_norm = generate_samples_ddim(
        model=model,
        codec=codec,
        device=device,
        normalizer=norm,
        angle=test_angle,
        num_samples=1,
        num_steps=min(20, diffusion_steps),
        eta=0.0,
        output_shape=(roi_h, roi_w),
    )

    out_png = os.path.join("/tmp", "dit_quick_test.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    im0 = axes[0].imshow(gen_norm, cmap="jet", aspect="auto", origin="lower", vmin=0, vmax=1)
    axes[0].set_title("Generated [0, 1]")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(
        gen_tl,
        cmap="jet",
        aspect="auto",
        origin="lower",
        vmin=norm.tl_min,
        vmax=norm.tl_max,
    )
    axes[1].set_title("Generated TL (dB)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)

    print(f"Saved quick test plot: {out_png}")
    print(
        "TL stats: "
        f"min={float(np.min(gen_tl)):.3f}, "
        f"max={float(np.max(gen_tl)):.3f}, "
        f"mean={float(np.mean(gen_tl)):.3f}"
    )


if __name__ == "__main__":
    main()