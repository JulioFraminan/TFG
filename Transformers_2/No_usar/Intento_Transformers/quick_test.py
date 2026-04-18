#!/usr/bin/env python3
"""Quick checkpoint sanity check (no training)."""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import matplotlib.pyplot as plt
import numpy as np
import torch

from Transformers_2.No_usar.Intento_Transformers.config import (
    DATA_FOLDER,
    DIT_MODEL_PATH,
    ROI_CORNER,
    ROI_HEIGHT,
    ROI_MODE,
    ROI_WIDTH,
    ROIS_PER_PLANE,
    USE_VAE,
    VAE_COMPRESSION_RATIO,
    VAE_CHECKPOINT_PATH,
    VAE_LATENT_CHANNELS,
)
from Transformers_2.No_usar.Intento_Transformers.data_utils import Normalizer, load_all_rois
from Transformers_2.No_usar.Intento_Transformers.generate import _build_model_from_checkpoint, _load_checkpoint_with_recovery, generate_samples_ddim
from Transformers_2.No_usar.Intento_Transformers.vae import build_codec_from_checkpoint


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not os.path.exists(DIT_MODEL_PATH):
        print(f"Checkpoint not found: {DIT_MODEL_PATH}")
        print("Run train.py first.")
        sys.exit(1)

    checkpoint, ckpt_path_used = _load_checkpoint_with_recovery(DIT_MODEL_PATH, device)
    print(f"Loaded checkpoint: {ckpt_path_used}")

    diffusion_steps = int(checkpoint.get("diffusion_steps", 1000))

    roi_h = int(checkpoint.get("roi_height", ROI_HEIGHT))
    roi_w = int(checkpoint.get("roi_width", ROI_WIDTH))
    model = _build_model_from_checkpoint(checkpoint, device)

    codec, codec_info = build_codec_from_checkpoint(
        checkpoint=checkpoint,
        device=device,
        default_use_vae=USE_VAE,
        default_vae_checkpoint_path=VAE_CHECKPOINT_PATH,
        default_compression_ratio=VAE_COMPRESSION_RATIO,
        default_latent_channels=VAE_LATENT_CHANNELS,
        verbose=True,
    )
    print(f"Codec restored: {codec_info.get('codec_type', 'latent')}")

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