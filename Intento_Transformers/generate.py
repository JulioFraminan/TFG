import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import (
    ATTENTION_CHUNK_SIZE,
    DATA_FOLDER,
    DIFFUSION_STEPS,
    DIT_MODEL_PATH,
    GENERATION_DDIM_STEPS,
    GENERATE_ANGLES,
    GENERATE_MAT_FOLDER,
    GENERATE_PNG_FOLDER,
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
    create_all_dirs,
)
from data_utils import Normalizer, compute_error_metrics, load_all_rois, save_mat
from model import DiT, LatentCodec


def ddim_sample_latent(model, z_t, timesteps, conditioning=None, eta=0.0):
    """DDIM sampling loop in latent space."""
    x_t = z_t.clone()
    bsz = x_t.shape[0]
    device = x_t.device

    with torch.no_grad():
        for i, t in enumerate(timesteps):
            t_int = int(t)
            t_tensor = torch.full((bsz,), t_int, dtype=torch.long, device=device)

            eps = model(x_t, t_tensor, conditioning)

            alpha_t = model.diffusion_schedule.alphas_cumprod[t_int].to(dtype=x_t.dtype)
            if i < len(timesteps) - 1:
                alpha_prev = model.diffusion_schedule.alphas_cumprod[int(timesteps[i + 1])].to(dtype=x_t.dtype)
            else:
                alpha_prev = torch.tensor(1.0, device=device, dtype=x_t.dtype)

            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(torch.clamp(1.0 - alpha_t, min=0.0))

            x0_pred = (x_t - sqrt_one_minus_alpha_t * eps) / torch.clamp(sqrt_alpha_t, min=1e-8)

            sigma = eta * torch.sqrt(
                torch.clamp((1.0 - alpha_prev) / torch.clamp(1.0 - alpha_t, min=1e-8), min=0.0)
                * torch.clamp(1.0 - alpha_t / torch.clamp(alpha_prev, min=1e-8), min=0.0)
            )

            dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_prev - sigma ** 2, min=0.0)) * eps
            x_prev = torch.sqrt(torch.clamp(alpha_prev, min=0.0)) * x0_pred + dir_xt

            if eta > 0.0 and i < len(timesteps) - 1:
                x_prev = x_prev + sigma * torch.randn_like(x_t)

            x_t = x_prev

    return x_t


def generate_samples_ddim(
    model,
    codec,
    device,
    normalizer,
    angle,
    num_samples=1,
    num_steps=50,
    eta=0.0,
    output_shape=None,
):
    """Generate TL fields conditioned on angle using DDIM."""
    if output_shape is None:
        output_shape = (ROI_HEIGHT, ROI_WIDTH)

    num_steps = max(1, min(int(num_steps), model.diffusion_schedule.num_steps))
    timesteps = np.linspace(model.diffusion_schedule.num_steps - 1, 0, num_steps, dtype=int)

    cond_value = float(normalizer.normalize_angle(angle))
    cond_tensor = torch.full((num_samples, 1), cond_value, dtype=torch.float32, device=device)

    latent_h, latent_w = codec.latent_shape(output_shape[0], output_shape[1])
    z_t = torch.randn(num_samples, model.latent_channels, latent_h, latent_w, device=device)

    z_0 = ddim_sample_latent(
        model=model,
        z_t=z_t,
        timesteps=timesteps,
        conditioning=cond_tensor,
        eta=eta,
    )

    x_0 = codec.decode(z_0, output_shape=output_shape)
    gen_np = x_0.detach().cpu().numpy()[:, 0, :, :]

    gen_norm_01 = np.clip((gen_np + 1.0) / 2.0, 0.0, 1.0)
    gen_tl = normalizer.denormalize_tl(gen_norm_01)

    if num_samples == 1:
        return gen_tl[0], gen_norm_01[0]
    return gen_tl, gen_norm_01


def _build_model_from_checkpoint(checkpoint, device):
    latent_channels = int(checkpoint.get("latent_channels", VAE_LATENT_CHANNELS))
    hidden_size = int(checkpoint.get("hidden_size", MODEL_HIDDEN_SIZE))
    depth = int(checkpoint.get("depth", MODEL_DEPTH))
    num_heads = int(checkpoint.get("num_heads", MODEL_NUM_HEADS))
    mlp_ratio = float(checkpoint.get("mlp_ratio", MLP_RATIO))
    patch_size = int(checkpoint.get("patch_size", MODEL_PATCH_SIZE))
    diffusion_steps = int(checkpoint.get("diffusion_steps", DIFFUSION_STEPS))
    attention_chunk_size = int(checkpoint.get("attention_chunk_size", ATTENTION_CHUNK_SIZE))

    model = DiT(
        latent_channels=latent_channels,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        num_diffusion_steps=diffusion_steps,
        cond_dim=1,
        patch_size=patch_size,
        attention_chunk_size=attention_chunk_size,
    ).to(device)

    state_key = "ema_model_state_dict" if "ema_model_state_dict" in checkpoint else "model_state_dict"
    missing, unexpected = model.load_state_dict(checkpoint[state_key], strict=False)
    print(f"Using checkpoint weights: {state_key}")
    if missing:
        print(f"[WARN] Missing keys while loading checkpoint: {len(missing)}")
    if unexpected:
        print(f"[WARN] Unexpected keys while loading checkpoint: {len(unexpected)}")

    model.eval()
    return model


def main():
    create_all_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not os.path.exists(DIT_MODEL_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found: {DIT_MODEL_PATH}. Train first with train.py"
        )

    checkpoint = torch.load(DIT_MODEL_PATH, map_location=device)
    model = _build_model_from_checkpoint(checkpoint, device)

    compression_ratio = int(checkpoint.get("compression_ratio", VAE_COMPRESSION_RATIO))
    latent_channels = int(checkpoint.get("latent_channels", VAE_LATENT_CHANNELS))
    roi_h = int(checkpoint.get("roi_height", ROI_HEIGHT))
    roi_w = int(checkpoint.get("roi_width", ROI_WIDTH))

    codec = LatentCodec(
        compression_ratio=compression_ratio,
        latent_channels=latent_channels,
    ).to(device)

    if all(k in checkpoint for k in ("tl_min", "tl_max", "angle_min", "angle_max")):
        norm = Normalizer.from_checkpoint(checkpoint)
        print("Normalizer restored from checkpoint")
    else:
        print("Checkpoint has no normalizer values; rebuilding from training data")
        rois_tmp, angles_tmp, _ = load_all_rois(
            DATA_FOLDER,
            roi_h,
            roi_w,
            ROIS_PER_PLANE,
            roi_mode=ROI_MODE,
            roi_corner=ROI_CORNER,
        )
        norm = Normalizer(rois_tmp, angles_tmp)

    ref_rois, ref_angles, ref_extents = load_all_rois(
        DATA_FOLDER,
        roi_h,
        roi_w,
        ROIS_PER_PLANE,
        roi_mode=ROI_MODE,
        roi_corner=ROI_CORNER,
    )
    has_ref = len(ref_rois) > 0

    print("=" * 70)
    print(f"Generating {len(GENERATE_ANGLES)} conditioned samples")
    print("=" * 70)

    for target_angle in GENERATE_ANGLES:
        gen_tl, _ = generate_samples_ddim(
            model=model,
            codec=codec,
            device=device,
            normalizer=norm,
            angle=float(target_angle),
            num_samples=1,
            num_steps=min(GENERATION_DDIM_STEPS, model.diffusion_schedule.num_steps),
            eta=0.0,
            output_shape=(roi_h, roi_w),
        )

        mat_path = os.path.join(GENERATE_MAT_FOLDER, f"plano_angulo_{target_angle:+07.2f}.mat")
        save_mat(mat_path, gen_tl)

        if has_ref:
            angle_arr = np.array(ref_angles, dtype=np.float32)
            idx = int(np.argmin(np.abs(angle_arr - float(target_angle))))
            seed_tl = ref_rois[idx]
            extent = ref_extents[idx]
            metrics = compute_error_metrics(seed_tl, gen_tl)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            im0 = axes[0].imshow(
                seed_tl,
                cmap="jet",
                aspect="auto",
                origin="lower",
                vmin=norm.tl_min,
                vmax=norm.tl_max,
                extent=extent,
            )
            axes[0].set_title(f"Nearest training ({ref_angles[idx]:.2f} deg)")
            plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

            im1 = axes[1].imshow(
                gen_tl,
                cmap="jet",
                aspect="auto",
                origin="lower",
                vmin=norm.tl_min,
                vmax=norm.tl_max,
                extent=extent,
            )
            axes[1].set_title(f"Generated ({target_angle:.2f} deg)")
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            err_map = np.abs(seed_tl - gen_tl)
            im2 = axes[2].imshow(err_map, cmap="hot", aspect="auto", origin="lower", extent=extent)
            axes[2].set_title(
                f"|Error| MAE={metrics['mae']:.3f} RMSE={metrics['rmse']:.3f}"
            )
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

            fig.suptitle(f"DiT generation for angle {target_angle:.2f} deg", fontsize=13)
            fig.tight_layout()

            print(
                f"angle={target_angle:+07.2f} | ref={ref_angles[idx]:+07.2f} "
                f"| MAE={metrics['mae']:.3f} RMSE={metrics['rmse']:.3f}"
            )
        else:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            im = ax.imshow(
                gen_tl,
                cmap="jet",
                aspect="auto",
                origin="lower",
                vmin=norm.tl_min,
                vmax=norm.tl_max,
            )
            ax.set_title(f"Generated ({target_angle:.2f} deg)")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            print(f"angle={target_angle:+07.2f} generated")

        png_path = os.path.join(GENERATE_PNG_FOLDER, f"plano_angulo_{target_angle:+07.2f}.png")
        fig.savefig(png_path, dpi=140)
        plt.close(fig)

    print("=" * 70)
    print(f"PNG outputs: {GENERATE_PNG_FOLDER}")
    print(f"MAT outputs: {GENERATE_MAT_FOLDER}")
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()