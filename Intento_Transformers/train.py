import os
import time
import math
import random
import sys
import gc
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from config import (
    ATTENTION_CHUNK_SIZE,
    BATCH_SIZE,
    BETA_END,
    BETA_START,
    DATA_FOLDER,
    DIFFUSION_SCHEDULE,
    DIFFUSION_STEPS,
    DIT_MODEL_PATH,
    EMA_DECAY,
    EPOCHS,
    GRAD_ACCUM_STEPS,
    GRADIENT_CHECKPOINTING,
    LEARNING_RATE,
    MLP_RATIO,
    MODEL_DEPTH,
    MODEL_HIDDEN_SIZE,
    MODEL_NUM_HEADS,
    MODEL_PATCH_SIZE,
    NUM_VALIDATION_ANGLES,
    ROI_CORNER,
    ROI_HEIGHT,
    ROI_MODE,
    ROI_WIDTH,
    ROIS_PER_PLANE,
    TRAIN_PNG_FOLDER,
    USE_AUGMENTATION,
    USE_VAE,
    VAE_COMPRESSION_RATIO,
    VAE_CHECKPOINT_PATH,
    VAE_LATENT_CHANNELS,
    VALIDATION_DDIM_STEPS,
    VALIDATION_FOLDER,
    VALIDATION_MAT_FOLDER,
    VALIDATION_PNG_FOLDER,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    create_all_dirs,
)
from data_utils import (
    Normalizer,
    augment,
    compute_error_metrics,
    load_all_rois,
    load_validation_rois,
    normalize_to_neg1_1,
    save_mat,
)
from generate import generate_samples_ddim
from model import DiT
from vae import build_codec


def format_seconds(seconds):
    seconds = int(max(0, seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def build_scheduler(optimizer, total_steps, warmup_steps):
    warmup_steps = max(1, int(warmup_steps))
    total_steps = max(1, int(total_steps))
    min_lr_ratio = 0.1

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        decay_span = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, float(step - warmup_steps) / float(decay_span)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _state_clone(state_dict):
    return {k: v.detach().clone() for k, v in state_dict.items()}


def _state_to_cpu(state_dict):
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def _load_state(model, src_state):
    current_state = model.state_dict()
    mapped = {
        k: src_state[k].to(device=current_state[k].device, dtype=current_state[k].dtype)
        for k in current_state.keys()
        if k in src_state
    }
    model.load_state_dict(mapped, strict=False)


def _update_ema(ema_state, model, decay):
    one_minus_decay = 1.0 - decay
    with torch.no_grad():
        for key, value in model.state_dict().items():
            ema_state[key].mul_(decay).add_(value.detach(), alpha=one_minus_decay)


def _is_oom_error(err):
    msg = str(err).lower()
    return isinstance(err, torch.cuda.OutOfMemoryError) or "out of memory" in msg


def _atomic_torch_save(payload, path):
    """Write checkpoints atomically to avoid truncated/corrupted .pt files."""
    tmp_path = f"{path}.tmp"
    prev_path = f"{path}.prev"

    try:
        torch.save(payload, tmp_path)
        if os.path.exists(path):
            try:
                shutil.copy2(path, prev_path)
            except OSError as err:
                print(f"[WARN] Could not create checkpoint backup {prev_path}: {err}")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def main():
    create_all_dirs()

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    use_pin = device.type == "cuda"

    print(f"Device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    print("=" * 70)
    print("Loading data and extracting ROIs")
    print("=" * 70)
    rois, roi_angles, _ = load_all_rois(
        DATA_FOLDER,
        ROI_HEIGHT,
        ROI_WIDTH,
        ROIS_PER_PLANE,
        roi_mode=ROI_MODE,
        roi_corner=ROI_CORNER,
    )
    if len(rois) == 0:
        raise RuntimeError(
            "No training ROIs found. Check input files and ROI settings in config.py"
        )

    norm = Normalizer(rois, roi_angles)
    rois_norm_01 = norm.normalize_tl(rois)
    angles_norm = norm.normalize_angles(roi_angles)

    print(f"ROIs loaded: {len(rois_norm_01)}")
    print(f"ROI shape: {rois_norm_01.shape[1]} x {rois_norm_01.shape[2]}")
    print(f"TL range: [{norm.tl_min:.3f}, {norm.tl_max:.3f}]")
    print(f"Angle range: [{norm.angle_min:.3f}, {norm.angle_max:.3f}]")

    if USE_AUGMENTATION:
        rois_train_01, angles_train = augment(rois_norm_01, angles_norm)
        print(f"Augmentation: ON ({len(rois_train_01)} samples)")
    else:
        rois_train_01 = rois_norm_01.copy()
        angles_train = np.array(angles_norm, dtype=np.float32)
        print(f"Augmentation: OFF ({len(rois_train_01)} samples)")

    rois_train = normalize_to_neg1_1(rois_train_01)

    shuffle_idx = np.random.permutation(len(rois_train))
    rois_train = rois_train[shuffle_idx]
    angles_train = angles_train[shuffle_idx]

    tensor_imgs = torch.from_numpy(rois_train[:, None, :, :]).float()
    tensor_angs = torch.from_numpy(angles_train[:, None]).float()

    loader = DataLoader(
        TensorDataset(tensor_imgs, tensor_angs),
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=use_pin,
        num_workers=0,
    )
    print(f"Batches per epoch: {len(loader)}")

    codec, codec_info = build_codec(
        device=device,
        use_vae=USE_VAE,
        vae_checkpoint_path=VAE_CHECKPOINT_PATH,
        fallback_compression_ratio=VAE_COMPRESSION_RATIO,
        fallback_latent_channels=VAE_LATENT_CHANNELS,
        verbose=True,
    )
    latent_h, latent_w = codec.latent_shape(ROI_HEIGHT, ROI_WIDTH)
    latent_channels = int(getattr(codec, "latent_channels", VAE_LATENT_CHANNELS))
    compression_ratio_h = float(ROI_HEIGHT) / float(latent_h)
    compression_ratio_w = float(ROI_WIDTH) / float(latent_w)
    print(
        f"Latent shape: {latent_h} x {latent_w} "
        f"(channels={latent_channels}, codec={codec_info.get('codec_type', 'latent')})"
    )

    model = DiT(
        latent_channels=latent_channels,
        hidden_size=MODEL_HIDDEN_SIZE,
        depth=MODEL_DEPTH,
        num_heads=MODEL_NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        num_diffusion_steps=DIFFUSION_STEPS,
        beta_start=BETA_START,
        beta_end=BETA_END,
        cond_dim=1,
        patch_size=MODEL_PATCH_SIZE,
        attention_chunk_size=ATTENTION_CHUNK_SIZE,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=" * 70)
    print("Training DiT")
    print("=" * 70)
    print(f"Trainable params: {n_params:,}")
    print(
        "Model config: "
        f"hidden={MODEL_HIDDEN_SIZE}, depth={MODEL_DEPTH}, heads={MODEL_NUM_HEADS}, "
        f"patch={MODEL_PATCH_SIZE}, attn_chunk={ATTENTION_CHUNK_SIZE}"
    )
    print(
        f"Optimization config: grad_accum={GRAD_ACCUM_STEPS}, ema_decay={EMA_DECAY}, "
        f"val_ddim_steps={VALIDATION_DDIM_STEPS}, ckpt={GRADIENT_CHECKPOINTING}"
    )

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    grad_accum = max(1, int(GRAD_ACCUM_STEPS))
    optimizer_steps_per_epoch = max(1, math.ceil(len(loader) / grad_accum))
    total_steps = max(1, EPOCHS * optimizer_steps_per_epoch)
    scheduler = build_scheduler(optimizer, total_steps, WARMUP_STEPS)
    criterion = nn.MSELoss()
    ema_state = _state_clone(model.state_dict())

    history = {"loss": [], "lr": []}
    smooth_loss = None
    smooth_beta = 0.92
    global_step = 0
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (batch_img, batch_ang) in enumerate(loader):
            batch_img = batch_img.to(device, non_blocking=use_pin)
            batch_ang = batch_ang.to(device, non_blocking=use_pin)

            max_oom_retries = 6
            for oom_try in range(max_oom_retries + 1):
                try:
                    t = torch.randint(0, DIFFUSION_STEPS, (batch_img.shape[0],), device=device)
                    with torch.no_grad():
                        z0 = codec.encode(batch_img, cond=batch_ang).detach()
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                        z_t, noise_target = model.diffusion_schedule.q_sample(z0, t)
                        noise_pred = model(z_t, t, batch_ang)
                        diffusion_loss = criterion(noise_pred, noise_target)
                        loss = diffusion_loss / grad_accum

                    scaler.scale(loss).backward()
                    diffusion_loss_value = float(diffusion_loss.item())
                    break
                except RuntimeError as err:
                    if not _is_oom_error(err):
                        raise
                    if oom_try >= max_oom_retries:
                        raise

                    optimizer.zero_grad(set_to_none=True)
                    gc.collect()

                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                    if (
                        oom_try >= 2
                        and hasattr(model, "gradient_checkpointing")
                        and not model.gradient_checkpointing
                        and model.get_attention_runtime_chunk() <= 80
                    ):
                        model.gradient_checkpointing = True
                        print("[OOM-RECOVERY] Enabled gradient checkpointing and retrying batch")

                    model.reduce_attention_runtime_chunk(factor=2)
                    print(
                        f"[OOM-RECOVERY] Reduced attention runtime chunk to "
                        f"{model.get_attention_runtime_chunk()} and retrying batch"
                    )

            do_step = ((batch_idx + 1) % grad_accum == 0) or (batch_idx + 1 == len(loader))
            if do_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                _update_ema(ema_state, model, EMA_DECAY)

            epoch_loss += diffusion_loss_value
            n_batches += 1
            global_step += 1

        avg_loss = epoch_loss / max(1, n_batches)
        if smooth_loss is None:
            smooth_loss = avg_loss
        else:
            smooth_loss = smooth_beta * smooth_loss + (1.0 - smooth_beta) * avg_loss
        history["loss"].append(avg_loss)
        history["lr"].append(scheduler.get_last_lr()[0])

        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            elapsed = time.time() - t0
            eta = elapsed / max(1, epoch) * (EPOCHS - epoch)
            print(
                f"Epoch {epoch:4d}/{EPOCHS} | loss={avg_loss:.6f} "
                f"| smooth={smooth_loss:.6f} "
                f"| lr={scheduler.get_last_lr()[0]:.2e} "
                f"| elapsed={format_seconds(elapsed)} eta={format_seconds(eta)}"
            )

    total_time = time.time() - t0
    print(f"Training finished in {format_seconds(total_time)}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["loss"], linewidth=1.2)
    axes[0].set_title("Training Loss (MSE)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["lr"], linewidth=1.2)
    axes[1].set_title("Learning Rate")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("LR")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    curve_path = os.path.join(TRAIN_PNG_FOLDER, "training_curve_dit.png")
    fig.savefig(curve_path, dpi=140)
    plt.close(fig)
    print(f"Training curve saved to: {curve_path}")

    print("=" * 70)
    print("Validation")
    print("=" * 70)
    val_rois, val_angles, val_extents = load_validation_rois(
        VALIDATION_FOLDER,
        ROI_HEIGHT,
        ROI_WIDTH,
        roi_mode=ROI_MODE,
        roi_corner=ROI_CORNER,
    )

    if val_rois is None or len(val_rois) == 0:
        print("No validation .mat files found in input/validation. Skipping validation.")
    else:
        raw_state = _state_clone(model.state_dict())
        _load_state(model, ema_state)
        model.eval()
        n_val = min(NUM_VALIDATION_ANGLES, len(val_rois))
        selected_idx = list(range(n_val))

        val_mae, val_rmse, val_mape, val_max = [], [], [], []
        fig_val, axes_val = plt.subplots(n_val, 3, figsize=(15, max(4, 4 * n_val)), squeeze=False)

        with torch.no_grad():
            for row, idx in enumerate(selected_idx):
                angle = float(val_angles[idx])
                real_tl = val_rois[idx]

                gen_tl, _ = generate_samples_ddim(
                    model=model,
                    codec=codec,
                    device=device,
                    normalizer=norm,
                    angle=angle,
                    num_samples=1,
                    num_steps=min(VALIDATION_DDIM_STEPS, DIFFUSION_STEPS),
                )

                metrics = compute_error_metrics(real_tl, gen_tl)
                val_mae.append(float(metrics["mae"]))
                val_rmse.append(float(metrics["rmse"]))
                val_mape.append(float(metrics["mape"]))
                val_max.append(float(metrics["max_error"]))

                err_map = np.abs(real_tl - gen_tl)
                extent = val_extents[idx] if idx < len(val_extents) else [0, ROI_WIDTH, 0, ROI_HEIGHT]

                im0 = axes_val[row, 0].imshow(
                    real_tl,
                    cmap="jet",
                    aspect="auto",
                    origin="lower",
                    vmin=norm.tl_min,
                    vmax=norm.tl_max,
                    extent=extent,
                )
                axes_val[row, 0].set_title(f"Real ({angle:.2f} deg)")
                plt.colorbar(im0, ax=axes_val[row, 0], fraction=0.046, pad=0.04)

                im1 = axes_val[row, 1].imshow(
                    gen_tl,
                    cmap="jet",
                    aspect="auto",
                    origin="lower",
                    vmin=norm.tl_min,
                    vmax=norm.tl_max,
                    extent=extent,
                )
                axes_val[row, 1].set_title(f"Generated ({angle:.2f} deg)")
                plt.colorbar(im1, ax=axes_val[row, 1], fraction=0.046, pad=0.04)

                im2 = axes_val[row, 2].imshow(
                    err_map,
                    cmap="hot",
                    aspect="auto",
                    origin="lower",
                    extent=extent,
                )
                axes_val[row, 2].set_title(
                    f"|Error| MAE={metrics['mae']:.3f} RMSE={metrics['rmse']:.3f}"
                )
                plt.colorbar(im2, ax=axes_val[row, 2], fraction=0.046, pad=0.04)

                mat_path = os.path.join(VALIDATION_MAT_FOLDER, f"val_generated_angle_{angle:+07.2f}.mat")
                save_mat(mat_path, gen_tl)

                print(
                    f"Validation angle={angle:+07.2f} | MAE={metrics['mae']:.3f} "
                    f"RMSE={metrics['rmse']:.3f} MAPE={metrics['mape']:.2f}%"
                )

        fig_val.suptitle("Validation: Real vs Generated", fontsize=13)
        fig_val.tight_layout()
        val_plot_path = os.path.join(VALIDATION_PNG_FOLDER, "validation_real_vs_generated.png")
        fig_val.savefig(val_plot_path, dpi=140)
        plt.close(fig_val)

        print(f"Validation figure saved to: {val_plot_path}")
        print(
            "Validation summary: "
            f"MAE={np.mean(val_mae):.3f} | RMSE={np.mean(val_rmse):.3f} "
            f"| MAPE={np.mean(val_mape):.2f}% | MaxAbs={np.max(val_max):.3f}"
        )
        _load_state(model, raw_state)

    model_state_cpu = _state_to_cpu(model.state_dict())
    ema_state_cpu = _state_to_cpu(ema_state)
    compression_ratio = codec_info.get(
        "compression_ratio",
        0.5 * (compression_ratio_h + compression_ratio_w),
    )

    checkpoint = {
        "model_state_dict": model_state_cpu,
        "ema_model_state_dict": ema_state_cpu,
        # DiT-main compatible aliases:
        "model": model_state_cpu,
        "ema": ema_state_cpu,
        "codec_type": codec_info.get("codec_type", "latent"),
        "vae_checkpoint_path": codec_info.get("vae_checkpoint_path"),
        "latent_channels": latent_channels,
        "latent_height": latent_h,
        "latent_width": latent_w,
        "compression_ratio": float(compression_ratio),
        "compression_ratio_h": compression_ratio_h,
        "compression_ratio_w": compression_ratio_w,
        "patch_size": MODEL_PATCH_SIZE,
        "attention_chunk_size": ATTENTION_CHUNK_SIZE,
        "gradient_checkpointing": GRADIENT_CHECKPOINTING,
        "hidden_size": MODEL_HIDDEN_SIZE,
        "depth": MODEL_DEPTH,
        "num_heads": MODEL_NUM_HEADS,
        "mlp_ratio": MLP_RATIO,
        "diffusion_steps": DIFFUSION_STEPS,
        "diffusion_schedule": DIFFUSION_SCHEDULE,
        "beta_start": BETA_START,
        "beta_end": BETA_END,
        "ema_decay": EMA_DECAY,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "validation_ddim_steps": VALIDATION_DDIM_STEPS,
        "roi_height": ROI_HEIGHT,
        "roi_width": ROI_WIDTH,
        **norm.state_dict(),
    }
    _atomic_torch_save(checkpoint, DIT_MODEL_PATH)
    print("=" * 70)
    print(f"Checkpoint saved to: {DIT_MODEL_PATH}")
    if os.path.exists(f"{DIT_MODEL_PATH}.prev"):
        print(f"Previous checkpoint backup: {DIT_MODEL_PATH}.prev")
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()