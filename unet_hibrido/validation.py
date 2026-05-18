import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import (
    DATA_FOLDER,
    MODEL_PATH,
    ROI_CORNER,
    ROI_HEIGHT,
    ROI_MODE,
    ROI_WIDTH,
    ROIS_PER_PLANE,
    SEED,
    VALIDATION_FOLDER,
    VALIDATION_MAT_FOLDER,
    VALIDATION_PNG_FOLDER,
    INPAINT_MODE,
    INPAINT_PRESERVE_KNOWN,
    INPAINT_MIN_HOLES,
    INPAINT_MAX_HOLES,
    INPAINT_MIN_HOLE_RATIO,
    INPAINT_MAX_HOLE_RATIO,
    INPAINT_KEEP_FULL_PROB,
    INPAINT_FILL_VALUE,
    INPAINT_LOSS_KNOWN_WEIGHT,
    NOISE_STD,
    create_output_dirs,
)
from data_utils import (
    Normalizer,
    adaptive_figsize,
    compute_error_metrics,
    generate_from_seed,
    generate_inpainting_known_masks,
    apply_inpainting_mask,
    load_all_rois,
    load_validation_rois,
    save_mat,
)
from model import ConditionalUNetAE


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if len(values) > 0 else float("nan")


def _default_validation_folder(folder: str) -> str:
    candidate = os.path.join(folder, "validation")
    return candidate if os.path.isdir(candidate) else folder


def _load_checkpoint(model_path: str, device: torch.device) -> Dict:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if "unet_state_dict" not in checkpoint and "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain a usable UNet state dict.")
    return checkpoint


def _build_model_from_checkpoint(checkpoint: Dict) -> ConditionalUNetAE:
    model = ConditionalUNetAE().to("cpu")
    state_dict = checkpoint.get("unet_state_dict", checkpoint.get("model_state_dict"))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _format_angle(angle: float) -> str:
    return f"{angle:+.2f}°"


def run_validation(
    model_path: str = MODEL_PATH,
    data_folder: str = DATA_FOLDER,
    validation_folder: str = VALIDATION_FOLDER,
    roi_height: int = ROI_HEIGHT,
    roi_width: int = ROI_WIDTH,
    rois_per_plane: int = ROIS_PER_PLANE,
    roi_mode: str = ROI_MODE,
    roi_corner: Tuple[float, float] = ROI_CORNER,
    noise_std: float = NOISE_STD,
    output_png_folder: str = VALIDATION_PNG_FOLDER,
    output_mat_folder: str = VALIDATION_MAT_FOLDER,
    seed: int = SEED,
    inpaint_mode: str = INPAINT_MODE,
    inpaint_preserve_known: bool = INPAINT_PRESERVE_KNOWN,
    inpaint_min_holes: int = INPAINT_MIN_HOLES,
    inpaint_max_holes: int = INPAINT_MAX_HOLES,
    inpaint_min_hole_ratio: float = INPAINT_MIN_HOLE_RATIO,
    inpaint_max_hole_ratio: float = INPAINT_MAX_HOLE_RATIO,
    inpaint_keep_full_prob: float = INPAINT_KEEP_FULL_PROB,
    inpaint_fill_value: float = INPAINT_FILL_VALUE,
    inpaint_loss_known_weight: float = INPAINT_LOSS_KNOWN_WEIGHT,
    print_fn=print,
) -> Dict:
    _set_seed(seed)
    create_output_dirs(output_png_folder, output_mat_folder)

    validation_folder = validation_folder if os.path.isdir(validation_folder) else _default_validation_folder(data_folder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print_fn(f"Dispositivo: {device} | GPU: {torch.cuda.get_device_name(0)}")
    else:
        print_fn(f"Dispositivo: {device}")

    checkpoint = _load_checkpoint(model_path, device)
    model = _build_model_from_checkpoint(checkpoint).to(device)
    norm = Normalizer.from_checkpoint(checkpoint)

    train_rois, train_angles, train_extents = load_all_rois(
        data_folder,
        roi_height,
        roi_width,
        rois_per_plane=rois_per_plane,
        roi_mode=roi_mode,
        roi_corner=roi_corner,
    )
    if train_rois is None or getattr(train_rois, "size", 0) == 0:
        raise RuntimeError(f"No training ROIs found in {data_folder}")

    val_rois, val_angles, val_extents = load_validation_rois(
        validation_folder,
        roi_height,
        roi_width,
        roi_mode=roi_mode,
        roi_corner=roi_corner,
    )
    if val_rois is None or len(val_rois) == 0:
        print_fn(f"[i] No validation planes found in {validation_folder}")
        return {
            "status": "skipped",
            "reason": "empty_validation_folder",
            "validation_folder": validation_folder,
        }

    train_angles_sorted = np.sort(np.array(train_angles, dtype=np.float32))
    train_samples = torch.from_numpy(norm.normalize_tl(train_rois)[:, np.newaxis, :, :]).to(device)
    train_angles_arr = np.array(train_angles, dtype=np.float32)

    use_inpaint = inpaint_mode in ("inference", "both")

    print_fn("\n" + "=" * 60)
    print_fn("  VALIDACION DESDE CHECKPOINT")
    print_fn("=" * 60)
    print_fn(f"  Checkpoint: {model_path}")
    print_fn(f"  Validation folder: {validation_folder}")
    print_fn(f"  Modo inpainting: {inpaint_mode}")
    print_fn(f"  Validando {len(val_rois)} planos no vistos")

    all_maes: List[float] = []
    all_rmses: List[float] = []
    all_mapes: List[float] = []
    all_maxes: List[float] = []
    all_gaps: List[Optional[float]] = []
    all_val_angles: List[float] = []

    n_val = len(val_rois)
    n_rows_per_page = min(n_val, 4)
    n_pages = int(np.ceil(n_val / n_rows_per_page))
    roi_angles_arr = train_angles_arr

    with torch.no_grad():
        for page in range(n_pages):
            start = page * n_rows_per_page
            end = min(start + n_rows_per_page, n_val)
            rows_this = end - start

            fig, axes = plt.subplots(
                rows_this, 3,
                figsize=adaptive_figsize(roi_height, roi_width, rows_this, 3),
            )
            if rows_this == 1:
                axes = axes[np.newaxis, :]

            for r, j in enumerate(range(start, end)):
                val_angle = float(val_angles[j])
                ext = val_extents[j]

                lower_mask = train_angles_sorted[train_angles_sorted <= val_angle]
                upper_mask = train_angles_sorted[train_angles_sorted >= val_angle]
                ang_lower = float(lower_mask[-1]) if len(lower_mask) > 0 else None
                ang_upper = float(upper_mask[0]) if len(upper_mask) > 0 else None
                diff_lower = abs(val_angle - ang_lower) if ang_lower is not None else None
                diff_upper = abs(ang_upper - val_angle) if ang_upper is not None else None
                gap = abs(ang_upper - ang_lower) if ang_lower is not None and ang_upper is not None else None

                idx_seed = int(np.argmin(np.abs(roi_angles_arr - val_angle)))
                seed = train_samples[idx_seed:idx_seed + 1]

                gen_tl = generate_from_seed(model, seed, val_angle, norm, noise_std, device)

                if use_inpaint:
                    gen_norm = norm.normalize_tl(gen_tl).astype(np.float32)
                    known_mask_np = generate_inpainting_known_masks(
                        batch_size=1,
                        height=gen_norm.shape[0],
                        width=gen_norm.shape[1],
                        min_holes=inpaint_min_holes,
                        max_holes=inpaint_max_holes,
                        min_hole_ratio=inpaint_min_hole_ratio,
                        max_hole_ratio=inpaint_max_hole_ratio,
                        keep_full_prob=inpaint_keep_full_prob,
                    )
                    masked_norm = apply_inpainting_mask(
                        gen_norm[np.newaxis, np.newaxis, :, :],
                        known_mask_np,
                        fill_value=inpaint_fill_value,
                    )
                    masked_tensor = torch.from_numpy(masked_norm).to(device)
                    known_mask_t = torch.from_numpy(known_mask_np).to(device)
                    cond_val = torch.tensor([[norm.normalize_angle(val_angle)]], dtype=torch.float32, device=device)
                    inpaint_norm = model(
                        masked_tensor,
                        cond_val,
                        known_mask=known_mask_t if inpaint_preserve_known else None,
                    ).cpu().squeeze().numpy()
                    gen_tl = norm.denormalize_tl(np.clip(inpaint_norm, 0, 1))

                # `val_rois` are loaded as raw TL (dB). Do NOT denormalize again.
                real_tl = val_rois[j]
                diff_map = np.abs(gen_tl - real_tl)
                errors = compute_error_metrics(real_tl, gen_tl)
                mae_val = float(errors["mae"])
                rmse_val = float(errors["rmse"])
                mape_val = float(errors["mape"])
                max_val = float(errors["max_error"])

                all_maes.append(mae_val)
                all_rmses.append(rmse_val)
                all_mapes.append(mape_val)
                all_maxes.append(max_val)
                all_gaps.append(gap)
                all_val_angles.append(val_angle)

                mat_path = os.path.join(output_mat_folder, f"val_generado_{val_angle:+.2f}.mat")
                save_mat(mat_path, gen_tl, extent=ext)

                lo_str = f"{ang_lower:+.2f}° (dif={diff_lower:.2f}°)" if ang_lower is not None else "---"
                hi_str = f"{ang_upper:+.2f}° (dif={diff_upper:.2f}°)" if ang_upper is not None else "---"
                gap_str = f"{gap:.2f}°" if gap is not None else "---"
                mape_str = f"{mape_val:.2f}%" if mape_val != np.inf else "undef"
                print_fn(
                    f"  Angulo val {val_angle:+.2f}°  |  vecino inf: {lo_str}  |  vecino sup: {hi_str}  |  "
                    f"gap: {gap_str}  |  MAE: {mae_val:.2f} dB  |  RMSE: {rmse_val:.2f} dB  |  "
                    f"MAPE: {mape_str}  |  Max: {max_val:.2f} dB"
                )

                neigh_parts = []
                if ang_lower is not None:
                    neigh_parts.append(f"Vecino inf: {ang_lower:+.1f}° (Δ={diff_lower:.1f}°)")
                if ang_upper is not None:
                    neigh_parts.append(f"Vecino sup: {ang_upper:+.1f}° (Δ={diff_upper:.1f}°)")
                if gap is not None:
                    neigh_parts.append(f"Gap total: {gap:.1f}°")
                neigh_txt = "  |  ".join(neigh_parts)
                angle_str = f"{val_angle:.2f}°"

                im0 = axes[r, 0].imshow(real_tl, cmap="jet", aspect="auto", origin="lower",
                                       vmin=norm.tl_min, vmax=norm.tl_max, extent=ext)
                fig.colorbar(im0, ax=axes[r, 0], shrink=0.6, label="TL (dB)")
                axes[r, 0].set_title(f"Real -- {angle_str}\n{neigh_txt}", fontsize=8)
                axes[r, 0].set_xlabel("X (m)")
                axes[r, 0].set_ylabel("Z (m)")

                im1 = axes[r, 1].imshow(gen_tl, cmap="jet", aspect="auto", origin="lower",
                                       vmin=norm.tl_min, vmax=norm.tl_max, extent=ext)
                fig.colorbar(im1, ax=axes[r, 1], shrink=0.6, label="TL (dB)")
                axes[r, 1].set_title(f"Generado -- {angle_str}\n(semilla {train_angles_arr[idx_seed]:.2f}°)", fontsize=8)
                axes[r, 1].set_xlabel("X (m)")
                axes[r, 1].set_ylabel("Z (m)")

                im2 = axes[r, 2].imshow(diff_map, cmap="turbo", aspect="auto", origin="lower",
                                       vmin=0, vmax=max(max_val, 1e-6), extent=ext)
                fig.colorbar(im2, ax=axes[r, 2], shrink=0.6, label="|Error| (dB)")
                axes[r, 2].set_title(
                    f"|Error| -- {angle_str}\nMAE={mae_val:.2f} dB  MAPE={mape_str}  RMSE={rmse_val:.2f} dB  Max={max_val:.2f} dB",
                    fontsize=7,
                )
                axes[r, 2].set_xlabel("X (m)")
                axes[r, 2].set_ylabel("Z (m)")

            fig.suptitle(f"Validacion: Real vs Generado -- UNet AE  (pag. {page+1}/{n_pages})", fontsize=14)
            fig.tight_layout()
            fig.savefig(os.path.join(output_png_folder, f"validacion_real_vs_gen_{page+1:02d}.png"), dpi=150)
            plt.close(fig)
            print_fn(f"  Validacion pag. {page+1}/{n_pages} guardada ({rows_this} planos).")

    plot_data = [
        (g, m, mx, r, mp, a)
        for g, m, mx, r, mp, a in zip(all_gaps, all_maes, all_maxes, all_rmses, all_mapes, all_val_angles)
        if g is not None
    ]
    if plot_data:
        plot_data.sort(key=lambda d: d[0])
        gaps_arr = np.array([d[0] for d in plot_data])
        maes_arr = np.array([d[1] for d in plot_data])
        maxes_arr = np.array([d[2] for d in plot_data])
        rmses_arr = np.array([d[3] for d in plot_data])
        mapes_arr = np.array([d[4] for d in plot_data])
        labels_arr = [f"{d[5]:+.2f}°" for d in plot_data]

        fig_err, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        ax1.plot(gaps_arr, maes_arr, color="#1f77b4", linewidth=1.2, alpha=0.6, zorder=2)
        ax1.plot(gaps_arr, rmses_arr, color="#ff7f0e", linewidth=1.2, alpha=0.6, zorder=2)
        ax1.plot(gaps_arr, maxes_arr, color="#d62728", linewidth=1.2, alpha=0.6, linestyle="--", zorder=2)
        ax1.scatter(gaps_arr, maes_arr, color="#1f77b4", s=50, zorder=3, label="MAE (dB)")
        ax1.scatter(gaps_arr, rmses_arr, color="#ff7f0e", s=50, zorder=3, label="RMSE (dB)")
        ax1.scatter(gaps_arr, maxes_arr, color="#d62728", s=50, marker="^", zorder=3, label="Max error (dB)")
        ax1.set_xlabel("Gap entre vecinos de entrenamiento (°)", fontsize=10)
        ax1.set_ylabel("Error (dB)", fontsize=10)
        ax1.set_title("Errores Absolutos vs Distancia Angular", fontsize=11)
        ax1.legend(fontsize=9, loc="upper left")
        ax1.grid(True, alpha=0.3)

        ax2.plot(gaps_arr, mapes_arr, color="#2ca02c", linewidth=1.2, alpha=0.6, zorder=2)
        ax2.scatter(gaps_arr, mapes_arr, color="#2ca02c", s=50, zorder=3, label="MAPE (%)")
        for xi, yi, lbl in zip(gaps_arr, mapes_arr, labels_arr):
            ax2.annotate(lbl, (xi, yi), textcoords="offset points", xytext=(5, 5), fontsize=7, color="#2ca02c")
        ax2.set_xlabel("Gap entre vecinos de entrenamiento (°)", fontsize=10)
        ax2.set_ylabel("MAPE (%)", fontsize=10)
        ax2.set_title("Error Relativo (MAPE) vs Distancia Angular", fontsize=11)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        fig_err.suptitle(
            "Análisis de Validación: Error vs Distancia Angular\nentre los dos vecinos de entrenamiento más cercanos",
            fontsize=12,
        )
        fig_err.tight_layout()
        err_path = os.path.join(output_png_folder, "error_vs_gap.png")
        fig_err.savefig(err_path, dpi=150)
        plt.close(fig_err)
        print_fn(f"  Grafico error vs gap guardado en: {err_path}")

    mean_mae = _safe_mean(all_maes)
    mean_rmse = _safe_mean(all_rmses)
    finite_mapes = [m for m in all_mapes if m != np.inf]
    mean_mape = _safe_mean(finite_mapes)
    mean_max = float(np.mean(all_maxes)) if len(all_maxes) > 0 else float("inf")

    print_fn("\n  Validacion completada:")
    print_fn(f"    MAE medio   = {mean_mae:.2f} dB")
    print_fn(f"    RMSE medio  = {mean_rmse:.2f} dB")
    print_fn(f"    MAPE medio  = {mean_mape:.1f}%")
    print_fn(f"    Max medio   = {mean_max:.2f} dB")
    print_fn(f"    sobre {len(all_maes)} planos.")
    print_fn(f"  Resultados en: {output_png_folder}")
    print_fn(f"                 {output_mat_folder}")

    summary = {
        "status": "ok",
        "validation_folder": validation_folder,
        "mean_mae": float(mean_mae),
        "mean_rmse": float(mean_rmse),
        "mean_mape": float(mean_mape),
        "mean_max_error": float(mean_max),
        "num_samples": int(len(all_maes)),
        "model_path": model_path,
    }

    metrics_csv = os.path.join(output_png_folder, "validation_metrics.csv")
    with open(metrics_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["angle_deg", "gap_deg", "mae", "rmse", "mape", "max_error"])
        writer.writeheader()
        for angle_deg, gap, mae, rmse, mape, max_error in zip(all_val_angles, all_gaps, all_maes, all_rmses, all_mapes, all_maxes):
            writer.writerow({
                "angle_deg": angle_deg,
                "gap_deg": gap,
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
                "max_error": max_error,
            })

    summary_path = os.path.join(output_png_folder, "validation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a saved UNet Hibrido checkpoint")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--data-folder", type=str, default=DATA_FOLDER)
    parser.add_argument("--validation-folder", type=str, default=VALIDATION_FOLDER)
    parser.add_argument("--roi-height", type=int, default=ROI_HEIGHT)
    parser.add_argument("--roi-width", type=int, default=ROI_WIDTH)
    parser.add_argument("--rois-per-plane", type=int, default=ROIS_PER_PLANE)
    parser.add_argument("--roi-mode", type=str, default=ROI_MODE, choices=["corner_fixed", "center_max"])
    parser.add_argument("--roi-corner-x", type=float, default=float(ROI_CORNER[0]))
    parser.add_argument("--roi-corner-z", type=float, default=float(ROI_CORNER[1]))
    parser.add_argument("--noise-std", type=float, default=NOISE_STD)
    parser.add_argument("--output-png-folder", type=str, default=VALIDATION_PNG_FOLDER)
    parser.add_argument("--output-mat-folder", type=str, default=VALIDATION_MAT_FOLDER)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--inpaint-mode", type=str, default=INPAINT_MODE, choices=["none", "train", "inference", "both"])
    parser.add_argument("--inpaint-preserve-known", action="store_true", default=INPAINT_PRESERVE_KNOWN)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_validation(
        model_path=args.model_path,
        data_folder=args.data_folder,
        validation_folder=args.validation_folder,
        roi_height=args.roi_height,
        roi_width=args.roi_width,
        rois_per_plane=args.rois_per_plane,
        roi_mode=args.roi_mode,
        roi_corner=(args.roi_corner_x, args.roi_corner_z),
        noise_std=args.noise_std,
        output_png_folder=args.output_png_folder,
        output_mat_folder=args.output_mat_folder,
        seed=args.seed,
        inpaint_mode=args.inpaint_mode,
        inpaint_preserve_known=args.inpaint_preserve_known,
    )


if __name__ == "__main__":
    main()