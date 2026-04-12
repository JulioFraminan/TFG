import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_intento_mat import (  # noqa: E402
    _as_abs_path,
    _build_diffusion_from_metadata,
    _build_model_from_metadata,
    _generate_single_sample,
    _load_metadata,
    _load_weights,
    _resolve_checkpoint_path,
    set_seed,
)
from intento_mat_utils import (  # noqa: E402
    DatasetBundle,
    NormalizationStats,
    denormalize_fields_01,
    load_rois_from_folder,
    normalize_single_angle,
    parse_angle_list,
    resize_fields,
    save_mat_h5,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate trained diffusion checkpoints on unseen PlaneAngle MAT/H5 data"
    )
    parser.add_argument("--results-folder", type=str, default="results/intento_mat/dit_gaussian")
    parser.add_argument("--metadata-path", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--milestone", type=int, default=-1)
    parser.add_argument("--prefer-ema", action="store_true")

    parser.add_argument("--validation-folder", type=str, default="")
    parser.add_argument("--angles", type=str, default="")

    parser.add_argument("--cond-scale", type=float, default=6.0)
    parser.add_argument("--sampler", type=str, choices=["ddpm", "ddim"], default="ddim")
    parser.add_argument("--num-inference-steps", type=int, default=-1)

    parser.add_argument("--output-subdir", type=str, default="validation")
    parser.add_argument("--rows-per-page", type=int, default=6)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def _default_extent_from_metadata(metadata: Dict) -> List[float]:
    roi_corner = metadata.get("roi_corner", [0.0, 0.0])
    x_left = float(roi_corner[0]) if len(roi_corner) >= 1 else 0.0
    z_top = float(roi_corner[1]) if len(roi_corner) >= 2 else 0.0
    roi_w = float(metadata.get("roi_width", 1.0))
    roi_h = float(metadata.get("roi_height", 1.0))
    return [x_left, x_left + roi_w, z_top - roi_h, z_top]


def _compute_error_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    diff = predicted - actual
    abs_diff = np.abs(diff)

    mae = float(np.mean(abs_diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    max_error = float(np.max(abs_diff))

    denom = np.abs(actual)
    valid = denom > 1e-8
    if np.any(valid):
        mape = float(np.mean(abs_diff[valid] / denom[valid]) * 100.0)
    else:
        mape = float("inf")

    return {
        "mae": mae,
        "rmse": rmse,
        "max_error": max_error,
        "mape": mape,
    }


def _select_indices(angles_deg: np.ndarray, requested_angles: List[float], print_fn=print) -> List[int]:
    if len(requested_angles) == 0:
        return list(range(len(angles_deg)))

    selected: List[int] = []
    used = set()

    for target in requested_angles:
        idx = int(np.argmin(np.abs(angles_deg - target)))
        if idx in used:
            continue

        used.add(idx)
        selected.append(idx)

        nearest = float(angles_deg[idx])
        if abs(nearest - target) > 1e-3:
            print_fn(
                f"[WARN] Requested angle {target:+.2f} not found exactly; using nearest {nearest:+.2f}."
            )

    selected.sort(key=lambda i: float(angles_deg[i]))
    return selected


def _subset_bundle(bundle: DatasetBundle, indices: List[int]) -> DatasetBundle:
    return DatasetBundle(
        rois=bundle.rois[indices],
        angles_deg=bundle.angles_deg[indices],
        extents=[bundle.extents[i] for i in indices],
    )


def _resolve_validation_folder(metadata: Dict, override_folder: str) -> str:
    if override_folder and os.path.isdir(override_folder):
        return override_folder

    candidate = str(metadata.get("validation_folder", ""))
    if candidate and os.path.isdir(candidate):
        return candidate

    input_folder = str(metadata.get("input_folder", ""))
    fallback = os.path.join(input_folder, "validation")
    if os.path.isdir(fallback):
        return fallback

    raise FileNotFoundError(
        "Validation folder not found. Provide --validation-folder or store a valid validation_folder in metadata."
    )


def _safe_mean(values: List[float]) -> float:
    return float(np.mean(values)) if len(values) > 0 else float("nan")


def _json_number(value: float):
    return None if not np.isfinite(value) else float(value)


def run_validation(
    results_folder: Path,
    metadata: Dict,
    checkpoint_path: Path,
    prefer_ema: bool = True,
    validation_folder_override: str = "",
    requested_angles: Optional[List[float]] = None,
    output_subdir: str = "validation",
    cond_scale: float = 6.0,
    sampler: str = "ddim",
    num_inference_steps: int = -1,
    rows_per_page: int = 6,
    max_samples: int = -1,
    seed: int = 42,
    print_fn=print,
) -> Dict:
    set_seed(seed)

    stats = NormalizationStats(
        tl_min=float(metadata["tl_min"]),
        tl_max=float(metadata["tl_max"]),
        angle_mean=float(metadata["angle_mean"]),
        angle_std=float(metadata["angle_std"]),
    )

    validation_folder = _resolve_validation_folder(metadata, validation_folder_override)

    validation_bundle = load_rois_from_folder(
        folder=validation_folder,
        roi_h=float(metadata["roi_height"]),
        roi_w=float(metadata["roi_width"]),
        rois_per_plane=1,
        roi_mode=str(metadata["roi_mode"]),
        roi_corner=(float(metadata["roi_corner"][0]), float(metadata["roi_corner"][1])),
        verbose=False,
    )

    if validation_bundle.rois.size == 0:
        print_fn(f"Validation skipped: no planes found in {validation_folder}")
        return {
            "status": "skipped",
            "reason": "empty_validation_folder",
            "validation_folder": validation_folder,
        }

    requested_angles = requested_angles or []
    selected_indices = _select_indices(validation_bundle.angles_deg, requested_angles, print_fn=print_fn)

    if max_samples > 0:
        selected_indices = selected_indices[:max_samples]

    if len(selected_indices) == 0:
        raise RuntimeError("No validation planes selected after filtering.")

    validation_bundle = _subset_bundle(validation_bundle, selected_indices)

    model = _build_model_from_metadata(metadata)
    diffusion = _build_diffusion_from_metadata(metadata, model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffusion = diffusion.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    loaded_from = _load_weights(diffusion, checkpoint, prefer_ema=prefer_ema)
    diffusion.eval()

    output_root = results_folder / output_subdir
    png_folder = output_root / "PNG"
    mat_folder = output_root / "MAT"
    png_folder.mkdir(parents=True, exist_ok=True)
    mat_folder.mkdir(parents=True, exist_ok=True)

    default_extent = _default_extent_from_metadata(metadata)
    train_h = int(metadata["train_height"])
    train_w = int(metadata["train_width"])

    records = []

    for idx in range(validation_bundle.rois.shape[0]):
        angle_deg = float(validation_bundle.angles_deg[idx])
        real_tl = validation_bundle.rois[idx].astype(np.float32)
        extent = (
            validation_bundle.extents[idx]
            if idx < len(validation_bundle.extents) and validation_bundle.extents[idx] is not None
            else default_extent
        )

        cond_value = normalize_single_angle(angle_deg, stats)
        cond = torch.tensor([[cond_value]], dtype=torch.float32, device=device)

        sample_small_01 = _generate_single_sample(
            diffusion=diffusion,
            cond_tensor=cond,
            algorithm=str(metadata.get("algorithm", "gaussian")),
            cond_scale=cond_scale,
            sampler=sampler,
            num_inference_steps=num_inference_steps,
        )

        if train_h != real_tl.shape[0] or train_w != real_tl.shape[1]:
            sample_01 = resize_fields(
                sample_small_01[None, ...],
                target_h=real_tl.shape[0],
                target_w=real_tl.shape[1],
            )[0]
        else:
            sample_01 = sample_small_01

        sample_tl = denormalize_fields_01(sample_01[None, ...], stats)[0]
        error_map = np.abs(sample_tl - real_tl)
        metrics = _compute_error_metrics(real_tl, sample_tl)

        mat_name = f"val_generado_{angle_deg:+07.2f}.mat"
        save_mat_h5(str(mat_folder / mat_name), sample_tl, extent=extent)

        records.append(
            {
                "angle_deg": angle_deg,
                "real_tl": real_tl,
                "generated_tl": sample_tl,
                "error_map": error_map,
                "extent": extent,
                "metrics": metrics,
                "mat_name": mat_name,
            }
        )

    rows_per_page = max(1, int(rows_per_page))
    n_samples = len(records)
    n_pages = int(np.ceil(n_samples / rows_per_page))

    for page in range(n_pages):
        start = page * rows_per_page
        end = min(start + rows_per_page, n_samples)
        page_records = records[start:end]

        fig, axes = plt.subplots(
            len(page_records),
            3,
            figsize=(15, max(4, 4.2 * len(page_records))),
            squeeze=False,
        )

        for row, rec in enumerate(page_records):
            real_tl = rec["real_tl"]
            generated_tl = rec["generated_tl"]
            error_map = rec["error_map"]
            extent = rec["extent"]
            angle_deg = rec["angle_deg"]
            metrics = rec["metrics"]

            im0 = axes[row, 0].imshow(
                real_tl,
                cmap="jet",
                origin="lower",
                aspect="auto",
                vmin=stats.tl_min,
                vmax=stats.tl_max,
                extent=extent,
            )
            axes[row, 0].set_title(f"Validation reference ({angle_deg:+.2f} deg)")
            axes[row, 0].set_xlabel("X [m]")
            axes[row, 0].set_ylabel("Z [m]")
            plt.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)

            im1 = axes[row, 1].imshow(
                generated_tl,
                cmap="jet",
                origin="lower",
                aspect="auto",
                vmin=stats.tl_min,
                vmax=stats.tl_max,
                extent=extent,
            )
            axes[row, 1].set_title(f"Generated ({angle_deg:+.2f} deg)")
            axes[row, 1].set_xlabel("X [m]")
            axes[row, 1].set_ylabel("Z [m]")
            plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)

            mape = metrics["mape"]
            mape_text = f"{mape:.2f}%" if np.isfinite(mape) else "inf"
            im2 = axes[row, 2].imshow(
                error_map,
                cmap="turbo",
                origin="lower",
                aspect="auto",
                extent=extent,
                vmin=0.0,
                vmax=max(metrics["max_error"], 1e-6),
            )
            axes[row, 2].set_title(
                "|Error| "
                f"MAE={metrics['mae']:.3f} "
                f"RMSE={metrics['rmse']:.3f} "
                f"MAPE={mape_text}"
            )
            axes[row, 2].set_xlabel("X [m]")
            axes[row, 2].set_ylabel("Z [m]")
            plt.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)

        fig.suptitle(f"Validation report (page {page + 1}/{n_pages})", fontsize=13)
        fig.tight_layout()
        fig.savefig(png_folder / f"validation_real_vs_gen_{page + 1:02d}.png", dpi=140)
        plt.close(fig)

    all_angles = [float(rec["angle_deg"]) for rec in records]
    all_mae = [float(rec["metrics"]["mae"]) for rec in records]
    all_rmse = [float(rec["metrics"]["rmse"]) for rec in records]

    fig_err, ax = plt.subplots(1, 1, figsize=(9, 4.5))
    ax.plot(all_angles, all_mae, marker="o", label="MAE [dB]")
    ax.plot(all_angles, all_rmse, marker="s", label="RMSE [dB]")
    ax.set_xlabel("Angle [deg]")
    ax.set_ylabel("Error [dB]")
    ax.set_title("Validation error vs angle")
    ax.grid(alpha=0.3)
    ax.legend()
    fig_err.tight_layout()
    fig_err.savefig(png_folder / "error_vs_angle.png", dpi=140)
    plt.close(fig_err)

    csv_path = output_root / "validation_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["angle_deg", "mae", "rmse", "max_error", "mape", "mat_file"])
        for rec in records:
            metrics = rec["metrics"]
            writer.writerow(
                [
                    f"{rec['angle_deg']:.6f}",
                    f"{metrics['mae']:.6f}",
                    f"{metrics['rmse']:.6f}",
                    f"{metrics['max_error']:.6f}",
                    "inf" if not np.isfinite(metrics["mape"]) else f"{metrics['mape']:.6f}",
                    rec["mat_name"],
                ]
            )

    finite_mapes = [float(rec["metrics"]["mape"]) for rec in records if np.isfinite(rec["metrics"]["mape"])]

    summary = {
        "status": "ok",
        "validation_folder": validation_folder,
        "checkpoint": str(checkpoint_path),
        "weights_source": loaded_from,
        "num_samples": len(records),
        "mean_mae": _json_number(_safe_mean(all_mae)),
        "mean_rmse": _json_number(_safe_mean(all_rmse)),
        "mean_mape": _json_number(_safe_mean(finite_mapes)),
        "output_png": str(png_folder),
        "output_mat": str(mat_folder),
        "metrics_csv": str(csv_path),
        "axis_units": "m",
        "sampler": sampler,
        "cond_scale": float(cond_scale),
        "num_inference_steps": int(num_inference_steps),
        "angles": all_angles,
    }

    with (output_root / "validation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print_fn("=" * 72)
    print_fn(f"Validation samples: {len(records)}")
    print_fn(f"Mean MAE: {summary['mean_mae']}")
    print_fn(f"Mean RMSE: {summary['mean_rmse']}")
    print_fn(f"PNG outputs: {png_folder}")
    print_fn(f"MAT outputs: {mat_folder}")
    print_fn("=" * 72)

    return summary


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    results_folder = _as_abs_path(PROJECT_DIR, args.results_folder)
    if not results_folder.exists():
        raise FileNotFoundError(f"Results folder not found: {results_folder}")

    metadata = _load_metadata(args, results_folder)
    checkpoint_path = _resolve_checkpoint_path(args, results_folder)
    requested_angles = parse_angle_list(args.angles) if args.angles.strip() else []

    run_validation(
        results_folder=results_folder,
        metadata=metadata,
        checkpoint_path=checkpoint_path,
        prefer_ema=args.prefer_ema,
        validation_folder_override=args.validation_folder,
        requested_angles=requested_angles,
        output_subdir=args.output_subdir,
        cond_scale=args.cond_scale,
        sampler=args.sampler,
        num_inference_steps=args.num_inference_steps,
        rows_per_page=args.rows_per_page,
        max_samples=args.max_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
