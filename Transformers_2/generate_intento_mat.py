import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from denoising_diffusion_pytorch.continuous_classifier_free_guidance import GaussianDiffusion, Unet  # noqa: E402
from denoising_diffusion_pytorch.dit import DiT_models  # noqa: E402
from config import get_generate_arg_defaults  # noqa: E402
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
        description="Generate conditioned MAT/PNG samples from train_intento_mat.py checkpoints"
    )
    defaults = get_generate_arg_defaults()
    parser.add_argument("--results-folder", type=str, default="results/intento_mat/dit_gaussian")
    parser.add_argument(
        "--auto-select-config-run",
        dest="auto_select_config_run",
        action="store_true",
        help="If results-folder is a parent folder with multiple cfg_* runs, select the latest one automatically.",
    )
    parser.add_argument(
        "--no-auto-select-config-run",
        dest="auto_select_config_run",
        action="store_false",
        help="Disable automatic selection of cfg_* subfolders.",
    )
    parser.add_argument("--metadata-path", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--milestone", type=int, default=-1)
    parser.add_argument("--prefer-ema", action="store_true")

    parser.add_argument("--angles", type=str, default="95,100,110,120,130,140,150,160,170,180")
    parser.add_argument("--cond-scale", type=float, default=6.0)
    parser.add_argument("--sampler", type=str, choices=["ddpm", "ddim"], default="ddim")
    parser.add_argument("--num-inference-steps", type=int, default=-1)

    parser.add_argument("--output-subdir", type=str, default="generate")
    parser.add_argument("--with-reference", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.set_defaults(**defaults)

    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _as_abs_path(base_dir: Path, maybe_relative: str) -> Path:
    path = Path(maybe_relative)
    return path if path.is_absolute() else (base_dir / path).resolve()


def _load_metadata(args: argparse.Namespace, results_folder: Path) -> Dict:
    metadata_path = (
        _as_abs_path(results_folder, args.metadata_path)
        if args.metadata_path
        else (results_folder / "intento_training_metadata.json")
    )

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


CHECKPOINT_SUBDIR = "checkpoints"


def _checkpoint_search_dirs(results_folder: Path) -> Tuple[Path, Path]:
    return (
        results_folder / CHECKPOINT_SUBDIR,
        results_folder,
    )


def _iter_checkpoint_candidates(results_folder: Path) -> Iterable[Tuple[int, Path]]:
    pattern = re.compile(r"model-(\d+)\.pt$")
    for directory in _checkpoint_search_dirs(results_folder):
        if not directory.exists():
            continue
        for path in directory.glob("model-*.pt"):
            match = pattern.search(path.name)
            if match:
                yield int(match.group(1)), path


def _discover_latest_checkpoint(results_folder: Path) -> Path:
    candidates = list(_iter_checkpoint_candidates(results_folder))
    if not candidates:
        searched = ", ".join(str(path) for path in _checkpoint_search_dirs(results_folder))
        raise FileNotFoundError(f"No model-*.pt checkpoints found in: {searched}")
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def _has_run_metadata(folder: Path) -> bool:
    return (folder / "intento_training_metadata.json").exists()


def _latest_artifact_mtime(folder: Path) -> float:
    latest = 0.0
    for _, checkpoint_path in _iter_checkpoint_candidates(folder):
        latest = max(latest, checkpoint_path.stat().st_mtime)

    metadata_path = folder / "intento_training_metadata.json"
    if metadata_path.exists():
        latest = max(latest, metadata_path.stat().st_mtime)

    return latest


def _auto_select_results_run(results_folder: Path) -> Optional[Path]:
    if _has_run_metadata(results_folder):
        return results_folder

    candidates = []
    for child in results_folder.iterdir():
        if not child.is_dir():
            continue
        if not _has_run_metadata(child):
            continue
        candidates.append((_latest_artifact_mtime(child), child))

    if len(candidates) == 0:
        return None

    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def _resolve_checkpoint_path(args: argparse.Namespace, results_folder: Path) -> Path:
    if args.checkpoint:
        checkpoint_path = _as_abs_path(results_folder, args.checkpoint)
        if not checkpoint_path.exists():
            # Backward compatible shorthand: --checkpoint model-21.pt
            fallback_checkpoint = results_folder / CHECKPOINT_SUBDIR / args.checkpoint
            if fallback_checkpoint.exists():
                checkpoint_path = fallback_checkpoint
            else:
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    if args.milestone >= 0:
        candidate_paths = [
            results_folder / CHECKPOINT_SUBDIR / f"model-{args.milestone}.pt",
            results_folder / f"model-{args.milestone}.pt",
        ]
        for checkpoint_path in candidate_paths:
            if checkpoint_path.exists():
                return checkpoint_path
        raise FileNotFoundError(
            "Checkpoint milestone not found. Checked: "
            + ", ".join(str(path) for path in candidate_paths)
        )

    return _discover_latest_checkpoint(results_folder)


def _build_model_from_metadata(metadata: Dict):
    model_type = metadata.get("model_type", "dit")
    train_h = int(metadata["train_height"])
    train_w = int(metadata["train_width"])

    if model_type == "unet":
        dim_mults = tuple(int(part.strip()) for part in str(metadata.get("unet_dim_mults", "1,2,4")).split(",") if part.strip())
        return Unet(
            dim=int(metadata.get("unet_dim", 64)),
            cond_dim=1,
            cond_drop_prob=float(metadata.get("unet_cond_drop_prob", 0.2)),
            channels=1,
            dim_mults=dim_mults,
        )

    variant = metadata.get("dit_variant", "DiT-XXS/2")
    if variant not in DiT_models:
        available = ", ".join(sorted(DiT_models.keys()))
        raise KeyError(f"Unknown DiT variant '{variant}'. Available variants: {available}")

    return DiT_models[variant](
        input_size=(train_h, train_w),
        cond_dim=1,
        class_dropout_prob=float(metadata.get("dit_class_dropout", 0.2)),
        in_channels=1,
        learn_sigma=False,
        attn_type=metadata.get("dit_attn_type", "vanilla"),
        mlp_ratio=float(metadata.get("dit_mlp_ratio", 2.5)),
        qk_norm=bool(metadata.get("dit_qk_norm", False)),
    )


def _build_diffusion_from_metadata(metadata: Dict, model: torch.nn.Module):
    train_h = int(metadata["train_height"])
    train_w = int(metadata["train_width"])

    if metadata.get("algorithm", "gaussian") == "gaussian":
        return GaussianDiffusion(
            model,
            image_size=(train_h, train_w),
            objective=metadata.get("objective", "pred_noise"),
            beta_schedule=metadata.get("beta_schedule", "cosine"),
            sampling_timesteps=int(metadata.get("sampling_timesteps", 300)),
            timesteps=int(metadata.get("timesteps", 1000)),
            min_snr_loss_weight=bool(metadata.get("min_snr_loss_weight", False)),
            min_snr_gamma=float(metadata.get("min_snr_gamma", 5.0)),
        )

    from denoising_diffusion_pytorch.transport import FlowMatching, Sampler, create_transport

    flow_loss_weight = metadata.get("flow_loss_weight", "none")
    if flow_loss_weight == "none":
        flow_loss_weight = None

    transport = create_transport(
        path_type=metadata.get("flow_path_type", "Linear"),
        prediction=metadata.get("flow_prediction", "velocity"),
        loss_weight=flow_loss_weight,
    )
    sampler = Sampler(transport=transport)

    return FlowMatching(
        sampler=sampler,
        neural_net=model,
        input_size=(train_h, train_w),
        cond_scale=2.0,
        num_sampling_steps=int(metadata.get("flow_num_steps", 250)),
        sampling_method=metadata.get("flow_sampling_method", "euler"),
        sampler_atol=float(metadata.get("flow_atol", 1e-6)),
        sampler_rtol=float(metadata.get("flow_rtol", 1e-3)),
    )


def _state_dict_candidate(state: object) -> bool:
    return isinstance(state, dict) and len(state) > 0 and all(torch.is_tensor(v) for v in state.values())


def _prefixed_state(state: Dict, prefix: str) -> Optional[Dict]:
    stripped = {k[len(prefix):]: v for k, v in state.items() if isinstance(k, str) and k.startswith(prefix)}
    return stripped if stripped else None


def _candidate_states(checkpoint: Dict, prefer_ema: bool) -> Iterable[Tuple[str, Dict]]:
    if prefer_ema:
        ema_state = checkpoint.get("ema")
        if isinstance(ema_state, dict):
            for key in ("ema_model", "model", "online_model", "state_dict"):
                value = ema_state.get(key)
                if _state_dict_candidate(value):
                    yield f"ema.{key}", value

            for prefix in ("ema_model.", "model."):
                prefixed = _prefixed_state(ema_state, prefix)
                if _state_dict_candidate(prefixed):
                    yield f"ema[{prefix}*]", prefixed

            if _state_dict_candidate(ema_state):
                yield "ema", ema_state

    for key in ("model", "model_state_dict", "state_dict"):
        value = checkpoint.get(key)
        if _state_dict_candidate(value):
            yield key, value


def _load_weights(diffusion: torch.nn.Module, checkpoint: Dict, prefer_ema: bool) -> str:
    errors = []
    for name, state_dict in _candidate_states(checkpoint, prefer_ema=prefer_ema):
        try:
            missing, unexpected = diffusion.load_state_dict(state_dict, strict=False)
            print(f"Loaded weights from checkpoint key '{name}'")
            if missing:
                print(f"[WARN] Missing keys: {len(missing)}")
            if unexpected:
                print(f"[WARN] Unexpected keys: {len(unexpected)}")
            return name
        except Exception as error:
            errors.append(f"{name}: {error}")

    error_text = "\n".join(errors) if errors else "No compatible state dict found"
    raise RuntimeError(f"Could not load checkpoint weights. Tried candidates:\n{error_text}")


def _nearest_reference(bundle: DatasetBundle, angle: float) -> Optional[Tuple[np.ndarray, float, List[float]]]:
    if bundle.rois.size == 0:
        return None
    idx = int(np.argmin(np.abs(bundle.angles_deg - angle)))
    return bundle.rois[idx], float(bundle.angles_deg[idx]), bundle.extents[idx]


def _default_extent_from_metadata(metadata: Dict) -> List[float]:
    roi_corner = metadata.get("roi_corner", [0.0, 0.0])
    x_left = float(roi_corner[0]) if len(roi_corner) >= 1 else 0.0
    z_top = float(roi_corner[1]) if len(roi_corner) >= 2 else 0.0
    roi_w = float(metadata.get("roi_width", 1.0))
    roi_h = float(metadata.get("roi_height", 1.0))
    return [x_left, x_left + roi_w, z_top - roi_h, z_top]


def _generate_single_sample(
    diffusion,
    cond_tensor: torch.Tensor,
    algorithm: str,
    cond_scale: float,
    sampler: str,
    num_inference_steps: int,
) -> np.ndarray:
    with torch.inference_mode():
        if algorithm == "gaussian":
            kwargs = {"cond_scale": cond_scale, "sampler": sampler}
            if num_inference_steps > 0:
                kwargs["num_inference_steps"] = num_inference_steps
            sample = diffusion.sample(cond_tensor, **kwargs)
        else:
            if num_inference_steps > 0 and hasattr(diffusion, "num_sampling_steps"):
                diffusion.num_sampling_steps = int(num_inference_steps)
            sample = diffusion.sample(cond_tensor, cond_scale=cond_scale)

    sample_np = sample.detach().cpu().numpy()[0, 0]
    return np.clip(sample_np, 0.0, 1.0)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    results_folder = _as_abs_path(PROJECT_DIR, args.results_folder)
    if not results_folder.exists():
        raise FileNotFoundError(f"Results folder not found: {results_folder}")

    if args.auto_select_config_run:
        selected_results_folder = _auto_select_results_run(results_folder)
        if selected_results_folder is not None and selected_results_folder != results_folder:
            print(f"[results] Auto-selected config run folder: {selected_results_folder}")
            results_folder = selected_results_folder

    metadata = _load_metadata(args, results_folder)
    checkpoint_path = _resolve_checkpoint_path(args, results_folder)

    stats = NormalizationStats(
        tl_min=float(metadata["tl_min"]),
        tl_max=float(metadata["tl_max"]),
        angle_mean=float(metadata["angle_mean"]),
        angle_std=float(metadata["angle_std"]),
    )

    model = _build_model_from_metadata(metadata)
    diffusion = _build_diffusion_from_metadata(metadata, model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffusion = diffusion.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    loaded_from = _load_weights(diffusion, checkpoint, prefer_ema=args.prefer_ema)
    diffusion.eval()

    output_root = results_folder / args.output_subdir
    png_folder = output_root / "PNG"
    mat_folder = output_root / "MAT"
    png_folder.mkdir(parents=True, exist_ok=True)
    mat_folder.mkdir(parents=True, exist_ok=True)

    angles = parse_angle_list(args.angles)
    if len(angles) == 0:
        raise ValueError("No generation angles were provided. Use --angles '95,100,...'")

    roi_height = int(round(float(metadata["roi_height"])))
    roi_width = int(round(float(metadata["roi_width"])))

    reference_bundle = DatasetBundle(
        rois=np.array([], dtype=np.float32),
        angles_deg=np.array([], dtype=np.float32),
        extents=[],
    )

    # Reference ROIs are loaded for physical extents (meters) and optional comparisons.
    input_folder = str(metadata.get("input_folder", ""))
    if input_folder and os.path.isdir(input_folder):
        reference_bundle = load_rois_from_folder(
            folder=input_folder,
            roi_h=roi_height,
            roi_w=roi_width,
            rois_per_plane=1,
            roi_mode=str(metadata["roi_mode"]),
            roi_corner=(float(metadata["roi_corner"][0]), float(metadata["roi_corner"][1])),
            verbose=False,
        )

    algorithm = str(metadata.get("algorithm", "gaussian"))

    print("=" * 72)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Weights source: {loaded_from}")
    print(f"Device: {device}")
    print(f"Generating {len(angles)} sample(s)")
    print("=" * 72)

    for angle in angles:
        cond_value = normalize_single_angle(angle, stats)
        cond = torch.tensor([[cond_value]], dtype=torch.float32, device=device)

        sample_small_01 = _generate_single_sample(
            diffusion=diffusion,
            cond_tensor=cond,
            algorithm=algorithm,
            cond_scale=args.cond_scale,
            sampler=args.sampler,
            num_inference_steps=args.num_inference_steps,
        )

        train_h = int(metadata["train_height"])
        train_w = int(metadata["train_width"])
        if train_h != roi_height or train_w != roi_width:
            sample_01 = resize_fields(sample_small_01[None, ...], roi_height, roi_width)[0]
        else:
            sample_01 = sample_small_01

        sample_tl = denormalize_fields_01(sample_01[None, ...], stats)[0]

        ref_info = _nearest_reference(reference_bundle, angle)
        sample_extent = ref_info[2] if ref_info is not None else _default_extent_from_metadata(metadata)

        filename = f"plano_angulo_{angle:+07.2f}"
        mat_path = mat_folder / f"{filename}.mat"
        png_path = png_folder / f"{filename}.png"

        save_mat_h5(str(mat_path), sample_tl, extent=sample_extent)

        if ref_info is not None and args.with_reference:
            ref_tl, ref_angle, extent = ref_info
            error_map = np.abs(ref_tl - sample_tl)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            im0 = axes[0].imshow(ref_tl, cmap="jet", origin="lower", aspect="auto", vmin=stats.tl_min, vmax=stats.tl_max, extent=extent)
            axes[0].set_title(f"Reference ({ref_angle:.2f} deg)")
            axes[0].set_xlabel("X [m]")
            axes[0].set_ylabel("Z [m]")
            plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

            im1 = axes[1].imshow(sample_tl, cmap="jet", origin="lower", aspect="auto", vmin=stats.tl_min, vmax=stats.tl_max, extent=extent)
            axes[1].set_title(f"Generated ({angle:.2f} deg)")
            axes[1].set_xlabel("X [m]")
            axes[1].set_ylabel("Z [m]")
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            mae = float(np.mean(error_map))
            rmse = float(np.sqrt(np.mean(error_map ** 2)))
            im2 = axes[2].imshow(error_map, cmap="hot", origin="lower", aspect="auto", extent=extent)
            axes[2].set_title(f"|Error| MAE={mae:.3f} RMSE={rmse:.3f}")
            axes[2].set_xlabel("X [m]")
            axes[2].set_ylabel("Z [m]")
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

            fig.tight_layout()
            print(f"angle={angle:+07.2f} | ref={ref_angle:+07.2f} | MAE={mae:.3f} RMSE={rmse:.3f}")
        else:
            fig, ax = plt.subplots(1, 1, figsize=(9, 4))
            im = ax.imshow(
                sample_tl,
                cmap="jet",
                origin="lower",
                aspect="auto",
                vmin=stats.tl_min,
                vmax=stats.tl_max,
                extent=sample_extent,
            )
            ax.set_title(f"Generated ({angle:.2f} deg)")
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Z [m]")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            print(f"angle={angle:+07.2f} generated")

        fig.savefig(png_path, dpi=140)
        plt.close(fig)

    summary = {
        "checkpoint": str(checkpoint_path),
        "weights_source": loaded_from,
        "angles": angles,
        "output_png": str(png_folder),
        "output_mat": str(mat_folder),
        "algorithm": algorithm,
        "sampler": args.sampler,
        "cond_scale": args.cond_scale,
        "num_inference_steps": args.num_inference_steps,
        "axis_units": "m",
    }
    with (output_root / "generation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("=" * 72)
    print(f"PNG outputs: {png_folder}")
    print(f"MAT outputs: {mat_folder}")
    print("Done")
    print("=" * 72)


if __name__ == "__main__":
    main()
