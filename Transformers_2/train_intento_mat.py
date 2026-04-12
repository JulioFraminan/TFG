import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR
REPO_ROOT = PROJECT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from denoising_diffusion_pytorch.continuous_classifier_free_guidance import (  # noqa: E402
    GaussianDiffusion,
    Trainer,
    Unet,
)
from denoising_diffusion_pytorch.dit import DiT_models  # noqa: E402
from intento_mat_utils import (  # noqa: E402
    NormalizationStats,
    DatasetBundle,
    default_intento_input_folder,
    default_intento_validation_folder,
    ensure_min_samples,
    load_rois_from_folder,
    normalize_angles,
    normalize_fields_01,
    parse_angle_list,
    resize_fields,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train diffusion models on Intento_Transformers .mat/.h5 data using Transformers_2 infrastructure"
    )

    parser.add_argument("--input-folder", type=str, default=default_intento_input_folder(str(REPO_ROOT)))
    parser.add_argument("--validation-folder", type=str, default=default_intento_validation_folder(str(REPO_ROOT)))

    parser.add_argument("--roi-height", type=float, default=700)
    parser.add_argument("--roi-width", type=float, default=2000)
    parser.add_argument("--rois-per-plane", type=int, default=1)
    parser.add_argument("--roi-mode", type=str, choices=["corner_fixed", "center_max"], default="corner_fixed")
    parser.add_argument("--roi-corner-x", type=float, default=0.0)
    parser.add_argument("--roi-corner-z", type=float, default=10.0)

    parser.add_argument("--train-height", type=int, default=256)
    parser.add_argument("--train-width", type=int, default=512)

    parser.add_argument("--model-type", type=str, choices=["unet", "dit"], default="dit")
    parser.add_argument("--dit-variant", type=str, default="DiT-XXS/2")
    parser.add_argument("--dit-class-dropout", type=float, default=0.2)
    parser.add_argument("--dit-attn-type", type=str, choices=["vanilla", "linear", "window"], default="vanilla")
    parser.add_argument("--dit-mlp-ratio", type=float, default=2.5)
    parser.add_argument("--dit-qk-norm", action="store_true")
    parser.add_argument(
        "--max-vanilla-attn-tokens",
        type=int,
        default=4096,
        help="Automatic safety limit for DiT vanilla attention tokens. Set <=0 to disable auto-adjust.",
    )

    parser.add_argument("--unet-dim", type=int, default=64)
    parser.add_argument("--unet-dim-mults", type=str, default="1,2,4")
    parser.add_argument("--unet-cond-drop-prob", type=float, default=0.2)

    parser.add_argument("--algorithm", type=str, choices=["gaussian", "flowmatching"], default="gaussian")
    parser.add_argument("--objective", type=str, choices=["pred_noise", "pred_x0", "pred_v"], default="pred_noise")
    parser.add_argument("--beta-schedule", type=str, choices=["linear", "cosine", "sigmoid"], default="cosine")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--sampling-timesteps", type=int, default=300)
    parser.add_argument("--min-snr-loss-weight", action="store_true")
    parser.add_argument("--min-snr-gamma", type=float, default=5.0)

    parser.add_argument("--flow-path-type", type=str, choices=["Linear", "GVP", "VP"], default="Linear")
    parser.add_argument("--flow-prediction", type=str, choices=["velocity", "noise", "score"], default="velocity")
    parser.add_argument("--flow-loss-weight", type=str, choices=["none", "velocity", "likelihood"], default="none")
    parser.add_argument("--flow-num-steps", type=int, default=250)
    parser.add_argument("--flow-sampling-method", type=str, default="euler")
    parser.add_argument("--flow-atol", type=float, default=1e-6)
    parser.add_argument("--flow-rtol", type=float, default=1e-3)

    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--train-lr", type=float, default=2e-4)
    parser.add_argument("--train-num-steps", type=int, default=60000)
    parser.add_argument("--gradient-accumulate-every", type=int, default=1)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--save-and-sample-every", type=int, default=10000)
    parser.add_argument("--num-samples", type=int, default=9)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--mixed-precision-type", type=str, default="bf16")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--split-batches", action="store_true")
    parser.add_argument("--disable-lr-scheduler", action="store_true")
    parser.add_argument("--use-cpu", action="store_true")

    parser.add_argument("--skip-post-validation", action="store_true")
    parser.add_argument("--validation-output-subdir", type=str, default="validation")
    parser.add_argument("--validation-angles", type=str, default="")
    parser.add_argument("--validation-cond-scale", type=float, default=6.0)
    parser.add_argument("--validation-sampler", type=str, choices=["ddpm", "ddim"], default="ddim")
    parser.add_argument("--validation-num-inference-steps", type=int, default=-1)
    parser.add_argument("--validation-rows-per-page", type=int, default=6)
    parser.add_argument("--validation-max-samples", type=int, default=-1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-folder", type=str, default="results/intento_mat/dit_gaussian")

    return parser.parse_args()


def parse_dim_mults(dim_mults_text: str) -> Tuple[int, ...]:
    chunks = [chunk.strip() for chunk in dim_mults_text.split(",") if chunk.strip()]
    if not chunks:
        raise ValueError("--unet-dim-mults is empty")
    return tuple(int(chunk) for chunk in chunks)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _extract_variant_patch_size(variant_name: str) -> int | None:
    if "/" not in variant_name:
        return None
    maybe_patch = variant_name.rsplit("/", 1)[-1]
    if maybe_patch.isdigit():
        return int(maybe_patch)
    return None


def _replace_variant_patch_size(variant_name: str, patch_size: int) -> str:
    if "/" not in variant_name:
        return variant_name
    return f"{variant_name.rsplit('/', 1)[0]}/{patch_size}"


def _token_count(train_h: int, train_w: int, patch_size: int) -> int:
    return (train_h // patch_size) * (train_w // patch_size)


def _discover_latest_checkpoint(results_folder: Path) -> Path:
    candidates = []
    for path in results_folder.glob("model-*.pt"):
        suffix = path.stem.split("-")[-1]
        if suffix.isdigit():
            candidates.append((int(suffix), path))

    if len(candidates) == 0:
        raise FileNotFoundError(f"No model-*.pt checkpoints found in {results_folder}")

    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def _maybe_auto_adjust_dit_variant(args: argparse.Namespace, train_h: int, train_w: int) -> None:
    if args.model_type != "dit" or args.dit_attn_type != "vanilla":
        return

    if args.max_vanilla_attn_tokens <= 0:
        return

    current_patch = _extract_variant_patch_size(args.dit_variant)
    if current_patch is None:
        return

    if train_h % current_patch != 0 or train_w % current_patch != 0:
        return

    current_tokens = _token_count(train_h, train_w, current_patch)
    if current_tokens <= args.max_vanilla_attn_tokens:
        return

    candidate_patches = (1, 2, 4, 8)
    replacement_variant = None
    replacement_patch = None

    for patch in candidate_patches:
        if patch < current_patch:
            continue
        if train_h % patch != 0 or train_w % patch != 0:
            continue

        candidate_variant = _replace_variant_patch_size(args.dit_variant, patch)
        if candidate_variant not in DiT_models:
            continue

        candidate_tokens = _token_count(train_h, train_w, patch)
        if candidate_tokens <= args.max_vanilla_attn_tokens:
            replacement_variant = candidate_variant
            replacement_patch = patch
            break

    if replacement_variant is None or replacement_patch is None:
        raise ValueError(
            "Vanilla DiT attention would exceed the token safety limit and no compatible larger patch variant "
            "was found. Use a larger --dit-variant patch (e.g. /8), reduce --train-height/--train-width, "
            "or switch --dit-attn-type to linear/window."
        )

    print(
        f"[auto] Switching DiT variant {args.dit_variant} -> {replacement_variant} "
        f"to keep vanilla attention tokens <= {args.max_vanilla_attn_tokens} "
        f"({current_tokens} -> {_token_count(train_h, train_w, replacement_patch)})."
    )
    args.dit_variant = replacement_variant


def build_model(args: argparse.Namespace, train_h: int, train_w: int) -> torch.nn.Module:
    if args.model_type == "unet":
        dim_mults = parse_dim_mults(args.unet_dim_mults)
        return Unet(
            dim=args.unet_dim,
            cond_dim=1,
            cond_drop_prob=args.unet_cond_drop_prob,
            channels=1,
            dim_mults=dim_mults,
        )

    if args.dit_variant not in DiT_models:
        available = ", ".join(sorted(DiT_models.keys()))
        raise KeyError(f"Unknown DiT variant '{args.dit_variant}'. Available variants: {available}")

    _maybe_auto_adjust_dit_variant(args, train_h=train_h, train_w=train_w)

    model = DiT_models[args.dit_variant](
        input_size=(train_h, train_w),
        cond_dim=1,
        class_dropout_prob=args.dit_class_dropout,
        in_channels=1,
        learn_sigma=False,
        attn_type=args.dit_attn_type,
        mlp_ratio=args.dit_mlp_ratio,
        qk_norm=args.dit_qk_norm,
    )

    patch_size = int(getattr(model, "patch_size", 1))
    if train_h % patch_size != 0 or train_w % patch_size != 0:
        raise ValueError(
            f"train shape ({train_h}, {train_w}) must be divisible by DiT patch_size={patch_size}."
        )

    return model


def build_diffusion(args: argparse.Namespace, model: torch.nn.Module, train_h: int, train_w: int):
    if args.algorithm == "gaussian":
        return GaussianDiffusion(
            model,
            image_size=(train_h, train_w),
            objective=args.objective,
            beta_schedule=args.beta_schedule,
            sampling_timesteps=args.sampling_timesteps,
            timesteps=args.timesteps,
            min_snr_loss_weight=args.min_snr_loss_weight,
            min_snr_gamma=args.min_snr_gamma,
        )

    from denoising_diffusion_pytorch.transport import FlowMatching, Sampler, create_transport

    flow_loss_weight = None if args.flow_loss_weight == "none" else args.flow_loss_weight
    transport = create_transport(
        path_type=args.flow_path_type,
        prediction=args.flow_prediction,
        loss_weight=flow_loss_weight,
    )
    sampler = Sampler(transport=transport)
    return FlowMatching(
        sampler=sampler,
        neural_net=model,
        input_size=(train_h, train_w),
        cond_scale=2.0,
        num_sampling_steps=args.flow_num_steps,
        sampling_method=args.flow_sampling_method,
        sampler_atol=args.flow_atol,
        sampler_rtol=args.flow_rtol,
    )


def _build_metadata(
    args: argparse.Namespace,
    stats: NormalizationStats,
    train_bundle: DatasetBundle,
    val_bundle: DatasetBundle,
) -> Dict:
    return {
        "input_folder": args.input_folder,
        "validation_folder": args.validation_folder,
        "roi_height": args.roi_height,
        "roi_width": args.roi_width,
        "rois_per_plane": args.rois_per_plane,
        "roi_mode": args.roi_mode,
        "roi_corner": [args.roi_corner_x, args.roi_corner_z],
        "train_height": args.train_height,
        "train_width": args.train_width,
        "model_type": args.model_type,
        "dit_variant": args.dit_variant,
        "dit_class_dropout": args.dit_class_dropout,
        "dit_attn_type": args.dit_attn_type,
        "dit_mlp_ratio": args.dit_mlp_ratio,
        "dit_qk_norm": args.dit_qk_norm,
        "max_vanilla_attn_tokens": args.max_vanilla_attn_tokens,
        "unet_dim": args.unet_dim,
        "unet_dim_mults": args.unet_dim_mults,
        "unet_cond_drop_prob": args.unet_cond_drop_prob,
        "algorithm": args.algorithm,
        "objective": args.objective,
        "beta_schedule": args.beta_schedule,
        "timesteps": args.timesteps,
        "sampling_timesteps": args.sampling_timesteps,
        "min_snr_loss_weight": args.min_snr_loss_weight,
        "min_snr_gamma": args.min_snr_gamma,
        "flow_path_type": args.flow_path_type,
        "flow_prediction": args.flow_prediction,
        "flow_loss_weight": args.flow_loss_weight,
        "flow_num_steps": args.flow_num_steps,
        "flow_sampling_method": args.flow_sampling_method,
        "flow_atol": args.flow_atol,
        "flow_rtol": args.flow_rtol,
        "train_batch_size": args.train_batch_size,
        "train_lr": args.train_lr,
        "train_num_steps": args.train_num_steps,
        "gradient_accumulate_every": args.gradient_accumulate_every,
        "ema_decay": args.ema_decay,
        "save_and_sample_every": args.save_and_sample_every,
        "validation_output_subdir": args.validation_output_subdir,
        "tl_min": stats.tl_min,
        "tl_max": stats.tl_max,
        "angle_mean": stats.angle_mean,
        "angle_std": stats.angle_std,
        "train_samples": int(train_bundle.rois.shape[0]),
        "validation_samples": int(val_bundle.rois.shape[0]),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    results_folder = (PROJECT_DIR / args.results_folder).resolve() if not os.path.isabs(args.results_folder) else Path(args.results_folder)
    results_folder.mkdir(parents=True, exist_ok=True)

    roi_corner = (args.roi_corner_x, args.roi_corner_z)

    print("=" * 72)
    print("Loading training data from .mat / h5")
    print("=" * 72)

    train_bundle = load_rois_from_folder(
        folder=args.input_folder,
        roi_h=args.roi_height,
        roi_w=args.roi_width,
        rois_per_plane=args.rois_per_plane,
        roi_mode=args.roi_mode,
        roi_corner=roi_corner,
        verbose=True,
    )

    if train_bundle.rois.size == 0:
        raise RuntimeError("No training ROIs found. Check input folder and ROI settings.")

    val_bundle = DatasetBundle(
        rois=np.array([], dtype=np.float32),
        angles_deg=np.array([], dtype=np.float32),
        extents=[],
    )
    if args.validation_folder and os.path.isdir(args.validation_folder):
        val_bundle = load_rois_from_folder(
            folder=args.validation_folder,
            roi_h=args.roi_height,
            roi_w=args.roi_width,
            rois_per_plane=1,
            roi_mode=args.roi_mode,
            roi_corner=roi_corner,
            verbose=False,
        )

    stats = NormalizationStats.from_training_data(train_bundle.rois, train_bundle.angles_deg)

    train_fields_01 = normalize_fields_01(train_bundle.rois, stats)
    train_angles_norm = normalize_angles(train_bundle.angles_deg, stats)

    train_fields_01 = resize_fields(train_fields_01, target_h=args.train_height, target_w=args.train_width)

    train_fields_01, train_angles_norm = ensure_min_samples(
        train_fields_01,
        train_angles_norm,
        minimum=100,
        noise_std=0.01,
    )

    train_tensor_x = torch.from_numpy(train_fields_01[:, None, :, :]).float()
    train_tensor_c = torch.from_numpy(train_angles_norm[:, None]).float()
    train_dataset = TensorDataset(train_tensor_x, train_tensor_c)

    print(f"Training samples used: {len(train_dataset)}")
    print(f"Train image shape: ({args.train_height}, {args.train_width})")
    print(f"TL range: [{stats.tl_min:.4f}, {stats.tl_max:.4f}]")
    print(f"Angle normalization: mean={stats.angle_mean:.4f}, std={stats.angle_std:.4f}")

    model = build_model(args, train_h=args.train_height, train_w=args.train_width)
    diffusion = build_diffusion(args, model=model, train_h=args.train_height, train_w=args.train_width)

    if args.model_type == "dit":
        patch_size = int(getattr(model, "patch_size", 1))
        token_count = _token_count(args.train_height, args.train_width, patch_size)
        print(
            f"DiT config: variant={args.dit_variant}, patch_size={patch_size}, "
            f"tokens={token_count}, attn={args.dit_attn_type}"
        )

    params = sum(param.numel() for param in model.parameters())
    print(f"Model parameters: {params:,}")

    trainer = Trainer(
        diffusion,
        dataset=train_dataset,
        train_batch_size=args.train_batch_size,
        train_lr=args.train_lr,
        num_samples=args.num_samples,
        train_num_steps=args.train_num_steps,
        gradient_accumulate_every=args.gradient_accumulate_every,
        ema_decay=args.ema_decay,
        calculate_fid=False,
        results_folder=str(results_folder),
        save_and_sample_every=args.save_and_sample_every,
        augment_horizontal_flip=False,
        amp=args.amp,
        mixed_precision_type=args.mixed_precision_type,
        split_batches=args.split_batches,
        max_grad_norm=args.max_grad_norm,
        use_cpu=args.use_cpu,
        use_lr_scheduler=not args.disable_lr_scheduler,
        compile_model=args.compile_model,
    )

    metadata = _build_metadata(args, stats, train_bundle, val_bundle)
    metadata_path = results_folder / "intento_training_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    shutil.copy2(__file__, results_folder / Path(__file__).name)
    shutil.copy2(SCRIPT_DIR / "intento_mat_utils.py", results_folder / "intento_mat_utils.py")

    print("=" * 72)
    print("Training starts")
    print("=" * 72)
    trainer.train()

    if args.skip_post_validation:
        print("Post-training validation skipped (--skip-post-validation).")
        return

    if not trainer.accelerator.is_main_process:
        return

    if val_bundle.rois.size == 0:
        print("Validation skipped: no validation .mat files found")
        return

    try:
        checkpoint_path = _discover_latest_checkpoint(results_folder)
    except FileNotFoundError as error:
        print(f"Validation skipped: {error}")
        return

    requested_angles = parse_angle_list(args.validation_angles) if args.validation_angles.strip() else []

    from validation import run_validation

    print("=" * 72)
    print("Running post-training validation report")
    print("=" * 72)

    run_validation(
        results_folder=results_folder,
        metadata=metadata,
        checkpoint_path=checkpoint_path,
        prefer_ema=True,
        validation_folder_override=args.validation_folder,
        requested_angles=requested_angles,
        output_subdir=args.validation_output_subdir,
        cond_scale=args.validation_cond_scale,
        sampler=args.validation_sampler,
        num_inference_steps=args.validation_num_inference_steps,
        rows_per_page=args.validation_rows_per_page,
        max_samples=args.validation_max_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
