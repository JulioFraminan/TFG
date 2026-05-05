import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

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
from config import get_train_arg_defaults  # noqa: E402
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


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train diffusion models on Transformers_2/input .mat/.h5 data using Transformers_2 infrastructure"
    )
    defaults = get_train_arg_defaults(str(REPO_ROOT))

    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Optional JSON config file. Keys match argparse destination names.",
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
    parser.add_argument(
        "--quality-profile",
        type=str,
        choices=["none", "8h-balanced", "8h-highres"],
        default="none",
        help="Optional preset to improve quality without changing pipeline structure.",
    )

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
    parser.add_argument(
        "--no-min-snr-loss-weight",
        dest="min_snr_loss_weight",
        action="store_false",
        help="Disable min-SNR loss weighting.",
    )
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
    parser.add_argument("--validation-cond-scale", type=float, default=2.5)
    parser.add_argument("--validation-sampler", type=str, choices=["ddpm", "ddim"], default="ddim")
    parser.add_argument("--validation-num-inference-steps", type=int, default=-1)
    parser.add_argument("--validation-rows-per-page", type=int, default=6)
    parser.add_argument("--validation-max-samples", type=int, default=-1)
    parser.add_argument(
        "--milestone-validation-png",
        dest="milestone_validation_png",
        action="store_true",
        help="Generate validation_real_vs_gen PNG pages at each training milestone.",
    )
    parser.add_argument(
        "--no-milestone-validation-png",
        dest="milestone_validation_png",
        action="store_false",
        help="Disable milestone validation PNG generation.",
    )
    parser.add_argument(
        "--milestone-validation-subdir",
        type=str,
        default="validation_milestones",
        help="Subdirectory inside results folder for milestone validation PNGs.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-folder", type=str, default="results/intento_mat/dit_gaussian")
    parser.add_argument(
        "--results-layout",
        type=str,
        choices=["legacy", "by_config"],
        default="by_config",
        help="legacy: write directly in results-folder. by_config: isolate each compatible config in cfg_<hash>.",
    )
    parser.add_argument(
        "--resume-if-compatible",
        dest="resume_if_compatible",
        action="store_true",
        help="When a compatible config folder already has checkpoints, resume from its latest milestone.",
    )
    parser.add_argument(
        "--no-resume-if-compatible",
        dest="resume_if_compatible",
        action="store_false",
        help="Disable automatic resume even if compatible checkpoints exist.",
    )

    parser.set_defaults(milestone_validation_png=True)
    parser.set_defaults(**defaults)

    return parser.parse_args(argv)


def _explicit_cli_destinations(argv: list[str]) -> Set[str]:
    explicit: Set[str] = set()
    for token in argv[1:]:
        if not token.startswith("--"):
            continue
        key = token[2:].split("=", 1)[0].strip()
        if not key:
            continue
        normalized = key.replace("-", "_")
        explicit.add(normalized)
        if normalized.startswith("no_") and len(normalized) > 3:
            explicit.add(normalized[3:])
    return explicit


def _apply_config_file(args: argparse.Namespace, explicit_cli_dests: Set[str]) -> None:
    if not args.config:
        return

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (PROJECT_DIR / config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config_data = json.load(handle)

    if not isinstance(config_data, dict):
        raise ValueError("Config file must contain a JSON object at top-level")

    valid_keys = set(vars(args).keys())
    unknown_keys = sorted([key for key in config_data.keys() if key not in valid_keys])
    if unknown_keys:
        raise KeyError(
            "Unknown keys in config file: "
            f"{', '.join(unknown_keys)}. "
            "Use argparse destination names (example: train_num_steps, train_batch_size)."
        )

    applied = []
    for key, value in config_data.items():
        # CLI options win over config values.
        if key in explicit_cli_dests and key != "config":
            continue
        setattr(args, key, value)
        applied.append(key)

    print(f"[config] Loaded {config_path}")
    if applied:
        print(f"[config] Applied keys: {', '.join(sorted(applied))}")


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


def _bytes_to_gib(value: float) -> float:
    return float(value) / float(1024 ** 3)


def _device_vram_gib(device: torch.device) -> float | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    try:
        props = torch.cuda.get_device_properties(device)
        return _bytes_to_gib(float(props.total_memory))
    except Exception:
        return None


def _apply_quality_profile(args: argparse.Namespace) -> None:
    if args.quality_profile == "none":
        return

    base_defaults = {
        "train_height": 256,
        "train_width": 512,
        "dit_variant": "DiT-XXS/2",
        "dit_mlp_ratio": 2.5,
        "dit_qk_norm": False,
        "dit_class_dropout": 0.2,
        "max_vanilla_attn_tokens": 4096,
        "train_batch_size": 8,
        "gradient_accumulate_every": 1,
        "train_lr": 2e-4,
        "train_num_steps": 60000,
        "save_and_sample_every": 10000,
        "objective": "pred_noise",
        "min_snr_loss_weight": False,
        "sampling_timesteps": 300,
        "validation_cond_scale": 2.5,
        "validation_num_inference_steps": -1,
    }

    if args.quality_profile == "8h-balanced":
        overrides = {
            "train_height": 320,
            "train_width": 960,
            "dit_variant": "DiT-S/8",
            "dit_mlp_ratio": 3.0,
            "dit_qk_norm": True,
            "dit_class_dropout": 0.10,
            "max_vanilla_attn_tokens": 6500,
            "train_batch_size": 4,
            "gradient_accumulate_every": 4,
            "train_lr": 1.2e-4,
            "train_num_steps": 260000,
            "save_and_sample_every": 20000,
            "objective": "pred_v",
            "min_snr_loss_weight": True,
            "sampling_timesteps": 350,
            "validation_cond_scale": 2.0,
            "validation_num_inference_steps": 450,
        }
    else:
        overrides = {
            "train_height": 384,
            "train_width": 1152,
            "dit_variant": "DiT-B/8",
            "dit_mlp_ratio": 3.5,
            "dit_qk_norm": True,
            "dit_class_dropout": 0.08,
            "max_vanilla_attn_tokens": 7500,
            "train_batch_size": 2,
            "gradient_accumulate_every": 6,
            "train_lr": 8e-5,
            "train_num_steps": 320000,
            "save_and_sample_every": 25000,
            "objective": "pred_v",
            "min_snr_loss_weight": True,
            "sampling_timesteps": 400,
            "validation_cond_scale": 2.0,
            "validation_num_inference_steps": 500,
        }

    for key, value in overrides.items():
        current = getattr(args, key)
        default_value = base_defaults.get(key, None)
        if key in base_defaults and current != default_value:
            continue
        setattr(args, key, value)

    print(
        f"[profile] Applied quality profile: {args.quality_profile} | "
        f"res=({args.train_height},{args.train_width}) | "
        f"variant={args.dit_variant} | batch={args.train_batch_size} x accum={args.gradient_accumulate_every}"
    )


def _print_dit_memory_report(args: argparse.Namespace, model: torch.nn.Module, token_count: int) -> None:
    hidden_size = int(getattr(model, "hidden_size", 0))
    num_heads = int(getattr(model, "num_heads", 1))
    depth = int(getattr(model, "depth", 0))
    if depth <= 0:
        blocks = getattr(model, "blocks", None)
        if blocks is not None:
            depth = len(blocks)
    if depth <= 0:
        depth = 1

    bytes_per_value = 2 if args.amp and args.mixed_precision_type in ("fp16", "bf16") else 4

    # Rough estimate of major activations in one transformer block during attention.
    attn_scores_bytes = args.train_batch_size * num_heads * token_count * token_count * bytes_per_value
    qkv_bytes = args.train_batch_size * token_count * hidden_size * 3 * bytes_per_value
    approx_block_bytes = (2.0 * attn_scores_bytes) + qkv_bytes
    approx_total_bytes = approx_block_bytes * depth

    block_gib = _bytes_to_gib(approx_block_bytes)
    total_gib = _bytes_to_gib(approx_total_bytes)
    vram_gib = _device_vram_gib(torch.device("cuda"))
    if vram_gib:
        block_pct = (block_gib / vram_gib) * 100.0
        total_pct = (total_gib / vram_gib) * 100.0
        vram_text = f" | vram~{vram_gib:.1f} GiB"
        pct_text = f" ({block_pct:.1f}% / {total_pct:.1f}%)"
    else:
        vram_text = ""
        pct_text = ""

    print(
        "DiT memory estimate (rough): "
        f"block~{block_gib:.2f} GiB, "
        f"all_blocks~{total_gib:.2f} GiB"
        f"{pct_text}"
        f" (micro_batch={args.train_batch_size}, effective_batch={args.train_batch_size * args.gradient_accumulate_every}, "
        f"precision_bytes={bytes_per_value}){vram_text}."
    )


CHECKPOINT_SUBDIR = "checkpoints"
COLORED_GRIDS_SUBDIR = "colored_grids"


def _checkpoint_search_dirs(results_folder: Path) -> Tuple[Path, ...]:
    return (
        results_folder / CHECKPOINT_SUBDIR,
        results_folder,
    )


def _iter_checkpoint_candidates(results_folder: Path):
    for directory in _checkpoint_search_dirs(results_folder):
        if not directory.exists():
            continue
        for path in directory.glob("model-*.pt"):
            suffix = path.stem.split("-")[-1]
            if suffix.isdigit():
                yield int(suffix), path


def _discover_latest_checkpoint(results_folder: Path) -> Path:
    candidates = list(_iter_checkpoint_candidates(results_folder))

    if len(candidates) == 0:
        searched = ", ".join(str(path) for path in _checkpoint_search_dirs(results_folder))
        raise FileNotFoundError(f"No model-*.pt checkpoints found in: {searched}")

    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def _discover_latest_milestone(results_folder: Path) -> int | None:
    candidates = list(_iter_checkpoint_candidates(results_folder))
    if len(candidates) == 0:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][0]


_CONFIG_SIGNATURE_EXCLUDED_KEYS = {
    "config",
    "results_folder",
    "results_layout",
    "resume_if_compatible",
    "train_num_steps",
    "save_and_sample_every",
    "num_samples",
    "skip_post_validation",
    "validation_output_subdir",
    "validation_angles",
    "validation_cond_scale",
    "validation_sampler",
    "validation_num_inference_steps",
    "validation_rows_per_page",
    "validation_max_samples",
    "milestone_validation_png",
    "milestone_validation_subdir",
}


def _json_stable_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_stable_value(item) for item in value]
    if isinstance(value, list):
        return [_json_stable_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_stable_value(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, float):
        return round(value, 12)
    return value


def _build_config_signature(args: argparse.Namespace) -> str:
    payload = {
        key: _json_stable_value(value)
        for key, value in sorted(vars(args).items())
        if key not in _CONFIG_SIGNATURE_EXCLUDED_KEYS
    }
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(payload_json.encode("utf-8")).hexdigest()[:12]


_CONFIG_RUN_INDEX_FILENAME = "_config_run_index.json"


def _slugify_name_component(value: str, max_len: int = 24) -> str:
    lowered = value.lower()
    out_chars: list[str] = []
    prev_dash = False

    for char in lowered:
        if char.isalnum():
            out_chars.append(char)
            prev_dash = False
        else:
            if not prev_dash:
                out_chars.append("-")
                prev_dash = True

    slug = "".join(out_chars).strip("-")
    if not slug:
        slug = "na"

    trimmed = slug[:max_len].strip("-")
    return trimmed if trimmed else "na"


def _format_lr_tag(train_lr: float) -> str:
    text = f"{float(train_lr):.0e}"
    if "e" in text:
        mantissa, exp = text.split("e", 1)
        try:
            text = f"{mantissa}e{int(exp)}"
        except ValueError:
            pass
    return _slugify_name_component(text, max_len=12)


def _build_readable_run_name(args: argparse.Namespace, run_id: int) -> str:
    res_tag = f"{int(args.train_height)}x{int(args.train_width)}"
    batch_tag = f"b{int(args.train_batch_size)}x{int(args.gradient_accumulate_every)}"
    algo_tag = _slugify_name_component(str(args.algorithm), max_len=10)

    if args.model_type == "dit":
        variant = _slugify_name_component(str(args.dit_variant).replace("/", "p"), max_len=18)
        attn = _slugify_name_component(str(args.dit_attn_type), max_len=10)
        model_tag = f"dit-{variant}-{attn}"
    else:
        model_tag = f"unet-d{int(args.unet_dim)}"

    lr_tag = f"lr{_format_lr_tag(float(args.train_lr))}"
    return f"run_{run_id:04d}_{model_tag}_{res_tag}_{algo_tag}_{batch_tag}_{lr_tag}"


def _discover_max_existing_run_id(base_folder: Path) -> int:
    max_run_id = 0
    if not base_folder.exists():
        return max_run_id

    for child in base_folder.iterdir():
        if not child.is_dir():
            continue

        parts = child.name.split("_", 2)
        if len(parts) < 2 or parts[0] != "run" or not parts[1].isdigit():
            continue

        max_run_id = max(max_run_id, int(parts[1]))

    return max_run_id


def _load_config_run_index(index_path: Path, base_folder: Path) -> Dict[str, Any]:
    existing_max = _discover_max_existing_run_id(base_folder)
    baseline_next_id = existing_max + 1 if existing_max > 0 else 1
    default_index: Dict[str, Any] = {
        "version": 1,
        "next_id": baseline_next_id,
        "by_signature": {},
    }

    if not index_path.exists():
        return default_index

    try:
        with index_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        if not isinstance(data, dict):
            return default_index

        by_signature = data.get("by_signature", {})
        if not isinstance(by_signature, dict):
            by_signature = {}

        next_id = data.get("next_id", baseline_next_id)
        if not isinstance(next_id, int) or next_id <= 0:
            next_id = baseline_next_id

        next_id = max(next_id, baseline_next_id)

        return {
            "version": 1,
            "next_id": next_id,
            "by_signature": by_signature,
        }
    except Exception:
        return default_index


def _save_config_run_index(index_path: Path, index_data: Dict[str, Any]) -> None:
    with index_path.open("w", encoding="utf-8") as handle:
        json.dump(index_data, handle, indent=2, sort_keys=True)


def _resolve_readable_config_run_folder(base_folder: Path, args: argparse.Namespace, signature: str) -> Path:
    base_folder.mkdir(parents=True, exist_ok=True)

    index_path = base_folder / _CONFIG_RUN_INDEX_FILENAME
    index_data = _load_config_run_index(index_path, base_folder)
    by_signature = index_data.setdefault("by_signature", {})

    mapped_name = by_signature.get(signature)
    if isinstance(mapped_name, str) and mapped_name.strip():
        return base_folder / mapped_name

    run_id = int(index_data.get("next_id", 1))
    while True:
        run_name = _build_readable_run_name(args, run_id)
        candidate = base_folder / run_name
        if not candidate.exists():
            break
        run_id += 1

    by_signature[signature] = run_name
    index_data["next_id"] = run_id + 1
    _save_config_run_index(index_path, index_data)

    return base_folder / run_name


def _resolve_results_folder(args: argparse.Namespace) -> tuple[Path, Path, Optional[str]]:
    base_folder = (PROJECT_DIR / args.results_folder).resolve() if not os.path.isabs(args.results_folder) else Path(args.results_folder)

    if args.results_layout == "legacy":
        return base_folder, base_folder, None

    signature = _build_config_signature(args)
    resolved_folder = _resolve_readable_config_run_folder(base_folder, args, signature)
    return base_folder, resolved_folder, signature


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


def run_training(
    args: argparse.Namespace,
    explicit_cli_dests: Optional[Set[str]] = None,
    on_validation: Optional[callable] = None,
) -> Dict[str, Any]:
    if explicit_cli_dests is None:
        explicit_cli_dests = _explicit_cli_destinations(sys.argv)

    _apply_config_file(args, explicit_cli_dests)
    _apply_quality_profile(args)

    requested_angles = parse_angle_list(args.validation_angles) if args.validation_angles.strip() else []

    set_seed(args.seed)

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

    roi_h_px = int(train_bundle.rois.shape[1])
    roi_w_px = int(train_bundle.rois.shape[2])
    if args.train_height != roi_h_px or args.train_width != roi_w_px:
        print(
            "[roi] Overriding train size to match ROI: "
            f"({args.train_height}, {args.train_width}) -> ({roi_h_px}, {roi_w_px})"
        )
        args.train_height = roi_h_px
        args.train_width = roi_w_px

    # Keep signature and final model config aligned when auto patch-size adjustment is enabled.
    _maybe_auto_adjust_dit_variant(args, train_h=args.train_height, train_w=args.train_width)

    base_results_folder, results_folder, config_signature = _resolve_results_folder(args)
    results_folder.mkdir(parents=True, exist_ok=True)

    print(f"[results] Base folder: {base_results_folder}")
    if config_signature is not None:
        print(f"[results] Config signature: {config_signature} -> {results_folder}")
    else:
        print(f"[results] Legacy layout folder: {results_folder}")

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
        _print_dit_memory_report(args, model=model, token_count=token_count)

    params = sum(param.numel() for param in model.parameters())
    print(f"Model parameters: {params:,}")

    metadata_for_validation = _build_metadata(args, stats, train_bundle, val_bundle)

    milestone_validation_callback = None
    if args.milestone_validation_png:
        if val_bundle.rois.size == 0:
            print("[milestone-validation] Skipped milestone PNG validation: no validation .mat files found")
        else:
            from validation import run_validation

            def _run_milestone_validation_png(milestone: int, checkpoint_path: Path) -> None:
                output_subdir = f"{args.milestone_validation_subdir}/m{milestone:04d}"
                print("=" * 72)
                print(f"Running milestone PNG validation (milestone {milestone})")
                print("=" * 72)
                summary = run_validation(
                    results_folder=results_folder,
                    metadata=metadata_for_validation,
                    checkpoint_path=checkpoint_path,
                    prefer_ema=True,
                    validation_folder_override=args.validation_folder,
                    requested_angles=requested_angles,
                    output_subdir=output_subdir,
                    cond_scale=args.validation_cond_scale,
                    sampler=args.validation_sampler,
                    num_inference_steps=args.validation_num_inference_steps,
                    rows_per_page=args.validation_rows_per_page,
                    max_samples=args.validation_max_samples,
                    seed=args.seed,
                    save_mat=False,
                    save_metrics_csv=False,
                    save_summary_json=False,
                    save_error_vs_angle=False,
                )
                if summary.get("status") == "ok":
                    print(
                        "[milestone-validation] Mean errors "
                        f"MAE={summary.get('mean_mae')} | "
                        f"RMSE={summary.get('mean_rmse')} | "
                        f"MAX={summary.get('mean_max_error')}"
                    )
                if callable(on_validation):
                    on_validation(milestone, summary)

            milestone_validation_callback = _run_milestone_validation_png

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
        on_milestone=milestone_validation_callback,
        checkpoint_subdir=CHECKPOINT_SUBDIR,
        sample_subdir=COLORED_GRIDS_SUBDIR,
    )

    resumed_from: int | None = None
    if args.resume_if_compatible:
        try:
            latest_checkpoint_path = _discover_latest_checkpoint(results_folder)
        except FileNotFoundError:
            latest_checkpoint_path = None

        if latest_checkpoint_path is not None:
            latest_milestone = int(latest_checkpoint_path.stem.split("-")[-1])
            try:
                trainer.load(latest_milestone, checkpoint_path=latest_checkpoint_path)
                resumed_from = latest_milestone
                print(f"[resume] Loaded {latest_checkpoint_path} | step={trainer.step}")
            except Exception as error:
                print(
                    "[resume][WARN] Could not resume from latest checkpoint "
                    f"{latest_checkpoint_path}: {error}. "
                    "Training will start from scratch."
                )

    metadata = dict(metadata_for_validation)
    metadata["results_layout"] = args.results_layout
    metadata["results_base_folder"] = str(base_results_folder)
    metadata["resolved_results_folder"] = str(results_folder)
    metadata["config_signature"] = config_signature
    metadata["resume_if_compatible"] = bool(args.resume_if_compatible)
    metadata["resumed_from_milestone"] = resumed_from
    metadata["checkpoint_subdir"] = CHECKPOINT_SUBDIR
    metadata["colored_grid_subdir"] = COLORED_GRIDS_SUBDIR
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
        return {
            "status": "skipped",
            "reason": "skip_post_validation",
        }

    if not trainer.accelerator.is_main_process:
        return {
            "status": "skipped",
            "reason": "not_main_process",
        }

    if val_bundle.rois.size == 0:
        print("Validation skipped: no validation .mat files found")
        return {
            "status": "skipped",
            "reason": "no_validation_data",
        }

    try:
        checkpoint_path = _discover_latest_checkpoint(results_folder)
    except FileNotFoundError as error:
        print(f"Validation skipped: {error}")
        return {
            "status": "skipped",
            "reason": "no_checkpoint",
        }

    from validation import run_validation

    print("=" * 72)
    print("Running post-training validation report")
    print("=" * 72)

    summary = run_validation(
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
    if callable(on_validation):
        final_step = int(args.train_num_steps) if int(args.train_num_steps) > 0 else 0
        on_validation(final_step, summary)
    return summary


def main() -> None:
    args = parse_args()
    explicit_cli_dests = _explicit_cli_destinations(sys.argv)
    run_training(args, explicit_cli_dests=explicit_cli_dests)


if __name__ == "__main__":
    main()
