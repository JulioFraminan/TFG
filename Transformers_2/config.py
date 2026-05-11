from pathlib import Path
from typing import Any, Dict

# Shared defaults for train_intento_mat.py and generate_intento_mat.py.
DEFAULT_RESULTS_FOLDER = "results/intento_mat/dit_gaussian"

ROI_DEFAULTS: Dict[str, Any] = {
    #"roi_height": 700,
    #"roi_width": 2000,
    "roi_height": 480,
    "roi_width": 1520,
    "rois_per_plane": 1,
    "roi_mode": "corner_fixed",
    "roi_corner_x": 50.0,
    "roi_corner_z": -5.0,
}

TRAIN_IMAGE_DEFAULTS: Dict[str, Any] = {
    "train_height": 480,
    "train_width": 1520,
    "quality_profile": "none",
}

MODEL_DEFAULTS: Dict[str, Any] = {
    "model_type": "dit",
    "dit_variant": "DiT-S/8",
    "dit_class_dropout": 0.124,
    "dit_attn_type": "linear",
    "dit_mlp_ratio": 2.872,
    "dit_qk_norm": True,
    "max_vanilla_attn_tokens": 4096,
    "unet_dim": 64,
    "unet_dim_mults": "1,2,4",
    "unet_cond_drop_prob": 0.2,
}

DIFFUSION_DEFAULTS: Dict[str, Any] = {
    "algorithm": "gaussian",
    "objective": "pred_v",
    "beta_schedule": "cosine",
    "timesteps": 1000,
    "sampling_timesteps": 450,
    "min_snr_loss_weight": False,
    "min_snr_gamma": 6.9,
}

FLOW_DEFAULTS: Dict[str, Any] = {
    "flow_path_type": "Linear",
    "flow_prediction": "velocity",
    "flow_loss_weight": "none",
    "flow_num_steps": 250,
    "flow_sampling_method": "euler",
    "flow_atol": 1e-6,
    "flow_rtol": 1e-3,
}

OPTIMIZATION_DEFAULTS: Dict[str, Any] = {
    "train_batch_size": 2,
    "train_lr": 2.15e-4,
    "train_num_steps": 25000,
    "gradient_accumulate_every": 1,
    "ema_decay": 0.995,
    "save_and_sample_every": 1000,
    "num_samples": 9,
    "max_grad_norm": 1.0,
}

RUNTIME_DEFAULTS: Dict[str, Any] = {
    "amp": True,
    "mixed_precision_type": "bf16",
    "compile_model": True,
    "split_batches": False,
    "disable_lr_scheduler": False,
    "use_cpu": False,
}

TRAIN_VALIDATION_DEFAULTS: Dict[str, Any] = {
    "skip_post_validation": False,
    "validation_output_subdir": "validation",
    "validation_angles": "",
    "validation_cond_scale": 2.649,
    "validation_sampler": "ddim",
    "validation_num_inference_steps": 450,
    "validation_rows_per_page": 6,
    "validation_max_samples": -1,
}

GENERATION_DEFAULTS: Dict[str, Any] = {
    "metadata_path": "",
    "checkpoint": "",
    "milestone": -1,
    "prefer_ema": True,
    "angles": "95,100,110,120,130,140,150,160,170,180",
    "cond_scale": 2.5,
    "sampler": "ddim",
    "num_inference_steps": 400,
    "output_subdir": "generate",
    "with_reference": False,
}

SEED_DEFAULT = 42


def _default_input_folder(repo_root: Path) -> str:
    return str((repo_root / "Transformers_2" / "input").resolve())


def _default_validation_folder(repo_root: Path) -> str:
    return str((Path(_default_input_folder(repo_root)) / "validation").resolve())


def get_train_arg_defaults(repo_root: str) -> Dict[str, Any]:
    repo_root_path = Path(repo_root).resolve()
    defaults: Dict[str, Any] = {
        "config": "",
        "input_folder": _default_input_folder(repo_root_path),
        "validation_folder": _default_validation_folder(repo_root_path),
        "results_folder": DEFAULT_RESULTS_FOLDER,
        "results_layout": "by_config",
        "resume_if_compatible": True,
        "seed": SEED_DEFAULT,
    }
    defaults.update(ROI_DEFAULTS)
    defaults.update(TRAIN_IMAGE_DEFAULTS)
    defaults.update(MODEL_DEFAULTS)
    defaults.update(DIFFUSION_DEFAULTS)
    defaults.update(FLOW_DEFAULTS)
    defaults.update(OPTIMIZATION_DEFAULTS)
    defaults.update(RUNTIME_DEFAULTS)
    defaults.update(TRAIN_VALIDATION_DEFAULTS)
    return defaults


def get_generate_arg_defaults() -> Dict[str, Any]:
    defaults = {
        "results_folder": DEFAULT_RESULTS_FOLDER,
        "auto_select_config_run": True,
        "seed": SEED_DEFAULT,
    }
    defaults.update(GENERATION_DEFAULTS)
    return defaults
