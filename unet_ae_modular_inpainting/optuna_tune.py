import argparse
import gc
import math
import importlib
import sys
from pathlib import Path

import optuna
import torch


PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config as cfg  # noqa: E402


DEFAULT_TRIALS = 100
DEFAULT_STUDY_NAME = "optuna_unet_ae_no_inpaint_v3"
DEFAULT_RESULTS_ROOT = "results/optuna_unet_ae_no_inpaint_v3"
DEFAULT_EPOCHS = 120
DEFAULT_TIMEOUT = 0
DEFAULT_SEED = 42
DEFAULT_N_JOBS = 1

SEARCH_SPACE = {
    "batch_size": [2, 4, 8],
    "learning_rate": (5e-6, 1e-3),
    "weight_decay": (1e-7, 1e-3),
    "model_dropout": (0.0, 0.25),
    "use_augmentation": [False, True],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna tuning for unet_ae_modular_inpainting/train.py without inpainting"
    )
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--study-name", type=str, default=DEFAULT_STUDY_NAME)
    parser.add_argument("--storage", type=str, default="")
    parser.add_argument("--results-root", type=str, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    return parser.parse_args()


def _trial_results_root(args: argparse.Namespace, trial: optuna.trial.Trial) -> Path:
    return Path(args.results_root).resolve() / f"trial_{trial.number:04d}"


def _patch_config_for_trial(args: argparse.Namespace, trial: optuna.trial.Trial) -> dict:
    trial_root = _trial_results_root(args, trial)
    output_root = trial_root / "output"

    batch_size = trial.suggest_categorical("batch_size", SEARCH_SPACE["batch_size"])
    learning_rate = trial.suggest_float(
        "learning_rate",
        SEARCH_SPACE["learning_rate"][0],
        SEARCH_SPACE["learning_rate"][1],
        log=True,
    )
    use_augmentation = trial.suggest_categorical(
        "use_augmentation",
        SEARCH_SPACE["use_augmentation"],
    )
    weight_decay = trial.suggest_float(
        "weight_decay",
        SEARCH_SPACE["weight_decay"][0],
        SEARCH_SPACE["weight_decay"][1],
        log=True,
    )
    model_dropout = trial.suggest_float(
        "model_dropout",
        SEARCH_SPACE["model_dropout"][0],
        SEARCH_SPACE["model_dropout"][1],
    )

    previous = {
        "BATCH_SIZE": cfg.BATCH_SIZE,
        "EPOCHS": cfg.EPOCHS,
        "LEARNING_RATE": cfg.LEARNING_RATE,
        "WEIGHT_DECAY": cfg.WEIGHT_DECAY,
        "MODEL_DROPOUT": cfg.MODEL_DROPOUT,
        "USE_AUGMENTATION": cfg.USE_AUGMENTATION,
        "INPAINT_MODE": cfg.INPAINT_MODE,
        "INPAINT_PRESERVE_KNOWN": cfg.INPAINT_PRESERVE_KNOWN,
        "OUTPUT_FOLDER": cfg.OUTPUT_FOLDER,
        "TRAIN_PNG_FOLDER": cfg.TRAIN_PNG_FOLDER,
        "TRAIN_MAT_FOLDER": cfg.TRAIN_MAT_FOLDER,
        "GENERATE_PNG_FOLDER": cfg.GENERATE_PNG_FOLDER,
        "GENERATE_MAT_FOLDER": cfg.GENERATE_MAT_FOLDER,
        "ANALYSIS_PNG_FOLDER": cfg.ANALYSIS_PNG_FOLDER,
        "VALIDATION_PNG_FOLDER": cfg.VALIDATION_PNG_FOLDER,
        "VALIDATION_MAT_FOLDER": cfg.VALIDATION_MAT_FOLDER,
        "MODEL_PATH": cfg.MODEL_PATH,
    }

    cfg.BATCH_SIZE = int(batch_size)
    cfg.EPOCHS = int(args.epochs)
    cfg.LEARNING_RATE = float(learning_rate)
    cfg.WEIGHT_DECAY = float(weight_decay)
    cfg.MODEL_DROPOUT = float(model_dropout)
    cfg.USE_AUGMENTATION = bool(use_augmentation)

    # Inpainting explicitly disabled for this tuning run.
    cfg.INPAINT_MODE = "none"
    cfg.INPAINT_PRESERVE_KNOWN = False

    cfg.OUTPUT_FOLDER = str(output_root)
    cfg.TRAIN_PNG_FOLDER = str(output_root / "train" / "PNG")
    cfg.TRAIN_MAT_FOLDER = str(output_root / "train" / "MAT")
    cfg.GENERATE_PNG_FOLDER = str(output_root / "generate" / "PNG")
    cfg.GENERATE_MAT_FOLDER = str(output_root / "generate" / "MAT")
    cfg.ANALYSIS_PNG_FOLDER = str(output_root / "analysis" / "PNG")
    cfg.VALIDATION_PNG_FOLDER = str(output_root / "validation" / "PNG")
    cfg.VALIDATION_MAT_FOLDER = str(output_root / "validation" / "MAT")
    cfg.MODEL_PATH = str(output_root / "unet_ae_model.pt")

    return previous


def _restore_config(previous: dict) -> None:
    for key, value in previous.items():
        setattr(cfg, key, value)


def _run_trial(args: argparse.Namespace, trial: optuna.trial.Trial) -> tuple[float, float]:
    previous = _patch_config_for_trial(args, trial)
    train_module = None

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        train_module = importlib.import_module("train")
        train_module = importlib.reload(train_module)

        summary = train_module.main()

        if not isinstance(summary, dict) or summary.get("status") != "ok":
            trial.set_user_attr("summary", summary)
            return float("inf"), float("inf")

        mape = float(summary.get("mean_mape", float("inf")))
        vram_mb = float(summary.get("peak_vram_mb", float("inf")))
        if not math.isfinite(mape):
            mape = float("inf")
        if not math.isfinite(vram_mb):
            vram_mb = float("inf")

        trial.set_user_attr("summary", summary)
        trial.set_user_attr("results_folder", str(_trial_results_root(args, trial)))
        trial.set_user_attr("mape", mape)
        trial.set_user_attr("peak_vram_mb", vram_mb)
        return mape, vram_mb
    except RuntimeError as error:
        message = str(error).lower()
        trial.set_user_attr("error", str(error))
        if "out of memory" in message or ("cuda" in message and "memory" in message):
            trial.set_user_attr("oom", True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return float("inf"), float("inf")
        return float("inf"), float("inf")
    except Exception as error:
        trial.set_user_attr("error", str(error))
        return float("inf"), float("inf")
    finally:
        _restore_config(previous)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def main() -> None:
    args = parse_args()
    if args.n_jobs != 1:
        print("[i] n_jobs > 1 no es seguro con recarga de config global; se fuerza n_jobs=1.")
        args.n_jobs = 1

    results_root = Path(args.results_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    storage = args.storage.strip()
    if not storage:
        storage = f"sqlite:///{results_root / 'optuna_study.db'}"

    sampler = optuna.samplers.NSGAIISampler(seed=args.seed)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        directions=["minimize", "minimize"],
        load_if_exists=True,
        sampler=sampler,
    )

    timeout = args.timeout if args.timeout > 0 else None
    study.optimize(
        lambda trial: _run_trial(args, trial),
        n_trials=args.trials,
        timeout=timeout,
        n_jobs=args.n_jobs,
    )

    best_trials = study.best_trials
    if best_trials:
        best = sorted(best_trials, key=lambda t: (t.values[0], t.values[1]))[0]
        print("=" * 72)
        print(f"Best trial: {best.number}")
        print(f"Best values: MAPE={best.values[0]} | VRAM={best.values[1]} MB")
        print(f"Best params: {best.params}")
        print("=" * 72)
    else:
        print("No se encontraron trials válidos.")


if __name__ == "__main__":
    main()