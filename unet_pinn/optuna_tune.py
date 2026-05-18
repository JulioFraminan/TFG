import argparse
import builtins
import gc
import math
import importlib
import re
import sys
from pathlib import Path

import optuna
import torch


PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config as cfg  # noqa: E402


DEFAULT_TRIALS = 30
DEFAULT_STUDY_NAME = "optuna_unet_pinn_v1"
DEFAULT_RESULTS_ROOT = "results/optuna_unet_pinn_v1"
DEFAULT_EPOCHS = 100
DEFAULT_TIMEOUT = 0
DEFAULT_SEED = 42
DEFAULT_N_JOBS = 1
DEFAULT_PRUNE_WARMUP = 50

SEARCH_SPACE = {
    "batch_size": [2, 4, 8],
    "learning_rate": (5e-6, 1e-3),
    "weight_decay": (1e-7, 1e-3),
    "model_dropout": (0.0, 0.25),
    "data_weight": (0.1, 2.0),
    "physics_weight": (1e-3, 1.0),
    "interface_weight": (0.0, 0.5),
    "physics_batch_size": [0, 4096, 16384],
    "interface_batch_size": [0, 1024, 2048],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna tuning for unet_pinn/train.py (PINN + UNet, inpainting disabled)"
    )
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--study-name", type=str, default=DEFAULT_STUDY_NAME)
    parser.add_argument("--storage", type=str, default="")
    parser.add_argument("--results-root", type=str, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--prune-warmup", type=int, default=DEFAULT_PRUNE_WARMUP)
    return parser.parse_args()


def _trial_results_root(args: argparse.Namespace, trial: optuna.trial.Trial) -> Path:
    return Path(args.results_root).resolve() / _trial_name(trial)


def _slug_float(value: float) -> str:
    txt = f"{float(value):.2f}"
    return txt.replace("-", "m").replace(".", "p")


def _trial_name(trial: optuna.trial.Trial) -> str:
    corner_x, corner_z = cfg.ROI_CORNER
    corner_tag = f"corner{_slug_float(corner_x)}_{_slug_float(corner_z)}"
    roi_tag = f"{cfg.ROI_MODE}_roi{cfg.ROI_HEIGHT}x{cfg.ROI_WIDTH}_{corner_tag}"
    return f"trial_{trial.number:04d}_{roi_tag}"


def _patch_config_for_trial(
    args: argparse.Namespace,
    trial: optuna.trial.Trial,
) -> tuple[dict, Path]:
    output_root = _trial_results_root(args, trial) / "output"

    batch_size = trial.suggest_categorical("batch_size", SEARCH_SPACE["batch_size"])
    learning_rate = trial.suggest_float(
        "learning_rate",
        SEARCH_SPACE["learning_rate"][0],
        SEARCH_SPACE["learning_rate"][1],
        log=True,
    )
    data_weight = trial.suggest_float(
        "data_weight",
        SEARCH_SPACE["data_weight"][0],
        SEARCH_SPACE["data_weight"][1],
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
    physics_weight = trial.suggest_float(
        "physics_weight",
        SEARCH_SPACE["physics_weight"][0],
        SEARCH_SPACE["physics_weight"][1],
        log=True,
    )
    interface_weight = trial.suggest_float(
        "interface_weight",
        SEARCH_SPACE["interface_weight"][0],
        SEARCH_SPACE["interface_weight"][1],
    )
    physics_batch_size = trial.suggest_categorical(
        "physics_batch_size",
        SEARCH_SPACE["physics_batch_size"],
    )
    interface_batch_size = trial.suggest_categorical(
        "interface_batch_size",
        SEARCH_SPACE["interface_batch_size"],
    )

    previous = {
        "BATCH_SIZE": cfg.BATCH_SIZE,
        "EPOCHS": cfg.EPOCHS,
        "LEARNING_RATE": cfg.LEARNING_RATE,
        "WEIGHT_DECAY": cfg.WEIGHT_DECAY,
        "MODEL_DROPOUT": cfg.MODEL_DROPOUT,
        "USE_AUGMENTATION": cfg.USE_AUGMENTATION,
        "DATA_WEIGHT": cfg.DATA_WEIGHT,
        "PHYSICS_WEIGHT": cfg.PHYSICS_WEIGHT,
        "INTERFACE_WEIGHT": cfg.INTERFACE_WEIGHT,
        "PHYSICS_BATCH_SIZE": cfg.PHYSICS_BATCH_SIZE,
        "INTERFACE_BATCH_SIZE": cfg.INTERFACE_BATCH_SIZE,
        "SEED": cfg.SEED,
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
    cfg.USE_AUGMENTATION = False
    cfg.DATA_WEIGHT = float(data_weight)
    cfg.PHYSICS_WEIGHT = float(physics_weight)
    cfg.INTERFACE_WEIGHT = float(interface_weight)
    cfg.PHYSICS_BATCH_SIZE = int(physics_batch_size)
    cfg.INTERFACE_BATCH_SIZE = int(interface_batch_size)
    cfg.SEED = int(args.seed)

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

    return previous, output_root


def _restore_config(previous: dict) -> None:
    for key, value in previous.items():
        setattr(cfg, key, value)


def _run_trial(args: argparse.Namespace, trial: optuna.trial.Trial) -> float:
    optuna_args = args
    previous, output_root = _patch_config_for_trial(optuna_args, trial)
    train_module = None
    epoch_re = re.compile(r"Epoca\s+(\d+)/(\d+)\s*\|\s*loss=([0-9.]+)")
    prune_flag = {"raised": False}
    original_print = builtins.print

    def _print_and_report(*print_args, **kwargs):
        original_print(*print_args, **kwargs)
        sep = kwargs.get("sep", " ")
        message = sep.join(str(arg) for arg in print_args)
        match = epoch_re.search(message)
        if not match:
            return
        epoch = int(match.group(1))
        loss = float(match.group(3))
        trial.report(loss, step=epoch)
        if epoch >= optuna_args.prune_warmup and trial.should_prune():
            prune_flag["raised"] = True
            raise optuna.TrialPruned(f"Pruned at epoch {epoch} with loss={loss}")

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        builtins.print = _print_and_report

        print("=" * 72)
        print(f"[trial {trial.number}] name: {_trial_name(trial)}")
        print(f"[trial {trial.number}] params: {trial.params}")

        train_module = importlib.import_module("train")
        train_module = importlib.reload(train_module)

        summary = train_module.main()

        if not isinstance(summary, dict) or summary.get("status") != "ok":
            trial.set_user_attr("summary", summary)
            original_print(
                f"[trial {trial.number}] invalid summary: {summary}"
            )
            return float("inf")

        mae = float(summary.get("mean_mae", float("inf")))
        rmse = float(summary.get("mean_rmse", float("inf")))
        mape = float(summary.get("mean_mape", float("inf")))
        max_err = float(summary.get("mean_max_error", float("inf")))
        vram_mb = float(summary.get("peak_vram_mb", float("inf")))

        if not math.isfinite(mae):
            mae = float("inf")
        if not math.isfinite(rmse):
            rmse = float("inf")
        if not math.isfinite(mape):
            mape = float("inf")
        if not math.isfinite(max_err):
            max_err = float("inf")
        if not math.isfinite(vram_mb):
            vram_mb = float("inf")

        trial.set_user_attr("summary", summary)
        trial.set_user_attr("results_folder", str(_trial_results_root(args, trial)))
        trial.set_user_attr("mean_mae", mae)
        trial.set_user_attr("mean_rmse", rmse)
        trial.set_user_attr("mean_mape", mape)
        trial.set_user_attr("mean_max_error", max_err)
        trial.set_user_attr("peak_vram_mb", vram_mb)

        print(
            f"[trial {trial.number}] results: "
            f"MAE={mae:.3f} RMSE={rmse:.3f} MAPE={mape:.3f}% "
            f"MAX={max_err:.3f} VRAM={vram_mb:.1f}MB"
        )
        return mape
    except optuna.TrialPruned:
        trial.set_user_attr("pruned", True)
        raise
    except RuntimeError as error:
        message = str(error).lower()
        trial.set_user_attr("error", str(error))
        if "out of memory" in message or ("cuda" in message and "memory" in message):
            trial.set_user_attr("oom", True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            original_print(f"[trial {trial.number}] OOM: {error}")
            return float("inf")
        original_print(f"[trial {trial.number}] runtime error: {error}")
        return float("inf")
    except Exception as error:
        trial.set_user_attr("error", str(error))
        original_print(f"[trial {trial.number}] error: {error}")
        return float("inf")
    finally:
        builtins.print = original_print
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

    print(f"[i] default seed: {DEFAULT_SEED} | using seed: {args.seed}")

    optuna.logging.set_verbosity(optuna.logging.INFO)

    storage = args.storage.strip()
    if not storage:
        storage = f"sqlite:///{results_root / 'optuna_study.db'}"

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=args.prune_warmup)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    timeout = args.timeout if args.timeout > 0 else None
    study.optimize(
        lambda trial: _run_trial(args, trial),
        n_trials=args.trials,
        timeout=timeout,
        n_jobs=args.n_jobs,
    )

    best = study.best_trial
    if best is not None:
        print("=" * 72)
        print(f"Best trial: {best.number}")
        print(f"Best value (MAPE): {best.value}")
        print(f"Best params: {best.params}")
        print("=" * 72)
    else:
        print("No se encontraron trials validos.")


if __name__ == "__main__":
    main()