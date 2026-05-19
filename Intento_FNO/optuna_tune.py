#!/usr/bin/env python3
"""
Optuna tuning script for Intento_FNO.

Usage examples:
  python Intento_FNO/optuna_tune.py --trials 20
  python Intento_FNO/optuna_tune.py --trials 10 --study-name fno_quick
"""
import argparse
import contextlib
import importlib
import importlib.util
import os
import re
import sys
from typing import Optional

import optuna


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


DEFAULT_TRIALS = 20
DEFAULT_STUDY_NAME = "optuna_fno_504x1920_v1"
DEFAULT_RESULTS_ROOT = "optuna_runs/optuna_fno_504x1920_v1"
DEFAULT_EPOCHS = 100
DEFAULT_TIMEOUT = 0
DEFAULT_SEED = 42
DEFAULT_N_JOBS = 1

SEARCH_SPACE = {
    "batch_size": [2, 4, 8],
    "learning_rate": (1e-5, 5e-4),
    "use_augmentation": [False, True],
    "fno_modes": [8, 12, 16, 20, 24, 32],
    "fno_width": [32, 64, 96, 128],
    "fno_depth": [3, 4, 5, 6],
    "fno_use_coords": [True, False],
    "fno_dropout": (0.0, 0.2),
}


def _ensure_local_module(module_name: str, module_dir: str):
    module_path = os.path.join(module_dir, f"{module_name}.py")
    module_path = os.path.abspath(module_path)
    if module_name in sys.modules:
        existing = sys.modules[module_name]
        existing_path = os.path.abspath(getattr(existing, "__file__", ""))
        if existing_path != module_path:
            del sys.modules[module_name]
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna tuning for Intento_FNO/train.py")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--study-name", type=str, default=DEFAULT_STUDY_NAME)
    parser.add_argument("--storage", type=str, default="")
    parser.add_argument("--results-root", type=str, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--input-dir", type=str, default="")
    parser.add_argument("--validation-dir", type=str, default="")
    return parser.parse_args()


def _trial_results_root(args: argparse.Namespace, trial: optuna.trial.Trial) -> str:
    return os.path.join(args.results_root, f"trial_{trial.number:04d}")


def _patch_config_for_trial(args: argparse.Namespace, trial: optuna.trial.Trial) -> dict:
    cfg = _ensure_local_module("config", _THIS_DIR)
    output_root = os.path.join(_REPO_ROOT, "Intento_FNO", _trial_results_root(args, trial), "output")

    fno_modes = trial.suggest_categorical("fno_modes", SEARCH_SPACE["fno_modes"])
    fno_width = trial.suggest_categorical("fno_width", SEARCH_SPACE["fno_width"])
    fno_depth = trial.suggest_categorical("fno_depth", SEARCH_SPACE["fno_depth"])
    fno_use_coords = trial.suggest_categorical("fno_use_coords", SEARCH_SPACE["fno_use_coords"])
    fno_dropout = trial.suggest_float(
        "fno_dropout",
        SEARCH_SPACE["fno_dropout"][0],
        SEARCH_SPACE["fno_dropout"][1],
    )
    batch_size = trial.suggest_categorical("batch_size", SEARCH_SPACE["batch_size"])
    learning_rate = trial.suggest_float(
        "learning_rate",
        SEARCH_SPACE["learning_rate"][0],
        SEARCH_SPACE["learning_rate"][1],
        log=True,
    )
    use_augmentation = trial.suggest_categorical("use_augmentation", SEARCH_SPACE["use_augmentation"])

    previous = {
        "BATCH_SIZE": cfg.BATCH_SIZE,
        "EPOCHS": cfg.EPOCHS,
        "LEARNING_RATE": cfg.LEARNING_RATE,
        "USE_AUGMENTATION": cfg.USE_AUGMENTATION,
        "FNO_MODES1": cfg.FNO_MODES1,
        "FNO_MODES2": cfg.FNO_MODES2,
        "FNO_WIDTH": cfg.FNO_WIDTH,
        "FNO_DEPTH": cfg.FNO_DEPTH,
        "FNO_USE_COORDS": cfg.FNO_USE_COORDS,
        "FNO_DROPOUT": cfg.FNO_DROPOUT,
        "DATA_FOLDER": cfg.DATA_FOLDER,
        "VALIDATION_FOLDER": cfg.VALIDATION_FOLDER,
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
    cfg.USE_AUGMENTATION = bool(use_augmentation)
    cfg.FNO_MODES1 = int(fno_modes)
    cfg.FNO_MODES2 = int(fno_modes)
    cfg.FNO_WIDTH = int(fno_width)
    cfg.FNO_DEPTH = int(fno_depth)
    cfg.FNO_USE_COORDS = bool(fno_use_coords)
    cfg.FNO_DROPOUT = float(fno_dropout)

    if args.input_dir:
        cfg.DATA_FOLDER = args.input_dir
        cfg.VALIDATION_FOLDER = args.validation_dir or os.path.join(args.input_dir, "validation")

    cfg.OUTPUT_FOLDER = output_root
    cfg.TRAIN_PNG_FOLDER = os.path.join(output_root, "train", "PNG")
    cfg.TRAIN_MAT_FOLDER = os.path.join(output_root, "train", "MAT")
    cfg.GENERATE_PNG_FOLDER = os.path.join(output_root, "generate", "PNG")
    cfg.GENERATE_MAT_FOLDER = os.path.join(output_root, "generate", "MAT")
    cfg.ANALYSIS_PNG_FOLDER = os.path.join(output_root, "analysis", "PNG")
    cfg.VALIDATION_PNG_FOLDER = os.path.join(output_root, "validation", "PNG")
    cfg.VALIDATION_MAT_FOLDER = os.path.join(output_root, "validation", "MAT")
    cfg.MODEL_PATH = os.path.join(output_root, "fno_model.pt")

    os.makedirs(cfg.TRAIN_PNG_FOLDER, exist_ok=True)
    os.makedirs(cfg.TRAIN_MAT_FOLDER, exist_ok=True)
    os.makedirs(cfg.VALIDATION_PNG_FOLDER, exist_ok=True)
    os.makedirs(cfg.VALIDATION_MAT_FOLDER, exist_ok=True)

    return previous


def _restore_config(previous: dict) -> None:
    cfg = _ensure_local_module("config", _THIS_DIR)
    for key, value in previous.items():
        setattr(cfg, key, value)


def _parse_mae_from_text(text: str) -> Optional[float]:
    match = re.search(r"Mean MAE\s*=\s*([0-9.+-eE]+)", text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except Exception:
        return None


def _run_trial(args: argparse.Namespace, trial: optuna.trial.Trial) -> float:
    previous = _patch_config_for_trial(args, trial)
    trial_root = os.path.join(_REPO_ROOT, "Intento_FNO", _trial_results_root(args, trial))
    trial_log = os.path.join(trial_root, "trial.log")

    try:
        if "train" in sys.modules:
            del sys.modules["train"]
        train = _ensure_local_module("train", _THIS_DIR)

        os.makedirs(trial_root, exist_ok=True)
        with open(trial_log, "w", buffering=1) as handle:
            handle.write(f"[optuna] trial {trial.number} start\n")
            with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
                train.main()
            handle.write(f"[optuna] trial {trial.number} finished\n")

        with open(trial_log, "r") as handle:
            log_text = handle.read()
        mae = _parse_mae_from_text(log_text)
        if mae is None:
            print("[optuna] could not parse MAE; returning large loss")
            return 1e6
        return mae
    except Exception as exc:
        with open(trial_log, "a") as handle:
            handle.write(f"[optuna] trial {trial.number} failed: {exc}\n")
        raise
    finally:
        _restore_config(previous)


def main() -> None:
    args = parse_args()

    if args.n_jobs != 1:
        print("[optuna] forcing n_jobs=1 to avoid stdout capture issues")
        args.n_jobs = 1

    results_root = os.path.join(_REPO_ROOT, "Intento_FNO", args.results_root)
    os.makedirs(results_root, exist_ok=True)

    storage = args.storage.strip()
    if not storage:
        storage = f"sqlite:///{os.path.join(results_root, 'optuna_study.db')}"

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
    )

    timeout = args.timeout if args.timeout > 0 else None
    study.optimize(
        lambda t: _run_trial(args, t),
        n_trials=args.trials,
        timeout=timeout,
        n_jobs=args.n_jobs,
    )

    print("Study complete. Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params:")
    for k, v in trial.params.items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
