#!/usr/bin/env python3
"""
Optuna tuning script for Intento_FNO.

ONLY optimizes:
    - epochs

Objective metric:
    - Mean MAPE

Usage examples:
    python optuna_epochs.py --trials 30
"""

import argparse
import contextlib
import gc
import importlib
import importlib.util
import os
import re
import sys
from typing import Optional

import optuna
import torch


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)

if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


DEFAULT_TRIALS = 50
DEFAULT_STUDY_NAME = "optuna_fno_epochs_mape_30"
DEFAULT_RESULTS_ROOT = "optuna_runs/optuna_fno_epochs_mape_30"
DEFAULT_TIMEOUT = 0
DEFAULT_SEED = 42
DEFAULT_N_JOBS = 1


SEARCH_SPACE = {
    "epochs": (50, 400),
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
    parser = argparse.ArgumentParser(
        description="Optuna tuning for epochs using Mean MAPE"
    )

    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--study-name", type=str, default=DEFAULT_STUDY_NAME)
    parser.add_argument("--storage", type=str, default="")
    parser.add_argument("--results-root", type=str, default=DEFAULT_RESULTS_ROOT)
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

    output_root = os.path.join(
        _REPO_ROOT,
        "Intento_FNO",
        _trial_results_root(args, trial),
        "output",
    )

    epochs = trial.suggest_int(
        "epochs",
        SEARCH_SPACE["epochs"][0],
        SEARCH_SPACE["epochs"][1],
    )

    previous = {
        "EPOCHS": cfg.EPOCHS,
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

    cfg.EPOCHS = int(epochs)

    if args.input_dir:
        cfg.DATA_FOLDER = args.input_dir
        cfg.VALIDATION_FOLDER = (
            args.validation_dir
            or os.path.join(args.input_dir, "validation")
        )

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


def _parse_mape_from_text(text: str) -> Optional[float]:
    match = re.search(r"Mean MAPE\s*=\s*([0-9.+-eE]+)", text)

    if not match:
        return None

    try:
        return float(match.group(1))
    except Exception:
        return None


def _is_oom_error(message: str) -> bool:
    message = message.lower()

    oom_patterns = [
        "out of memory",
        "cuda out of memory",
        "hip out of memory",
        "hsa memory",
        "rocblas",
        "miopenstatusallocfailed",
        "memory error",
        "cannot allocate memory",
    ]

    return any(pattern in message for pattern in oom_patterns)


def _cleanup_memory() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _run_trial(args: argparse.Namespace, trial: optuna.trial.Trial) -> float:
    previous = _patch_config_for_trial(args, trial)

    trial_root = os.path.join(
        _REPO_ROOT,
        "Intento_FNO",
        _trial_results_root(args, trial),
    )

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

        mape = _parse_mape_from_text(log_text)

        if mape is None:
            print("[optuna] could not parse MAPE; returning large loss")
            return 1e6

        print(
            f"[trial {trial.number:04d}] "
            f"EPOCHS={trial.params['epochs']} "
            f"MAPE={mape:.6f}"
        )

        return mape

    except RuntimeError as exc:
        error_message = str(exc)

        with open(trial_log, "a") as handle:
            handle.write(
                f"[optuna] trial {trial.number} runtime error:\n{error_message}\n"
            )

        if _is_oom_error(error_message):
            print(f"[trial {trial.number:04d}] OOM detected")

            trial.set_user_attr("oom", True)
            trial.set_user_attr("error", error_message)

            _cleanup_memory()

            return float("inf")

        trial.set_user_attr("error", error_message)

        print(f"[trial {trial.number:04d}] RuntimeError: {error_message}")

        _cleanup_memory()

        return float("inf")

    except Exception as exc:
        error_message = str(exc)

        with open(trial_log, "a") as handle:
            handle.write(
                f"[optuna] trial {trial.number} failed:\n{error_message}\n"
            )

        trial.set_user_attr("error", error_message)

        print(f"[trial {trial.number:04d}] Error: {error_message}")

        _cleanup_memory()

        return float("inf")

    finally:
        _restore_config(previous)
        _cleanup_memory()


def main() -> None:
    args = parse_args()

    if args.n_jobs != 1:
        print("[optuna] forcing n_jobs=1 to avoid stdout capture issues")
        args.n_jobs = 1

    results_root = os.path.join(
        _REPO_ROOT,
        "Intento_FNO",
        args.results_root,
    )

    os.makedirs(results_root, exist_ok=True)

    storage = args.storage.strip()

    if not storage:
        storage = (
            f"sqlite:///{os.path.join(results_root, 'optuna_study.db')}"
        )

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

    print("=" * 72)
    print("Study complete. Best trial:")
    print("=" * 72)

    trial = study.best_trial

    print(f"  Trial: {trial.number}")
    print(f"  Best MAPE: {trial.value}")

    print("  Params:")

    for k, v in trial.params.items():
        print(f"    {k}: {v}")

    print("=" * 72)


if __name__ == "__main__":
    main()