#!/usr/bin/env python3
"""
Optuna tuning script for the PINN in `pinn_modular`.

Usage examples:
  python -m pinn_modular.optuna_tune --trials 12
  python -m pinn_modular.optuna_tune --trials 8 --study-name quick_test

The script overrides values in `pinn_modular.config` before importing
`pinn_modular.train` so the training run uses the trial parameters.
It captures stdout from `train.main()` and parses the printed validation
summary to return the mean MAE as the objective (minimize).
"""
import os
import sys
import re
import io
import argparse
import importlib
import contextlib

import optuna
# Ensure the repository root is on sys.path so `importlib.import_module("pinn_modular...")`
# works even when the script is executed from a different working directory or when
# Optuna spawns worker processes.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


DEFAULT_TRIALS = 12
DEFAULT_N_JOBS = 1


def set_trial_config(cfg, trial, run_dir):
    # Model / optimizer
    cfg.LEARNING_RATE = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    cfg.PINN_HIDDEN_DIM = int(trial.suggest_categorical("hidden_dim", [64, 128, 256]))
    cfg.PINN_NUM_LAYERS = int(trial.suggest_int("num_layers", 2, 8))
    cfg.PINN_DROPOUT = float(trial.suggest_float("dropout", 0.0, 0.5))

    # Data / collocation sizes
    cfg.POINTS_PER_ROI = int(trial.suggest_int("points_per_roi", 2000, 20000, step=2000))
    cfg.PHYSICS_BATCH_SIZE = int(trial.suggest_int("physics_batch", 1024, 8192, step=1024))

    # Loss weights
    cfg.DATA_WEIGHT = 1.0
    cfg.PHYSICS_WEIGHT = float(trial.suggest_loguniform("physics_weight", 1e-4, 1.0))
    cfg.INTERFACE_WEIGHT = float(trial.suggest_loguniform("interface_weight", 1e-4, 1.0))

    # Fix epochs for comparability across trials
    cfg.EPOCHS = 100

    # Use per-trial output paths to avoid collisions
    cfg.OUTPUT_FOLDER = os.path.join(cfg.BASE_DIR, "optuna_runs", run_dir)
    cfg.TRAIN_PNG_FOLDER = os.path.join(cfg.OUTPUT_FOLDER, "train", "PNG")
    cfg.TRAIN_MAT_FOLDER = os.path.join(cfg.OUTPUT_FOLDER, "train", "MAT")
    cfg.VALIDATION_PNG_FOLDER = os.path.join(cfg.OUTPUT_FOLDER, "validation", "PNG")
    cfg.VALIDATION_MAT_FOLDER = os.path.join(cfg.OUTPUT_FOLDER, "validation", "MAT")
    cfg.MODEL_PATH = os.path.join(cfg.OUTPUT_FOLDER, "pinn_model.pt")

    # Ensure output dirs exist
    os.makedirs(cfg.TRAIN_PNG_FOLDER, exist_ok=True)
    os.makedirs(cfg.TRAIN_MAT_FOLDER, exist_ok=True)
    os.makedirs(cfg.VALIDATION_PNG_FOLDER, exist_ok=True)
    os.makedirs(cfg.VALIDATION_MAT_FOLDER, exist_ok=True)


def parse_mae_from_output(text):
    # Search for the printed validation summary line: "MAE mean  = <value>"
    m = re.search(r"MAE mean\s*=\s*([0-9.+-eE]+)", text)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


def objective(trial, study_name):
    # Import config and set trial parameters
    cfg = importlib.import_module("pinn_modular.config")
    run_dir = f"{study_name}_trial_{trial.number:03d}"
    set_trial_config(cfg, trial, run_dir)

    # Ensure fresh import of train so it re-reads values from config (train uses
    # `from config import ...` at import time). Remove cached module if present.
    if "pinn_modular.train" in sys.modules:
        del sys.modules["pinn_modular.train"]
    train = importlib.import_module("pinn_modular.train")

    # Capture stdout/stderr to a per-trial log file so we can inspect hanging
    trial_log = os.path.join(cfg.OUTPUT_FOLDER, "trial.log")
    os.makedirs(os.path.dirname(trial_log), exist_ok=True)
    try:
        with open(trial_log, "w", buffering=1) as f:  # line-buffered
            f.write(f"[optuna] trial {trial.number} start\n")
            f.flush()
            with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                train.main()
            f.write(f"[optuna] trial {trial.number} finished\n")
            f.flush()
    except Exception as e:
        # Write traceback to the trial log for easier debugging
        import traceback
        with open(trial_log, "a") as f:
            f.write(f"[optuna] trial {trial.number} failed with exception: {e}\n")
            traceback.print_exc(file=f)
            f.flush()
        # Re-raise so Optuna records the failure
        raise

    # Read the trial log to parse the MAE
    with open(trial_log, "r") as f:
        out = f.read()
    mae = parse_mae_from_output(out)
    if mae is None:
        # If parsing failed, try to warn and return a large value
        print("[optuna] could not parse MAE from training output; returning large loss")
        return 1e6
    return mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-name", type=str, default="pinn_optuna")
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.INFO)
    study = optuna.create_study(direction="minimize", study_name=args.study_name)

    # Wrap objective to pass study name for directory naming
    func = lambda t: objective(t, args.study_name)

    study.optimize(func, n_trials=DEFAULT_TRIALS, n_jobs=DEFAULT_N_JOBS)

    print("Study complete. Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params:")
    for k, v in trial.params.items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
