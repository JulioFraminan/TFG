import argparse
from pathlib import Path
from typing import Optional

import optuna
import torch

from train_intento_mat import parse_args as train_parse_args
from train_intento_mat import run_training
from train_intento_mat import _explicit_cli_destinations


# Base training config is loaded by train_intento_mat.py defaults.
# Values in SEARCH_SPACE override those defaults via CLI args per trial.

# Edit these defaults instead of passing CLI args.
# - DEFAULT_TRIALS: how many Optuna trials to run.
# - DEFAULT_METRIC: which validation metric to minimize.
#   Options: "mean_mae", "mean_rmse", "mean_max_error", "mean_mape".
# - DEFAULT_STUDY_NAME: Optuna study name (stored in the sqlite db).
#   If you change SEARCH_SPACE, use a new study name (or delete the old DB)
#   to avoid "CategoricalDistribution does not support dynamic value space".
# - DEFAULT_RESULTS_ROOT: folder where trial runs and study db are stored.
# - DEFAULT_TRAIN_NUM_STEPS: override training steps for faster trials (-1 keeps config).
# - DEFAULT_TIMEOUT: max seconds for the full study (0 means no timeout).
# - DEFAULT_SEED: seed passed to train_intento_mat.py.
# - DEFAULT_N_JOBS: parallel Optuna workers (use 1 if GPU is shared).
# - DEFAULT_PRUNER: pruning strategy ("median" or "none").
# - DEFAULT_PRUNER_WARMUP_TRIALS: number of trials before pruning starts.
# - DEFAULT_PRUNER_WARMUP_STEPS: number of reported steps before pruning starts.
DEFAULT_TRIALS = 30
DEFAULT_METRIC = "mean_mae"
DEFAULT_STUDY_NAME = "optuna_intento_1920x480_v2"
DEFAULT_RESULTS_ROOT = "results/intento_mat/optuna_trials_1920x480_v2"
DEFAULT_TRAIN_NUM_STEPS = 3000
DEFAULT_TIMEOUT = 0
DEFAULT_SEED = 42
DEFAULT_N_JOBS = 1
DEFAULT_PRUNER = "median"
DEFAULT_PRUNER_WARMUP_TRIALS = 3
DEFAULT_PRUNER_WARMUP_STEPS = 1

SEARCH_SPACE = {
    # Model and optimizer knobs to explore. Keep this list aligned with train_intento_mat.py flags.
    # - dit_variant options: "DiT-S/8".
    "dit_variant": ["DiT-S/8"],
    # - dit_attn_type options: "linear".
    "dit_attn_type": ["linear"],
    # - dit_mlp_ratio range: (min, max).
    "dit_mlp_ratio": (2.0, 4.5),
    # - dit_qk_norm options: True, False.
    "dit_qk_norm": [True, False],
    # - dit_class_dropout range: (min, max).
    "dit_class_dropout": (0.0, 0.30),
    # - train_lr range: (min, max).
    "train_lr": (1e-5, 1e-3),
    # - ema_decay range: (min, max).
    "ema_decay": (0.990, 0.9995),
    # - train_batch_size options: 2, 4, 8.
    "train_batch_size": [2, 4, 8],
    # - gradient_accumulate_every options: 1, 2, 4.
    "gradient_accumulate_every": [1, 2, 4],
    # - objective options: "pred_v", "pred_noise".
    "objective": ["pred_v", "pred_noise"],
    # - sampling_timesteps range: (min, max, step).
    "sampling_timesteps": (100, 600, 50),
    # - validation_cond_scale range: (min, max).
    "validation_cond_scale": (1.0, 4.0),
    # - min_snr_loss_weight options: True, False.
    "min_snr_loss_weight": [True, False],
    # - min_snr_gamma range: (min, max).
    "min_snr_gamma": (2.0, 10.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna tuning for Transformers_2 train_intento_mat.py"
    )
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--study-name", type=str, default=DEFAULT_STUDY_NAME)
    parser.add_argument("--storage", type=str, default="")
    parser.add_argument("--results-root", type=str, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--train-num-steps", type=int, default=DEFAULT_TRAIN_NUM_STEPS)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument(
        "--metric",
        type=str,
        choices=["mean_mae", "mean_rmse", "mean_max_error", "mean_mape"],
        default=DEFAULT_METRIC,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    return parser.parse_args()


def _build_train_argv(args: argparse.Namespace, trial: optuna.trial.Trial, results_root: Path) -> list[str]:
    train_lr = trial.suggest_float("train_lr", SEARCH_SPACE["train_lr"][0], SEARCH_SPACE["train_lr"][1], log=True)
    dit_class_dropout = trial.suggest_float(
        "dit_class_dropout", SEARCH_SPACE["dit_class_dropout"][0], SEARCH_SPACE["dit_class_dropout"][1]
    )
    objective = trial.suggest_categorical("objective", SEARCH_SPACE["objective"])
    sampling_timesteps = trial.suggest_int(
        "sampling_timesteps",
        SEARCH_SPACE["sampling_timesteps"][0],
        SEARCH_SPACE["sampling_timesteps"][1],
        step=SEARCH_SPACE["sampling_timesteps"][2],
    )
    validation_cond_scale = trial.suggest_float(
        "validation_cond_scale",
        SEARCH_SPACE["validation_cond_scale"][0],
        SEARCH_SPACE["validation_cond_scale"][1],
    )
    min_snr_loss_weight = trial.suggest_categorical("min_snr_loss_weight", SEARCH_SPACE["min_snr_loss_weight"])
    dit_variant = trial.suggest_categorical("dit_variant", SEARCH_SPACE["dit_variant"])
    dit_attn_type = trial.suggest_categorical("dit_attn_type", SEARCH_SPACE["dit_attn_type"])
    dit_mlp_ratio = trial.suggest_float(
        "dit_mlp_ratio", SEARCH_SPACE["dit_mlp_ratio"][0], SEARCH_SPACE["dit_mlp_ratio"][1]
    )
    dit_qk_norm = trial.suggest_categorical("dit_qk_norm", SEARCH_SPACE["dit_qk_norm"])
    ema_decay = trial.suggest_float("ema_decay", SEARCH_SPACE["ema_decay"][0], SEARCH_SPACE["ema_decay"][1])
    train_batch_size = trial.suggest_categorical("train_batch_size", SEARCH_SPACE["train_batch_size"])
    gradient_accumulate_every = trial.suggest_categorical(
        "gradient_accumulate_every", SEARCH_SPACE["gradient_accumulate_every"]
    )
    min_snr_gamma = trial.suggest_float("min_snr_gamma", SEARCH_SPACE["min_snr_gamma"][0], SEARCH_SPACE["min_snr_gamma"][1])

    trial_dir = results_root / f"trial_{trial.number:04d}"
    argv: list[str] = [
        "--results-folder",
        str(trial_dir),
        "--results-layout",
        "legacy",
        "--no-resume-if-compatible",
        "--no-milestone-validation-png",
        "--train-lr",
        f"{train_lr}",
        "--train-batch-size",
        f"{train_batch_size}",
        "--ema-decay",
        f"{ema_decay}",
        "--gradient-accumulate-every",
        f"{gradient_accumulate_every}",
        "--dit-class-dropout",
        f"{dit_class_dropout}",
        "--dit-variant",
        dit_variant,
        "--dit-attn-type",
        dit_attn_type,
        "--dit-mlp-ratio",
        f"{dit_mlp_ratio}",
        "--objective",
        objective,
        "--min-snr-gamma",
        f"{min_snr_gamma}",
        "--sampling-timesteps",
        f"{sampling_timesteps}",
        "--validation-cond-scale",
        f"{validation_cond_scale}",
        "--seed",
        str(args.seed),
    ]

    if dit_qk_norm:
        argv.append("--dit-qk-norm")

    if args.train_num_steps > 0:
        argv += ["--train-num-steps", str(args.train_num_steps)]

    if min_snr_loss_weight:
        argv.append("--min-snr-loss-weight")
    else:
        argv.append("--no-min-snr-loss-weight")

    return argv


def _objective_factory(
    args: argparse.Namespace,
    results_root: Path,
    metric: str,
):
    def _objective(trial: optuna.trial.Trial) -> float:
        try:
            def _report_validation(step: int, summary: dict) -> None:
                if not isinstance(summary, dict) or summary.get("status") != "ok":
                    return
                value = summary.get(metric)
                if value is None:
                    return
                safe_step = int(step) if int(step) >= 0 else 0
                trial.report(float(value), safe_step)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            train_argv = _build_train_argv(args, trial, results_root)
            train_args = train_parse_args(train_argv)
            explicit = _explicit_cli_destinations(["optuna"] + train_argv)
            summary = run_training(
                train_args,
                explicit_cli_dests=explicit,
                on_validation=_report_validation,
            )

            trial.set_user_attr("results_folder", str(results_root / f"trial_{trial.number:04d}"))
            trial.set_user_attr("summary", summary)

            if not isinstance(summary, dict) or summary.get("status") != "ok":
                print(
                    f"[trial {trial.number:04d}] Invalid summary status: {summary}"
                )
                return float("inf")

            value = summary.get(metric)
            return float(value) if value is not None else float("inf")
        except optuna.exceptions.TrialPruned:
            print(f"[trial {trial.number:04d}] Pruned")
            raise
        except RuntimeError as error:
            message = str(error).lower()
            if "out of memory" in message or "cuda" in message and "memory" in message:
                trial.set_user_attr("oom", True)
                trial.set_user_attr("error", str(error))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"[trial {trial.number:04d}] OOM: {error}")
                return float("inf")
            trial.set_user_attr("error", str(error))
            print(f"[trial {trial.number:04d}] RuntimeError: {error}")
            return float("inf")
        except Exception as error:
            trial.set_user_attr("error", str(error))
            print(f"[trial {trial.number:04d}] Error: {error}")
            return float("inf")

    return _objective


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    storage = args.storage.strip()
    if not storage:
        storage = f"sqlite:///{results_root / 'optuna_study.db'}"

    pruner = None
    if DEFAULT_PRUNER == "median":
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=DEFAULT_PRUNER_WARMUP_TRIALS,
            n_warmup_steps=DEFAULT_PRUNER_WARMUP_STEPS,
        )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        pruner=pruner,
    )

    timeout = args.timeout if args.timeout > 0 else None
    study.optimize(
        _objective_factory(args, results_root, args.metric),
        n_trials=args.trials,
        timeout=timeout,
        n_jobs=args.n_jobs,
    )

    best = study.best_trial
    print("=" * 72)
    print(f"Best trial: {best.number}")
    print(f"Best value ({args.metric}): {best.value}")
    print(f"Best params: {best.params}")
    print("=" * 72)


if __name__ == "__main__":
    main()
