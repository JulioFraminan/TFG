#!/usr/bin/env python3
"""Run Optuna tuning across multiple project folders with a shared input path."""
import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Optuna tuning across all experiments.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(REPO_ROOT / "input"),
        help="Path to input directory (default: workspace input/)",
    )
    parser.add_argument(
        "--validation-dir",
        type=str,
        default=str(REPO_ROOT / "input" / "validation"),
        help="Path to validation directory (default: input/validation)",
    )
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=100,
        help="Number of jobs to run in parallel (default: 100)",
    )
    parser.add_argument("--only", type=str, default="")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def _normalize_only(value: str) -> set[str]:
    if not value:
        return set()
    return {chunk.strip() for chunk in value.split(",") if chunk.strip()}


def main() -> int:
    args = parse_args()
    input_dir = os.path.abspath(args.input_dir)
    validation_dir = os.path.abspath(args.validation_dir) if args.validation_dir else ""
    only = _normalize_only(args.only)

    jobs = [
        ("Intento_FNO", REPO_ROOT / "Intento_FNO" / "optuna_tune.py"),
        ("pinn_modular", REPO_ROOT / "pinn_modular" / "optuna_tune.py"),
        ("Transformers_2", REPO_ROOT / "Transformers_2" / "optuna_tune.py"),
        ("unet_ae_modular_inpainting", REPO_ROOT / "unet_ae_modular_inpainting" / "optuna_tune.py"),
        ("unet_hibrido", REPO_ROOT / "unet_hibrido" / "optuna_tune.py"),
    ]

    for name, script_path in jobs:
        if only and name not in only:
            continue
        if not script_path.exists():
            msg = f"[skip] {name}: missing script {script_path}"
            if args.continue_on_error:
                print(msg)
                continue
            print(msg)
            return 1

        cmd = [sys.executable, str(script_path), "--input-dir", input_dir]
        if validation_dir:
            cmd += ["--validation-dir", validation_dir]
        if args.trials is not None:
            cmd += ["--trials", str(args.trials)]
        if args.n_jobs is not None:
            cmd += ["--n-jobs", str(args.n_jobs)]

        print("=" * 72)
        print(f"[run] {name}")
        print("Command:", " ".join(cmd))
        print("=" * 72)

        try:
            subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
        except subprocess.CalledProcessError as exc:
            print(f"[error] {name} failed with exit code {exc.returncode}")
            if not args.continue_on_error:
                return exc.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
