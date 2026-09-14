#!/usr/bin/env python3
"""
Evaluador de Métricas 3D para Mallas Estructuradas VTS (Transmission Loss)
"""

import os
import glob
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

# =============================================================================
# CONFIGURACIÓN Y RUTAS
# =============================================================================
VTS_FOLDER = r"/home/j.framinan/TFG_repo/input/VTS"
GT_FILENAME = "ground_truth_case30.vts"

SCALAR_NAME = "tl"
THRESHOLDS = [-50.0]  # Umbrales en dB para topología 3D

# Guardar informe en un archivo .txt en la carpeta VTS
SAVE_TXT_REPORT = True
OUTPUT_TXT_PATH = os.path.join(VTS_FOLDER, "metricas_evaluacion_3d_all_vts.txt")
# =============================================================================


class VTS3DEvaluator:
    def __init__(self, path_pred: str, path_gt: str, scalar_name: str = "tl"):
        if not os.path.exists(path_pred):
            raise FileNotFoundError(f"No se encontró el archivo PRED: {path_pred}")
        if not os.path.exists(path_gt):
            raise FileNotFoundError(f"No se encontró el archivo GT: {path_gt}")

        print(f"📦 Cargando GT:   {path_gt}")
        self.grid_gt = pv.read(path_gt)
        print(f"📦 Cargando PRED: {path_pred}")
        self.grid_pred = pv.read(path_pred)

        self.scalar_name = scalar_name

        if scalar_name not in self.grid_gt.array_names:
            raise KeyError(f"El escalar '{scalar_name}' no existe en la malla GT.")
        if scalar_name not in self.grid_pred.array_names:
            raise KeyError(f"El escalar '{scalar_name}' no existe en la malla PRED.")

        if self.grid_gt.dimensions != self.grid_pred.dimensions:
            raise ValueError(
                f"Dimensiones incompatibles: GT {self.grid_gt.dimensions} vs PRED {self.grid_pred.dimensions}"
            )

        self.dims = self.grid_gt.dimensions  # (nx, ny, nz)
        self.nx, self.ny, self.nz = self.dims

        self.tl_gt_3d = self.grid_gt[scalar_name].reshape(self.dims, order="F")
        self.tl_pred_3d = self.grid_pred[scalar_name].reshape(self.dims, order="F")
        self.pts_3d = self.grid_gt.points.reshape((*self.dims, 3), order="F")

    def compute_volumetric_metrics(self) -> dict:
        gt_cells = self.grid_gt.compute_cell_sizes()
        cell_volumes = gt_cells.cell_data["Volume"]

        gt_cell_data = self.grid_gt.point_data_to_cell_data()[self.scalar_name]
        pred_cell_data = self.grid_pred.point_data_to_cell_data()[self.scalar_name]

        diff = pred_cell_data - gt_cell_data
        total_vol = np.sum(cell_volumes)

        mae_v = np.sum(np.abs(diff) * cell_volumes) / total_vol
        mse_v = np.sum((diff**2) * cell_volumes) / total_vol
        rmse_v = np.sqrt(mse_v)

        return {
            "Total_Volume_m3": float(total_vol),
            "MAE_Volumetric": float(mae_v),
            "RMSE_Volumetric": float(rmse_v),
        }

    def compute_azimuthal_gradient_and_tv(self) -> dict:
        X = self.pts_3d[:, :, :, 0]
        Y = self.pts_3d[:, :, :, 1]
        Theta = np.arctan2(Y, X)

        dtheta = np.mean(np.diff(Theta[0, 0, :]))
        if abs(dtheta) < 1e-6:
            dtheta = 1.0

        grad_theta_gt = np.gradient(self.tl_gt_3d, dtheta, axis=2)
        grad_theta_pred = np.gradient(self.tl_pred_3d, dtheta, axis=2)

        mae_grad_theta = np.mean(np.abs(grad_theta_pred - grad_theta_gt))

        grad_gt_3d = np.stack(np.gradient(self.tl_gt_3d), axis=-1)
        grad_pred_3d = np.stack(np.gradient(self.tl_pred_3d), axis=-1)

        norm_grad_gt = np.linalg.norm(grad_gt_3d, axis=-1)
        norm_grad_pred = np.linalg.norm(grad_pred_3d, axis=-1)

        tv_3d_gt = np.sum(norm_grad_gt)
        tv_3d_pred = np.sum(norm_grad_pred)
        delta_tv_3d = float(np.abs(tv_3d_pred - tv_3d_gt) / (tv_3d_gt + 1e-8))

        return {
            "MAE_Grad_Theta": float(mae_grad_theta),
            "TV3D_GT": float(tv_3d_gt),
            "TV3D_PRED": float(tv_3d_pred),
            "Delta_TV3D_Rel": delta_tv_3d,
        }

    def compute_isosurface_topology(self, threshold: float) -> dict:
        mask_gt = self.tl_gt_3d <= threshold
        mask_pred = self.tl_pred_3d <= threshold

        intersection = np.logical_and(mask_gt, mask_pred).sum()
        union = np.logical_or(mask_gt, mask_pred).sum()

        gt_voxels = int(mask_gt.sum())
        pred_voxels = int(mask_pred.sum())

        iou_3d = float(intersection / (union + 1e-8))
        dice_3d = float((2 * intersection) / (gt_voxels + pred_voxels + 1e-8))

        iso_gt = self.grid_gt.contour(isosurfaces=[threshold], scalars=self.scalar_name)
        iso_pred = self.grid_pred.contour(isosurfaces=[threshold], scalars=self.scalar_name)

        if iso_gt.n_points == 0 or iso_pred.n_points == 0:
            return {
                f"IoU_3D_at_{threshold:.0f}dB": iou_3d,
                f"Dice_3D_at_{threshold:.0f}dB": dice_3d,
                f"HD95_3D_at_{threshold:.0f}dB_m": np.nan,
                f"Chamfer_3D_at_{threshold:.0f}dB_m": np.nan,
                f"GT_Voxels_at_{threshold:.0f}dB": gt_voxels,
                f"PRED_Voxels_at_{threshold:.0f}dB": pred_voxels,
            }

        tree_gt = cKDTree(iso_gt.points)
        tree_pred = cKDTree(iso_pred.points)

        d_pred_to_gt, _ = tree_gt.query(iso_pred.points)
        d_gt_to_pred, _ = tree_pred.query(iso_gt.points)

        hd95_dist = max(np.percentile(d_pred_to_gt, 95), np.percentile(d_gt_to_pred, 95))
        chamfer_dist = 0.5 * (np.mean(d_pred_to_gt) + np.mean(d_gt_to_pred))

        return {
            f"IoU_3D_at_{threshold:.0f}dB": iou_3d,
            f"Dice_3D_at_{threshold:.0f}dB": dice_3d,
            f"HD95_3D_at_{threshold:.0f}dB_m": float(hd95_dist),
            f"Chamfer_3D_at_{threshold:.0f}dB_m": float(chamfer_dist),
            f"GT_Voxels_at_{threshold:.0f}dB": gt_voxels,
            f"PRED_Voxels_at_{threshold:.0f}dB": pred_voxels,
        }

    def compute_azimuthal_spectral_metrics(self) -> dict:
        fft_gt = np.fft.rfft(self.tl_gt_3d, axis=2)
        fft_pred = np.fft.rfft(self.tl_pred_3d, axis=2)

        energy_gt = np.sum(np.abs(fft_gt) ** 2, axis=(0, 1))
        energy_pred = np.sum(np.abs(fft_pred) ** 2, axis=(0, 1))

        num = np.dot(energy_gt, energy_pred)
        den = (np.linalg.norm(energy_gt) * np.linalg.norm(energy_pred)) + 1e-8
        spectral_cosine_sim = float(num / den)

        half_modes = len(energy_gt) // 2
        hf_err_rel = float(
            np.mean(np.abs(energy_pred[half_modes:] - energy_gt[half_modes:]))
            / (np.mean(energy_gt[half_modes:]) + 1e-8)
        )

        return {
            "Spectral_Modal_Cosine_Similarity": spectral_cosine_sim,
            "HighFreq_Mode_Error_Rel": hf_err_rel,
        }

    def evaluate_all(self, thresholds=THRESHOLDS) -> dict:
        results = {}
        results.update(self.compute_volumetric_metrics())
        results.update(self.compute_azimuthal_gradient_and_tv())

        for thresh in thresholds:
            results.update(self.compute_isosurface_topology(thresh))

        results.update(self.compute_azimuthal_spectral_metrics())
        return results


def format_results_table(results: dict) -> str:
    lines = []
    lines.append("=" * 65)
    lines.append(f"{'MÉTRICA 3D':<42} | {'VALOR':<18}")
    lines.append("=" * 65)
    for key, val in results.items():
        if isinstance(val, float):
            lines.append(f"{key:<42} | {val:18.6f}")
        else:
            lines.append(f"{key:<42} | {str(val):<18}")
    lines.append("=" * 65)
    return "\n".join(lines)


def collect_vts_paths(vts_folder: str, gt_filename: str):
    if not os.path.isdir(vts_folder):
        raise FileNotFoundError(f"No existe la carpeta VTS: {vts_folder}")

    path_gt = os.path.join(vts_folder, gt_filename)
    if not os.path.exists(path_gt):
        raise FileNotFoundError(f"No se encontró el GT: {path_gt}")

    all_vts = sorted(glob.glob(os.path.join(vts_folder, "*.vts")))
    pred_paths = [p for p in all_vts if os.path.abspath(p) != os.path.abspath(path_gt)]

    if len(pred_paths) == 0:
        raise FileNotFoundError(
            f"No se encontraron VTS de predicción en {vts_folder} (aparte de {gt_filename})."
        )

    return path_gt, pred_paths


if __name__ == "__main__":
    path_gt, pred_paths = collect_vts_paths(VTS_FOLDER, GT_FILENAME)

    report_blocks = []
    report_blocks.append(f"GT: {path_gt}")
    report_blocks.append(f"Numero de predicciones a evaluar: {len(pred_paths)}")
    skipped_cases = []

    for idx, path_pred in enumerate(pred_paths, start=1):
        print(f"\n[{idx}/{len(pred_paths)}] Evaluando: {path_pred}")
        try:
            evaluator = VTS3DEvaluator(
                path_pred=path_pred, path_gt=path_gt, scalar_name=SCALAR_NAME
            )
            metrics = evaluator.evaluate_all(thresholds=THRESHOLDS)
        except ValueError as exc:
            message = f"[SKIP] {path_pred} -> {exc}"
            print(message)
            skipped_cases.append(message)
            report_blocks.append("#" * 90)
            report_blocks.append(f"PRED: {path_pred}")
            report_blocks.append(f"SKIPPED: {exc}")
            continue
        except Exception as exc:
            message = f"[ERROR] {path_pred} -> {exc}"
            print(message)
            skipped_cases.append(message)
            report_blocks.append("#" * 90)
            report_blocks.append(f"PRED: {path_pred}")
            report_blocks.append(f"ERROR: {exc}")
            continue

        report_str = format_results_table(metrics)

        block = []
        block.append("#" * 90)
        block.append(f"PRED: {path_pred}")
        block.append(report_str)
        report_blocks.append("\n".join(block))

        print("\n" + report_str + "\n")

    full_report = "\n\n".join(report_blocks)
    if skipped_cases:
        full_report += "\n\n" + "=" * 65 + "\n"
        full_report += "CASOS OMITIDOS\n"
        full_report += "=" * 65 + "\n"
        full_report += "\n".join(skipped_cases)

    if SAVE_TXT_REPORT:
        out_dir = os.path.dirname(OUTPUT_TXT_PATH)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(OUTPUT_TXT_PATH, "w", encoding="utf-8") as f:
            f.write(full_report)
        print(f"📄 Informe guardado en: {OUTPUT_TXT_PATH}")