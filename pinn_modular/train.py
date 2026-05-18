import os
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import PINN
from data_utils import (
    load_all_rois, load_validation_rois,
    Normalizer, CoordNormalizer, augment,
    save_mat, adaptive_figsize,
    sample_points_from_rois, predict_on_grid,
    compute_error_metrics, get_global_bounds,
)
from config import (
    ROI_HEIGHT, ROI_WIDTH, ROIS_PER_PLANE,
    ROI_MODE, ROI_CORNER,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, USE_AUGMENTATION,
    POINTS_PER_ROI, PHYSICS_BATCH_SIZE, DATA_WEIGHT, PHYSICS_WEIGHT,
    INTERFACE_BATCH_SIZE, INTERFACE_WEIGHT,
    PDE_TYPE, HELMHOLTZ_K,
    FREQUENCY_HZ,
    AIR_SOUND_SPEED, AIR_DENSITY,
    WATER_SOUND_SPEED, WATER_DENSITY,
    AIR_ABOVE_INTERFACE, INTERFACE_Z, INTERFACE_EPS,
    TL_REF_PRESSURE, OUTPUT_IS_TL,
    TL_DB_SIGN, TL_DB_OFFSET, TL_DB_CLAMP_MIN, TL_DB_CLAMP_MAX,
    PINN_HIDDEN_DIM, PINN_NUM_LAYERS, PINN_ACTIVATION, PINN_DROPOUT,
    DATA_FOLDER, VALIDATION_FOLDER, MODEL_PATH,
    TRAIN_PNG_FOLDER, TRAIN_MAT_FOLDER,
    VALIDATION_PNG_FOLDER, VALIDATION_MAT_FOLDER,
    INFER_BATCH_SIZE, SEED,
    create_output_dirs,
)


def format_seconds(seconds):
    seconds = int(max(0, seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def sample_collocation_points(num_points, bounds, angle_min, angle_max, rng):
    x_min, x_max, z_min, z_max = bounds
    x = rng.uniform(x_min, x_max, size=num_points).astype(np.float32)
    z = rng.uniform(z_min, z_max, size=num_points).astype(np.float32)
    a = rng.uniform(angle_min, angle_max, size=num_points).astype(np.float32)
    return x, z, a


def model_output_to_pressure(u_norm, norm, output_is_tl, ref_pressure,
                             tl_db_sign, tl_db_offset, tl_db_min, tl_db_max):
    if not output_is_tl:
        return u_norm
    tl_db = u_norm * (norm.tl_max - norm.tl_min) + norm.tl_min
    tl_db = tl_db * float(tl_db_sign) + float(tl_db_offset)
    tl_db = torch.clamp(tl_db, min=float(tl_db_min), max=float(tl_db_max))
    scale = float(np.log(10.0) / 20.0)
    return ref_pressure * torch.exp(-tl_db * scale)


def norm_to_phys_z(z_norm, coord_norm):
    return (z_norm + 1.0) / coord_norm.scale_z + coord_norm.z_min


def compute_pde_residual(model, coords, coord_norm, norm, pde_type, helmholtz_k,
                         frequency_hz, air_c, water_c, bounds, interface_z, air_above,
                         output_is_tl, ref_pressure,
                         tl_db_sign, tl_db_offset, tl_db_min, tl_db_max):
    if pde_type == "none":
        return torch.zeros_like(coords[:, :1])

    coords.requires_grad_(True)
    u = model(coords)
    p = model_output_to_pressure(
        u, norm, output_is_tl, ref_pressure,
        tl_db_sign, tl_db_offset, tl_db_min, tl_db_max,
    )

    grads = torch.autograd.grad(
        p, coords,
        grad_outputs=torch.ones_like(p),
        create_graph=True,
    )[0]
    p_xn = grads[:, 0:1]
    p_zn = grads[:, 1:2]

    grads_xn = torch.autograd.grad(
        p_xn, coords,
        grad_outputs=torch.ones_like(p_xn),
        create_graph=True,
    )[0]
    grads_zn = torch.autograd.grad(
        p_zn, coords,
        grad_outputs=torch.ones_like(p_zn),
        create_graph=True,
    )[0]

    p_xx = grads_xn[:, 0:1] * (coord_norm.scale_x ** 2)
    p_zz = grads_zn[:, 1:2] * (coord_norm.scale_z ** 2)

    if pde_type == "laplace":
        return p_xx + p_zz
    if pde_type == "helmholtz":
        k2 = float(helmholtz_k) ** 2
        return p_xx + p_zz + k2 * p
    if pde_type == "two_layer_helmholtz":
        k_air = 2.0 * np.pi * float(frequency_hz) / float(air_c)
        k_water = 2.0 * np.pi * float(frequency_hz) / float(water_c)
        k_air_t = torch.tensor(k_air, device=coords.device, dtype=p.dtype)
        k_water_t = torch.tensor(k_water, device=coords.device, dtype=p.dtype)

        z_phys = norm_to_phys_z(coords[:, 1:2], coord_norm)
        z_min = float(bounds[2])
        z_max = float(bounds[3])

        interface_in_roi = z_min < float(interface_z) < z_max
        if not interface_in_roi:
            if air_above:
                use_air = z_min >= float(interface_z)
            else:
                use_air = z_max <= float(interface_z)
            k_single = k_air_t if use_air else k_water_t
            return p_xx + p_zz + (k_single ** 2) * p

        if air_above:
            mask_air = z_phys >= float(interface_z)
        else:
            mask_air = z_phys < float(interface_z)
        k = torch.where(mask_air, k_air_t, k_water_t)
        return p_xx + p_zz + (k ** 2) * p
    return p_xx + p_zz


def compute_interface_loss(model, coord_norm, norm, bounds, angle_min, angle_max, rng,
                           batch_size, interface_z, interface_eps, air_above,
                           air_rho, water_rho, output_is_tl, ref_pressure,
                           tl_db_sign, tl_db_offset, tl_db_min, tl_db_max,
                           device):
    x_min, x_max, z_min, z_max = bounds
    if interface_z <= z_min or interface_z >= z_max:
        return torch.tensor(0.0, device=device)

    x = rng.uniform(x_min, x_max, size=batch_size).astype(np.float32)
    a = rng.uniform(angle_min, angle_max, size=batch_size).astype(np.float32)

    z_minus = np.full_like(x, interface_z - interface_eps, dtype=np.float32)
    z_plus = np.full_like(x, interface_z + interface_eps, dtype=np.float32)
    z_minus = np.clip(z_minus, z_min, z_max)
    z_plus = np.clip(z_plus, z_min, z_max)

    x_n_m, z_n_m, a_n_m = coord_norm.normalize(x, z_minus, a)
    x_n_p, z_n_p, a_n_p = coord_norm.normalize(x, z_plus, a)

    coords_m = np.stack([x_n_m, z_n_m, a_n_m], axis=1).astype(np.float32)
    coords_p = np.stack([x_n_p, z_n_p, a_n_p], axis=1).astype(np.float32)

    coords_m = torch.from_numpy(coords_m).to(device)
    coords_p = torch.from_numpy(coords_p).to(device)
    coords_m.requires_grad_(True)
    coords_p.requires_grad_(True)

    p_m = model_output_to_pressure(
        model(coords_m), norm, output_is_tl, ref_pressure,
        tl_db_sign, tl_db_offset, tl_db_min, tl_db_max,
    )
    p_p = model_output_to_pressure(
        model(coords_p), norm, output_is_tl, ref_pressure,
        tl_db_sign, tl_db_offset, tl_db_min, tl_db_max,
    )

    grad_m = torch.autograd.grad(
        p_m, coords_m,
        grad_outputs=torch.ones_like(p_m),
        create_graph=True,
    )[0]
    grad_p = torch.autograd.grad(
        p_p, coords_p,
        grad_outputs=torch.ones_like(p_p),
        create_graph=True,
    )[0]

    dpdz_m = grad_m[:, 1:2] * coord_norm.scale_z
    dpdz_p = grad_p[:, 1:2] * coord_norm.scale_z

    if air_above:
        rho_p = float(air_rho)
        rho_m = float(water_rho)
    else:
        rho_p = float(water_rho)
        rho_m = float(air_rho)

    rho_p_t = torch.tensor(rho_p, device=device, dtype=dpdz_p.dtype)
    rho_m_t = torch.tensor(rho_m, device=device, dtype=dpdz_m.dtype)

    res_p = p_p - p_m
    res_v = dpdz_p / rho_p_t - dpdz_m / rho_m_t

    return torch.mean(res_p ** 2) + torch.mean(res_v ** 2)


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    rng = np.random.default_rng(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    create_output_dirs(TRAIN_PNG_FOLDER, TRAIN_MAT_FOLDER)

    print("=" * 60)
    print("  Loading planes and extracting ROIs")
    print("=" * 60)
    rois, roi_angles, roi_extents = load_all_rois(
        DATA_FOLDER, ROI_HEIGHT, ROI_WIDTH, ROIS_PER_PLANE,
        roi_mode=ROI_MODE, roi_corner=ROI_CORNER,
    )
    if rois is None or len(rois) == 0:
        print("[error] No ROIs found. Check input data and ROI settings.")
        return

    print(f"ROIs: {len(rois)}  |  Shape: {rois.shape}")

    try:
        norm = Normalizer(rois, roi_angles)
    except ValueError as e:
        print(f"[error] Normalizer failed: {e}")
        return

    rois_norm = norm.normalize_tl(rois)

    if USE_AUGMENTATION:
        rois_aug, angles_aug = augment(rois_norm, roi_angles)
        extents_aug = list(roi_extents)
        for ext in roi_extents:
            extents_aug.extend([ext, ext, ext])
        print(f"Augmentation enabled: {len(rois_aug)} samples")
    else:
        rois_aug = rois_norm
        angles_aug = np.array(roi_angles, dtype=np.float32)
        extents_aug = list(roi_extents)
        print(f"Augmentation disabled: {len(rois_aug)} samples")

    bounds = get_global_bounds(extents_aug)
    interface_z = INTERFACE_Z
    if interface_z is None:
        interface_z = 0.5 * (bounds[2] + bounds[3])
        print(f"Auto interface_z set to {interface_z:.3f}")
    coord_norm = CoordNormalizer(
        bounds[0], bounds[1], bounds[2], bounds[3],
        norm.angle_min, norm.angle_max,
    )

    model = PINN(
        hidden_dim=PINN_HIDDEN_DIM,
        num_layers=PINN_NUM_LAYERS,
        activation=PINN_ACTIVATION,
        dropout=PINN_DROPOUT,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}")

    history = {"loss": [], "data_loss": [], "physics_loss": [], "interface_loss": []}
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        xs, zs, angs, tls = sample_points_from_rois(
            rois_aug, angles_aug, extents_aug, POINTS_PER_ROI, rng,
        )
        x_n, z_n, a_n = coord_norm.normalize(xs, zs, angs)
        coords = np.stack([x_n, z_n, a_n], axis=1).astype(np.float32)
        targets = tls.astype(np.float32).reshape(-1, 1)

        dataset = TensorDataset(torch.from_numpy(coords), torch.from_numpy(targets))
        use_pin = device.type == "cuda"
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            pin_memory=use_pin,
            num_workers=0,
        )

        model.train()
        ep_loss = 0.0
        ep_data = 0.0
        ep_phys = 0.0
        ep_iface = 0.0
        n_batches = 0

        for batch_coords, batch_targets in loader:
            batch_coords = batch_coords.to(device, non_blocking=use_pin)
            batch_targets = batch_targets.to(device, non_blocking=use_pin)

            optimizer.zero_grad()

            pred = model(batch_coords)
            data_loss = criterion(pred, batch_targets)

            physics_loss = torch.tensor(0.0, device=device)
            interface_loss = torch.tensor(0.0, device=device)
            if PHYSICS_WEIGHT > 0:
                x_c, z_c, a_c = sample_collocation_points(
                    PHYSICS_BATCH_SIZE, bounds, norm.angle_min, norm.angle_max, rng,
                )
                x_cn, z_cn, a_cn = coord_norm.normalize(x_c, z_c, a_c)
                coords_c = np.stack([x_cn, z_cn, a_cn], axis=1).astype(np.float32)
                coords_c = torch.from_numpy(coords_c).to(device)
                coords_c.requires_grad_(True)

                residual = compute_pde_residual(
                    model, coords_c, coord_norm, norm, PDE_TYPE, HELMHOLTZ_K,
                    FREQUENCY_HZ, AIR_SOUND_SPEED, WATER_SOUND_SPEED, bounds,
                    interface_z, AIR_ABOVE_INTERFACE,
                    OUTPUT_IS_TL, TL_REF_PRESSURE,
                    TL_DB_SIGN, TL_DB_OFFSET, TL_DB_CLAMP_MIN, TL_DB_CLAMP_MAX,
                )
                physics_loss = torch.mean(residual ** 2)

                if (PDE_TYPE == "two_layer_helmholtz" and INTERFACE_WEIGHT > 0
                        and INTERFACE_BATCH_SIZE > 0):
                    interface_loss = compute_interface_loss(
                        model, coord_norm, norm, bounds, norm.angle_min, norm.angle_max, rng,
                        INTERFACE_BATCH_SIZE, interface_z, INTERFACE_EPS, AIR_ABOVE_INTERFACE,
                        AIR_DENSITY, WATER_DENSITY,
                        OUTPUT_IS_TL, TL_REF_PRESSURE,
                        TL_DB_SIGN, TL_DB_OFFSET, TL_DB_CLAMP_MIN, TL_DB_CLAMP_MAX,
                        device,
                    )

            physics_total = physics_loss + INTERFACE_WEIGHT * interface_loss
            loss = DATA_WEIGHT * data_loss + PHYSICS_WEIGHT * physics_total
            loss.backward()
            optimizer.step()

            ep_loss += loss.item()
            ep_data += data_loss.item()
            ep_phys += physics_loss.item()
            ep_iface += interface_loss.item()
            n_batches += 1

        avg_loss = ep_loss / n_batches
        avg_data = ep_data / n_batches
        avg_phys = ep_phys / n_batches
        avg_iface = ep_iface / n_batches

        history["loss"].append(avg_loss)
        history["data_loss"].append(avg_data)
        history["physics_loss"].append(avg_phys)
        history["interface_loss"].append(avg_iface)

        if epoch == 1 or epoch % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / epoch * (EPOCHS - epoch)
            print(
                f"Epoch {epoch:4d}/{EPOCHS} | loss={avg_loss:.6f} "
                f"data={avg_data:.6f} phys={avg_phys:.6f} iface={avg_iface:.6f} "
                f"| elapsed={format_seconds(elapsed)} ETA={format_seconds(eta)}"
            )

    total_time = time.time() - t0
    print(f"Training done in {format_seconds(total_time)}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["loss"], label="total", linewidth=1.2)
    ax.plot(history["data_loss"], label="data", linewidth=1.2)
    if PHYSICS_WEIGHT > 0:
        ax.plot(history["physics_loss"], label="physics", linewidth=1.2)
        ax.plot(history["interface_loss"], label="interface", linewidth=1.2)
    ax.set_title("Training loss - PINN")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(TRAIN_PNG_FOLDER, "training_curve_pinn.png"), dpi=150)
    plt.close(fig)

    print("Exporting original ROIs as .mat...")
    for j in range(len(rois)):
        roi_mat_path = os.path.join(TRAIN_MAT_FOLDER, f"roi_original_{roi_angles[j]:+.2f}.mat")
        save_mat(roi_mat_path, rois[j], extent=roi_extents[j])

    print("Comparing original vs predicted (training set)...")
    n_compare = len(rois)
    n_cols = min(n_compare, 6)
    n_pages = int(np.ceil(n_compare / n_cols))

    model.eval()
    with torch.no_grad():
        for page in range(n_pages):
            start = page * n_cols
            end = min(start + n_cols, n_compare)
            cols_this = end - start

            fig, axes = plt.subplots(
                3, cols_this,
                figsize=adaptive_figsize(ROI_HEIGHT, ROI_WIDTH, 3, cols_this),
            )
            if cols_this == 1:
                axes = axes[:, np.newaxis]

            for c, j in enumerate(range(start, end)):
                angle = roi_angles[j]
                extent = roi_extents[j]
                roi = rois[j]

                pred_norm = predict_on_grid(
                    model, coord_norm, angle, extent, roi.shape, device,
                    batch_size=INFER_BATCH_SIZE,
                )
                pred_tl = norm.denormalize_tl(np.clip(pred_norm, 0, 1))

                errors = compute_error_metrics(roi, pred_tl)
                err_map = np.abs(roi - pred_tl)
                mape_str = f"{errors['mape']:.1f}%" if errors["mape"] != np.inf else "undef"

                im0 = axes[0, c].imshow(
                    roi, cmap="jet", aspect="auto", origin="lower",
                    vmin=norm.tl_min, vmax=norm.tl_max, extent=extent,
                )
                fig.colorbar(im0, ax=axes[0, c], shrink=0.6, label="TL")
                axes[0, c].set_title(f"Original - {angle:.2f} deg", fontsize=9)

                im1 = axes[1, c].imshow(
                    pred_tl, cmap="jet", aspect="auto", origin="lower",
                    vmin=norm.tl_min, vmax=norm.tl_max, extent=extent,
                )
                fig.colorbar(im1, ax=axes[1, c], shrink=0.6, label="TL")
                axes[1, c].set_title(f"Predicted - {angle:.2f} deg", fontsize=9)

                im2 = axes[2, c].imshow(
                    err_map, cmap="turbo", aspect="auto", origin="lower",
                    vmin=0, vmax=errors["max_error"] or 1e-6, extent=extent,
                )
                fig.colorbar(im2, ax=axes[2, c], shrink=0.6, label="abs error")
                axes[2, c].set_title(
                    f"Err - {angle:.2f} deg\n"
                    f"MAE={errors['mae']:.2f} RMSE={errors['rmse']:.2f} "
                    f"MAPE={mape_str} Max={errors['max_error']:.2f}",
                    fontsize=7,
                )

            fig.suptitle(
                f"Training: original vs predicted (page {page + 1}/{n_pages})",
                fontsize=12,
            )
            fig.tight_layout()
            fig.savefig(
                os.path.join(TRAIN_PNG_FOLDER, f"compare_train_{page + 1:02d}.png"),
                dpi=150,
            )
            plt.close(fig)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "roi_height": ROI_HEIGHT,
        "roi_width": ROI_WIDTH,
        "pde_type": PDE_TYPE,
        "helmholtz_k": HELMHOLTZ_K,
        "frequency_hz": FREQUENCY_HZ,
        "air_sound_speed": AIR_SOUND_SPEED,
        "air_density": AIR_DENSITY,
        "water_sound_speed": WATER_SOUND_SPEED,
        "water_density": WATER_DENSITY,
        "air_above_interface": AIR_ABOVE_INTERFACE,
        "interface_z": interface_z,
        "interface_eps": INTERFACE_EPS,
        "output_is_tl": OUTPUT_IS_TL,
        "tl_ref_pressure": TL_REF_PRESSURE,
        "tl_db_sign": TL_DB_SIGN,
        "tl_db_offset": TL_DB_OFFSET,
        "tl_db_clamp_min": TL_DB_CLAMP_MIN,
        "tl_db_clamp_max": TL_DB_CLAMP_MAX,
        "pinn_hidden_dim": PINN_HIDDEN_DIM,
        "pinn_num_layers": PINN_NUM_LAYERS,
        "pinn_activation": PINN_ACTIVATION,
        "pinn_dropout": PINN_DROPOUT,
        **norm.state_dict(),
        **coord_norm.state_dict(),
    }
    torch.save(checkpoint, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")

    val_rois, val_angles, val_extents = load_validation_rois(
        VALIDATION_FOLDER, ROI_HEIGHT, ROI_WIDTH,
        roi_mode=ROI_MODE, roi_corner=ROI_CORNER,
    )
    if val_rois is None or len(val_rois) == 0:
        print("No validation planes found.")
        return

    create_output_dirs(VALIDATION_PNG_FOLDER, VALIDATION_MAT_FOLDER)

    train_angles_sorted = np.sort(np.array(roi_angles))
    roi_angles_arr = np.array(roi_angles)

    n_val = len(val_rois)
    n_rows_per_page = min(n_val, 6)
    n_pages = int(np.ceil(n_val / n_rows_per_page))

    all_maes = []
    all_maxes = []
    all_rmses = []
    all_mapes = []
    all_gaps = []
    all_val_angles = []

    with torch.no_grad():
        for page in range(n_pages):
            start = page * n_rows_per_page
            end = min(start + n_rows_per_page, n_val)
            rows_this = end - start

            fig, axes = plt.subplots(
                rows_this, 3,
                figsize=adaptive_figsize(ROI_HEIGHT, ROI_WIDTH, rows_this, 3),
            )
            if rows_this == 1:
                axes = axes[np.newaxis, :]

            for r, j in enumerate(range(start, end)):
                val_angle = val_angles[j]
                extent = val_extents[j]
                real_tl = val_rois[j]

                lower_mask = train_angles_sorted[train_angles_sorted <= val_angle]
                upper_mask = train_angles_sorted[train_angles_sorted >= val_angle]
                ang_lower = float(lower_mask[-1]) if len(lower_mask) > 0 else None
                ang_upper = float(upper_mask[0]) if len(upper_mask) > 0 else None

                diff_lower = abs(val_angle - ang_lower) if ang_lower is not None else None
                diff_upper = abs(ang_upper - val_angle) if ang_upper is not None else None
                if ang_lower is not None and ang_upper is not None:
                    gap = abs(ang_upper - ang_lower)
                else:
                    gap = None

                lo_str = f"{ang_lower:+.2f} (d={diff_lower:.2f})" if ang_lower is not None else "-"
                hi_str = f"{ang_upper:+.2f} (d={diff_upper:.2f})" if ang_upper is not None else "-"
                gap_str = f"{gap:.2f}" if gap is not None else "-"
                print(
                    f"Val angle {val_angle:+.2f} | low {lo_str} | high {hi_str} | gap {gap_str}"
                )

                pred_norm = predict_on_grid(
                    model, coord_norm, val_angle, extent, real_tl.shape, device,
                    batch_size=INFER_BATCH_SIZE,
                )
                pred_tl = norm.denormalize_tl(np.clip(pred_norm, 0, 1))

                errors = compute_error_metrics(real_tl, pred_tl)
                diff_map = np.abs(real_tl - pred_tl)

                all_maes.append(errors["mae"])
                all_maxes.append(errors["max_error"])
                all_rmses.append(errors["rmse"])
                all_mapes.append(errors["mape"])
                all_gaps.append(gap)
                all_val_angles.append(val_angle)

                mat_path = os.path.join(
                    VALIDATION_MAT_FOLDER,
                    f"val_generated_{val_angle:+.2f}.mat",
                )
                save_mat(mat_path, pred_tl, extent=extent)

                mape_str = f"{errors['mape']:.1f}%" if errors["mape"] != np.inf else "undef"

                im0 = axes[r, 0].imshow(
                    real_tl, cmap="jet", aspect="auto", origin="lower",
                    vmin=norm.tl_min, vmax=norm.tl_max, extent=extent,
                )
                fig.colorbar(im0, ax=axes[r, 0], shrink=0.6, label="TL")
                axes[r, 0].set_title(
                    f"Real - {val_angle:.2f} deg\nlow {lo_str} | high {hi_str} | gap {gap_str}",
                    fontsize=8,
                )

                im1 = axes[r, 1].imshow(
                    pred_tl, cmap="jet", aspect="auto", origin="lower",
                    vmin=norm.tl_min, vmax=norm.tl_max, extent=extent,
                )
                fig.colorbar(im1, ax=axes[r, 1], shrink=0.6, label="TL")
                axes[r, 1].set_title(f"Predicted - {val_angle:.2f} deg", fontsize=8)

                im2 = axes[r, 2].imshow(
                    diff_map, cmap="turbo", aspect="auto", origin="lower",
                    vmin=0, vmax=errors["max_error"] or 1e-6, extent=extent,
                )
                fig.colorbar(im2, ax=axes[r, 2], shrink=0.6, label="abs error")
                axes[r, 2].set_title(
                    f"Err - {val_angle:.2f} deg\nMAE={errors['mae']:.2f} "
                    f"RMSE={errors['rmse']:.2f} MAPE={mape_str} Max={errors['max_error']:.2f}",
                    fontsize=7,
                )

            fig.suptitle(
                f"Validation: real vs predicted (page {page + 1}/{n_pages})",
                fontsize=12,
            )
            fig.tight_layout()
            fig.savefig(
                os.path.join(VALIDATION_PNG_FOLDER, f"validation_{page + 1:02d}.png"),
                dpi=150,
            )
            plt.close(fig)

    plot_data = [
        (g, m, mx, r, mp, a)
        for g, m, mx, r, mp, a in zip(all_gaps, all_maes, all_maxes, all_rmses, all_mapes, all_val_angles)
        if g is not None
    ]
    if plot_data:
        plot_data.sort(key=lambda d: d[0])
        gaps_arr = np.array([d[0] for d in plot_data])
        maes_arr = np.array([d[1] for d in plot_data])
        maxes_arr = np.array([d[2] for d in plot_data])
        rmses_arr = np.array([d[3] for d in plot_data])
        mapes_arr = np.array([d[4] for d in plot_data])
        labels_arr = [f"{d[5]:+.2f}" for d in plot_data]

        fig_err, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        ax1.plot(gaps_arr, maes_arr, color="#1f77b4", linewidth=1.2, alpha=0.6)
        ax1.plot(gaps_arr, rmses_arr, color="#ff7f0e", linewidth=1.2, alpha=0.6)
        ax1.plot(gaps_arr, maxes_arr, color="#d62728", linewidth=1.2, alpha=0.6, linestyle="--")

        ax1.scatter(gaps_arr, maes_arr, color="#1f77b4", s=40, label="MAE")
        ax1.scatter(gaps_arr, rmses_arr, color="#ff7f0e", s=40, label="RMSE")
        ax1.scatter(gaps_arr, maxes_arr, color="#d62728", s=40, marker="^", label="Max")

        ax1.set_xlabel("Gap between training neighbors (deg)")
        ax1.set_ylabel("Error")
        ax1.set_title("Absolute error vs angular gap")
        ax1.legend(fontsize=9, loc="upper left")
        ax1.grid(True, alpha=0.3)

        ax2.plot(gaps_arr, mapes_arr, color="#2ca02c", linewidth=1.2, alpha=0.6)
        ax2.scatter(gaps_arr, mapes_arr, color="#2ca02c", s=40, label="MAPE")

        for xi, yi, lbl in zip(gaps_arr, mapes_arr, labels_arr):
            ax2.annotate(lbl, (xi, yi), textcoords="offset points", xytext=(5, 5), fontsize=7)

        ax2.set_xlabel("Gap between training neighbors (deg)")
        ax2.set_ylabel("MAPE (%)")
        ax2.set_title("Relative error vs angular gap")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        fig_err.suptitle("Validation: error vs angular gap", fontsize=12)
        fig_err.tight_layout()
        err_path = os.path.join(VALIDATION_PNG_FOLDER, "error_vs_gap.png")
        fig_err.savefig(err_path, dpi=150)
        plt.close(fig_err)

    mean_mae = np.mean(all_maes)
    mean_rmse = np.mean(all_rmses)
    mean_mape = np.mean([m for m in all_mapes if m != np.inf])

    print("Validation summary:")
    print(f"  MAE mean  = {mean_mae:.2f}")
    print(f"  RMSE mean = {mean_rmse:.2f}")
    print(f"  MAPE mean = {mean_mape:.1f}%")
    print(f"  Planes    = {len(all_maes)}")


if __name__ == "__main__":
    main()
