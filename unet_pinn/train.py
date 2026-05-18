import os
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast
 
from model import ConditionalUNetAE
from data_utils import (
    load_all_rois, load_validation_rois,
    Normalizer, augment, save_mat, adaptive_figsize,
    generate_from_seed, compute_error_metrics,
    generate_inpainting_known_masks, apply_inpainting_mask,
)
from config import (
    ROI_HEIGHT, ROI_WIDTH,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, MODEL_DROPOUT, ROIS_PER_PLANE,
    ROI_MODE, ROI_CORNER, NOISE_STD, USE_AUGMENTATION,
    INPAINT_MODE, INPAINT_PRESERVE_KNOWN,
    INPAINT_MIN_HOLES, INPAINT_MAX_HOLES,
    INPAINT_MIN_HOLE_RATIO, INPAINT_MAX_HOLE_RATIO,
    INPAINT_KEEP_FULL_PROB, INPAINT_FILL_VALUE,
    INPAINT_LOSS_KNOWN_WEIGHT,
    SEED,
    PHYSICS_BATCH_SIZE, DATA_WEIGHT, PHYSICS_WEIGHT,
    INTERFACE_BATCH_SIZE, INTERFACE_WEIGHT,
    PDE_TYPE, HELMHOLTZ_K,
    FREQUENCY_HZ, AIR_SOUND_SPEED, AIR_DENSITY,
    WATER_SOUND_SPEED, WATER_DENSITY,
    AIR_ABOVE_INTERFACE, INTERFACE_Z, INTERFACE_EPS,
    TL_REF_PRESSURE, OUTPUT_IS_TL,
    TL_DB_SIGN, TL_DB_OFFSET, TL_DB_CLAMP_MIN, TL_DB_CLAMP_MAX,
    DATA_FOLDER, MODEL_PATH,
    VALIDATION_FOLDER,
    TRAIN_PNG_FOLDER, TRAIN_MAT_FOLDER,
    VALIDATION_PNG_FOLDER, VALIDATION_MAT_FOLDER,
    create_output_dirs,
) 


def format_seconds(seconds):
    """Convierte segundos a un formato legible (h, min, s).
    
    Ejemplos:
        58000 → "16h 6m 40s"
        3661  → "1h 1m 1s"
        125   → "2m 5s"
        45    → "45s"
    """
    seconds = int(max(0, seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def model_output_to_pressure(u_norm, norm, output_is_tl, ref_pressure,
                             tl_db_sign, tl_db_offset, tl_db_min, tl_db_max):
    if not output_is_tl:
        return u_norm
    tl_db = u_norm * (norm.tl_max - norm.tl_min) + norm.tl_min
    tl_db = tl_db * float(tl_db_sign) + float(tl_db_offset)
    tl_db = torch.clamp(tl_db, min=float(tl_db_min), max=float(tl_db_max))
    scale = float(np.log(10.0) / 20.0)
    return float(ref_pressure) * torch.exp(-tl_db * scale)


def _extent_to_spacing(extents, height, width):
    x_left = extents[:, 0]
    x_right = extents[:, 1]
    z_bottom = extents[:, 2]
    z_top = extents[:, 3]

    denom_x = max(1, int(width) - 1)
    denom_z = max(1, int(height) - 1)
    dx = (x_right - x_left) / float(denom_x)
    dz = (z_top - z_bottom) / float(denom_z)
    return dx.clamp_min(1e-8), dz.clamp_min(1e-8)


def _get_interface_z(extents, interface_z):
    z_bottom = extents[:, 2]
    z_top = extents[:, 3]
    if interface_z is None:
        return 0.5 * (z_bottom + z_top)
    return torch.full_like(z_bottom, float(interface_z))


def compute_pde_residual_grid(p, extents, pde_type, helmholtz_k,
                              frequency_hz, air_c, water_c,
                              interface_z, air_above):
    if pde_type == "none":
        return None

    bsz, _, height, width = p.shape
    if height < 3 or width < 3:
        return None

    dx, dz = _extent_to_spacing(extents, height, width)
    dx2 = (dx ** 2).view(bsz, 1, 1, 1)
    dz2 = (dz ** 2).view(bsz, 1, 1, 1)

    p_xx = (p[:, :, :, 2:] - 2.0 * p[:, :, :, 1:-1] + p[:, :, :, :-2]) / dx2
    p_zz = (p[:, :, 2:, :] - 2.0 * p[:, :, 1:-1, :] + p[:, :, :-2, :]) / dz2

    p_xx = p_xx[:, :, 1:-1, :]
    p_zz = p_zz[:, :, :, 1:-1]
    lap = p_xx + p_zz
    p_int = p[:, :, 1:-1, 1:-1]

    if pde_type == "laplace":
        return lap
    if pde_type == "helmholtz":
        k2 = float(helmholtz_k) ** 2
        return lap + k2 * p_int
    if pde_type == "two_layer_helmholtz":
        k_air = 2.0 * np.pi * float(frequency_hz) / float(air_c)
        k_water = 2.0 * np.pi * float(frequency_hz) / float(water_c)
        k_air2 = torch.tensor(k_air ** 2, device=p.device, dtype=p.dtype)
        k_water2 = torch.tensor(k_water ** 2, device=p.device, dtype=p.dtype)

        z_bottom = extents[:, 2]
        z_top = extents[:, 3]
        interface = _get_interface_z(extents, interface_z)

        base = torch.linspace(0.0, 1.0, height, device=p.device, dtype=p.dtype)
        z_axis = z_bottom.view(bsz, 1) + base.view(1, height) * (z_top - z_bottom).view(bsz, 1)
        z_interior = z_axis[:, 1:-1].view(bsz, 1, height - 2, 1)

        k2_grid = torch.empty_like(z_interior)
        interface_in_roi = (interface > z_bottom) & (interface < z_top)

        for i in range(bsz):
            if not interface_in_roi[i]:
                if air_above:
                    use_air = z_bottom[i] >= interface[i]
                else:
                    use_air = z_top[i] <= interface[i]
                k2_grid[i] = k_air2 if use_air else k_water2
            else:
                if air_above:
                    mask_air = z_interior[i] >= interface[i]
                else:
                    mask_air = z_interior[i] < interface[i]
                k2_grid[i] = torch.where(mask_air, k_air2, k_water2)

        return lap + k2_grid * p_int
    return lap


def _select_row(p, idx):
    return p[:, :, idx, :]


def compute_interface_loss_grid(p, extents, interface_z, interface_eps, air_above,
                                air_rho, water_rho, sample_size):
    bsz, _, height, width = p.shape
    if height < 2:
        return torch.tensor(0.0, device=p.device)

    z_bottom = extents[:, 2]
    z_top = extents[:, 3]
    interface = _get_interface_z(extents, interface_z)

    base = torch.linspace(0.0, 1.0, height, device=p.device, dtype=p.dtype)
    z_axis = z_bottom.view(bsz, 1) + base.view(1, height) * (z_top - z_bottom).view(bsz, 1)

    loss_vals = []
    for i in range(bsz):
        z_min = float(z_bottom[i])
        z_max = float(z_top[i])
        iface = float(interface[i])
        if iface <= z_min or iface >= z_max:
            continue

        z_minus = max(z_min, iface - float(interface_eps))
        z_plus = min(z_max, iface + float(interface_eps))
        z_axis_i = z_axis[i]

        idx_minus = int(torch.argmin(torch.abs(z_axis_i - z_minus)))
        idx_plus = int(torch.argmin(torch.abs(z_axis_i - z_plus)))
        if idx_plus == idx_minus:
            idx_plus = min(idx_plus + 1, height - 1)
            idx_minus = max(idx_minus - 1, 0)

        dz = float((z_max - z_min) / max(1, height - 1))
        dz = max(dz, 1e-8)

        p_minus = _select_row(p[i:i + 1], idx_minus)
        p_plus = _select_row(p[i:i + 1], idx_plus)

        if idx_minus > 0:
            p_m_prev = _select_row(p[i:i + 1], idx_minus - 1)
            dpdz_minus = (p_minus - p_m_prev) / dz
        else:
            p_m_next = _select_row(p[i:i + 1], idx_minus + 1)
            dpdz_minus = (p_m_next - p_minus) / dz

        if idx_plus < height - 1:
            p_p_next = _select_row(p[i:i + 1], idx_plus + 1)
            dpdz_plus = (p_p_next - p_plus) / dz
        else:
            p_p_prev = _select_row(p[i:i + 1], idx_plus - 1)
            dpdz_plus = (p_plus - p_p_prev) / dz

        if air_above:
            rho_plus = float(air_rho)
            rho_minus = float(water_rho)
        else:
            rho_plus = float(water_rho)
            rho_minus = float(air_rho)

        res_p = p_plus - p_minus
        res_v = dpdz_plus / rho_plus - dpdz_minus / rho_minus

        if sample_size > 0 and sample_size < width:
            idx = torch.randint(0, width, (sample_size,), device=p.device)
            res_p = res_p[:, :, idx]
            res_v = res_v[:, :, idx]

        loss_vals.append(torch.mean(res_p ** 2) + torch.mean(res_v ** 2))

    if not loss_vals:
        return torch.tensor(0.0, device=p.device)
    return torch.mean(torch.stack(loss_vals))


def compute_physics_losses(recon, batch_extents, norm, device):
    physics_loss = torch.tensor(0.0, device=device)
    interface_loss = torch.tensor(0.0, device=device)

    if PHYSICS_WEIGHT <= 0 or PDE_TYPE == "none":
        return physics_loss, interface_loss

    pressure = model_output_to_pressure(
        recon, norm, OUTPUT_IS_TL, TL_REF_PRESSURE,
        TL_DB_SIGN, TL_DB_OFFSET, TL_DB_CLAMP_MIN, TL_DB_CLAMP_MAX,
    )

    residual = compute_pde_residual_grid(
        pressure, batch_extents, PDE_TYPE, HELMHOLTZ_K,
        FREQUENCY_HZ, AIR_SOUND_SPEED, WATER_SOUND_SPEED,
        INTERFACE_Z, AIR_ABOVE_INTERFACE,
    )

    if residual is not None:
        res_flat = residual.reshape(residual.shape[0], -1)
        if PHYSICS_BATCH_SIZE > 0 and PHYSICS_BATCH_SIZE < res_flat.shape[1]:
            idx = torch.randint(
                0, res_flat.shape[1],
                (res_flat.shape[0], PHYSICS_BATCH_SIZE),
                device=res_flat.device,
            )
            res_sel = res_flat.gather(1, idx)
            physics_loss = torch.mean(res_sel ** 2)
        else:
            physics_loss = torch.mean(res_flat ** 2)

    if (PDE_TYPE == "two_layer_helmholtz" and INTERFACE_WEIGHT > 0
            and INTERFACE_BATCH_SIZE != 0):
        interface_loss = compute_interface_loss_grid(
            pressure, batch_extents, INTERFACE_Z, INTERFACE_EPS,
            AIR_ABOVE_INTERFACE, AIR_DENSITY, WATER_DENSITY,
            INTERFACE_BATCH_SIZE,
        )

    return physics_loss, interface_loss

 
def main():
    # ══════════════════════════════════════════════════════════════════════
    #  DISPOSITIVO
    # ══════════════════════════════════════════════════════════════════════
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enable_train_inpaint = INPAINT_MODE in ("train", "both")
    enable_infer_inpaint = INPAINT_MODE in ("inference", "both")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    print(f"Dispositivo: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Modo inpainting: {INPAINT_MODE}")
    if enable_train_inpaint or enable_infer_inpaint:
        print(
            "  Mascaras: "
            f"holes=[{INPAINT_MIN_HOLES},{INPAINT_MAX_HOLES}] "
            f"ratio=[{INPAINT_MIN_HOLE_RATIO:.2f},{INPAINT_MAX_HOLE_RATIO:.2f}]"
        )

    create_output_dirs(TRAIN_PNG_FOLDER, TRAIN_MAT_FOLDER)

    # ══════════════════════════════════════════════════════════════════════
    #  1. CARGAR DATOS Y EXTRAER ROIs
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("  CARGANDO PLANOS Y EXTRAYENDO ROIs")
    print("=" * 60)
    rois, roi_angles, roi_extents = load_all_rois(
        DATA_FOLDER, ROI_HEIGHT, ROI_WIDTH, ROIS_PER_PLANE,
        roi_mode=ROI_MODE, roi_corner=ROI_CORNER,
    )
    print(f"\nROIs base: {len(rois)}  |  Forma: {rois.shape}")

    # ══════════════════════════════════════════════════════════════════════
    #  2. NORMALIZACIÓN
    # ══════════════════════════════════════════════════════════════════════
    try:
        norm = Normalizer(rois, roi_angles)
    except ValueError as e:
        print(f"\n  [ERROR] No se pudieron inicializar normalizadores: {e}")
        print("  Revisa DATA_FOLDER, la estructura de los .mat y los parámetros ROI en config.py.")
        return
    rois_norm = norm.normalize_tl(rois)
    angles_norm = norm.normalize_angles(roi_angles)
    print(f"Rango TL: [{norm.tl_min:.2f}, {norm.tl_max:.2f}]")
    print(f"Rango ángulos: [{norm.angle_min:.2f}°, {norm.angle_max:.2f}°]")

    # ══════════════════════════════════════════════════════════════════════
    #  3. DATA AUGMENTATION
    # ══════════════════════════════════════════════════════════════════════
    if USE_AUGMENTATION:
        rois_aug, angles_aug = augment(rois_norm, angles_norm)
        extents_aug = list(roi_extents)
        for ext in roi_extents:
            extents_aug.extend([ext, ext, ext])
        print(f"Data augmentation activado: {len(rois_aug)} muestras (x{len(rois_aug)//len(rois_norm)})")
    else:
        rois_aug = rois_norm.copy()
        angles_aug = angles_norm.copy()
        extents_aug = list(roi_extents)
        print(f"Data augmentation desactivado: {len(rois_aug)} muestras (sin aumentar)")
    idx_shuf = np.random.permutation(len(rois_aug))
    rois_aug = rois_aug[idx_shuf]
    angles_aug = angles_aug[idx_shuf]
    extents_aug = np.array(extents_aug, dtype=np.float32)[idx_shuf]

    # Tensores en CPU — cada batch se mueve a GPU en el bucle de entrenamiento
    tensor_imgs = torch.from_numpy(rois_aug[:, np.newaxis, :, :])
    tensor_angs = torch.from_numpy(angles_aug[:, np.newaxis])
    tensor_extents = torch.from_numpy(extents_aug)
    use_pin = device.type == "cuda"
    loader = DataLoader(
        TensorDataset(tensor_imgs, tensor_angs, tensor_extents),
        batch_size=BATCH_SIZE, shuffle=True,
        pin_memory=use_pin, num_workers=0,
    )
    print(f"Batches por época: {len(loader)}")

    # ══════════════════════════════════════════════════════════════════════
    #  4. ENTRENAMIENTO CON MIXED PRECISION
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  ENTRENANDO UNet Autoencoder CONDICIONAL")
    print("=" * 60)

    model = ConditionalUNetAE(dropout_prob=MODEL_DROPOUT).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.L1Loss()
    scaler = None
    #scaler = GradScaler("cuda") if device.type == "cuda" else None

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros entrenables: {n_params:,}")

    history = {"loss": [], "data_loss": [], "physics_loss": [], "interface_loss": []}
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        ep_loss, ep_data, ep_phys, ep_iface, n_batches = 0.0, 0.0, 0.0, 0.0, 0

        for batch_img, batch_ang, batch_extents in loader:
            # Mover batch a GPU (non-blocking si pin_memory=True)
            batch_img = batch_img.to(device, non_blocking=use_pin)
            batch_ang = batch_ang.to(device, non_blocking=use_pin)
            batch_extents = batch_extents.to(device, non_blocking=use_pin)
            batch_in = batch_img
            known_mask = None

            if enable_train_inpaint:
                known_mask_np = generate_inpainting_known_masks(
                    batch_size=batch_img.shape[0],
                    height=batch_img.shape[2],
                    width=batch_img.shape[3],
                    min_holes=INPAINT_MIN_HOLES,
                    max_holes=INPAINT_MAX_HOLES,
                    min_hole_ratio=INPAINT_MIN_HOLE_RATIO,
                    max_hole_ratio=INPAINT_MAX_HOLE_RATIO,
                    keep_full_prob=INPAINT_KEEP_FULL_PROB,
                )
                known_mask = torch.from_numpy(known_mask_np).to(device, non_blocking=use_pin)
                batch_in_np = apply_inpainting_mask(
                    batch_img.detach().cpu().numpy(),
                    known_mask_np,
                    fill_value=INPAINT_FILL_VALUE,
                )
                batch_in = torch.from_numpy(batch_in_np).to(device, non_blocking=use_pin)

            optimizer.zero_grad()

            if scaler is not None:
                with autocast("cuda"):
                    recon = model(
                        batch_in,
                        batch_ang,
                        known_mask=known_mask if INPAINT_PRESERVE_KNOWN else None,
                    )
                    if enable_train_inpaint:
                        hole_mask = 1.0 - known_mask
                        loss_holes = (torch.abs(recon - batch_img) * hole_mask).sum() / (hole_mask.sum() + 1e-8)
                        loss_known = (torch.abs(recon - batch_img) * known_mask).sum() / (known_mask.sum() + 1e-8)
                        data_loss = loss_holes + INPAINT_LOSS_KNOWN_WEIGHT * loss_known
                    else:
                        data_loss = criterion(recon, batch_img)

                    physics_loss, interface_loss = compute_physics_losses(
                        recon, batch_extents, norm, device,
                    )
                    physics_total = physics_loss + INTERFACE_WEIGHT * interface_loss
                    loss = DATA_WEIGHT * data_loss + PHYSICS_WEIGHT * physics_total
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                recon = model(
                    batch_in,
                    batch_ang,
                    known_mask=known_mask if INPAINT_PRESERVE_KNOWN else None,
                )
                if enable_train_inpaint:
                    hole_mask = 1.0 - known_mask
                    loss_holes = (torch.abs(recon - batch_img) * hole_mask).sum() / (hole_mask.sum() + 1e-8)
                    loss_known = (torch.abs(recon - batch_img) * known_mask).sum() / (known_mask.sum() + 1e-8)
                    data_loss = loss_holes + INPAINT_LOSS_KNOWN_WEIGHT * loss_known
                else:
                    data_loss = criterion(recon, batch_img)

                physics_loss, interface_loss = compute_physics_losses(
                    recon, batch_extents, norm, device,
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

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            eta = elapsed / epoch * (EPOCHS - epoch)
            print(
                f"  Epoca {epoch:3d}/{EPOCHS} | loss={avg_loss:.6f} "
                f"data={avg_data:.6f} phys={avg_phys:.6f} iface={avg_iface:.6f} "
                f"| Elapsed: {format_seconds(elapsed)} ETA: {format_seconds(eta)}"
            )

    total_time = time.time() - t0
    print(f"\nEntrenamiento completado en {format_seconds(total_time)}")

    # ══════════════════════════════════════════════════════════════════════
    #  5. CURVA DE ENTRENAMIENTO
    # ══════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["loss"], label="total", linewidth=1.2)
    ax.plot(history["data_loss"], label="data", linewidth=1.2)
    if PHYSICS_WEIGHT > 0:
        ax.plot(history["physics_loss"], label="physics", linewidth=1.2)
        ax.plot(history["interface_loss"], label="interface", linewidth=1.2)
    ax.set_title("Training loss - UNet PINN")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(TRAIN_PNG_FOLDER, "training_curve_unet_ae.png"), dpi=150)
    plt.close(fig)
    print("Curva de entrenamiento guardada.")

    # ══════════════════════════════════════════════════════════════════════
    #  6. EXPORTAR ROIs ORIGINALES COMO .mat
    # ══════════════════════════════════════════════════════════════════════
    print("Exportando ROIs originales como .mat...")
    for j in range(len(rois_norm)):
        roi_tl = norm.denormalize_tl(rois_norm[j])
        roi_mat_path = os.path.join(TRAIN_MAT_FOLDER, f"roi_original_{roi_angles[j]:+.2f}.mat")
        save_mat(roi_mat_path, roi_tl, extent=roi_extents[j])
    print(f"  {len(rois_norm)} ROIs exportadas en {TRAIN_MAT_FOLDER}")

    # ══════════════════════════════════════════════════════════════════════
    #  7. COMPARACIÓN: ORIGINAL vs RECONSTRUCCIÓN (TODOS los planos)
    # ══════════════════════════════════════════════════════════════════════
    n_compare = len(rois_norm)
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
                sample = torch.from_numpy(
                    rois_norm[j][np.newaxis, np.newaxis, :, :]
                ).to(device)
                cond = torch.tensor(
                    [[angles_norm[j]]], dtype=torch.float32, device=device
                )
                recon = model(sample, cond)
                recon_np = recon.cpu().squeeze().numpy()

                orig_tl = norm.denormalize_tl(rois_norm[j])
                recon_tl = norm.denormalize_tl(np.clip(recon_np, 0, 1))
                errors = compute_error_metrics(orig_tl, recon_tl)
                angle_str = f"{roi_angles[j]:.2f}"
                ext = roi_extents[j]

                im0 = axes[0, c].imshow(
                    orig_tl, cmap="jet", aspect="auto", origin="lower",
                    vmin=norm.tl_min, vmax=norm.tl_max, extent=ext,
                )
                fig.colorbar(im0, ax=axes[0, c], shrink=0.6, label="TL (dB)")
                axes[0, c].set_title(f"Original -- {angle_str}", fontsize=9)
                axes[0, c].set_xlabel("X"); axes[0, c].set_ylabel("Z")

                im1 = axes[1, c].imshow(
                    recon_tl, cmap="jet", aspect="auto", origin="lower",
                    vmin=norm.tl_min, vmax=norm.tl_max, extent=ext,
                )
                fig.colorbar(im1, ax=axes[1, c], shrink=0.6, label="TL (dB)")
                axes[1, c].set_title(f"Reconstruccion -- {angle_str}", fontsize=9)
                axes[1, c].set_xlabel("X"); axes[1, c].set_ylabel("Z")
                im2 = axes[2, c].imshow(
                    np.abs(orig_tl - recon_tl), cmap="turbo", aspect="auto", origin="lower",
                    vmin=0, vmax=errors['max_error'] or 1e-6, extent=ext,
                )
                fig.colorbar(im2, ax=axes[2, c], shrink=0.6, label="|Error| (dB)")
                
                # Mostrar MAE, MAPE, RMSE y Max
                mape_str = f"{errors['mape']:.1f}%" if errors['mape'] != np.inf else "undef"
                axes[2, c].set_title(
                    f"|Error| -- {angle_str}\nMAE={errors['mae']:.2f} dB  "
                    f"MAPE={mape_str}  RMSE={errors['rmse']:.2f} dB  Max={errors['max_error']:.2f} dB",
                    fontsize=7,
                )
                axes[2, c].set_xlabel("X"); axes[2, c].set_ylabel("Z")

            fig.suptitle(
                f"Originales vs Reconstrucciones — Cond. UNet AE  "
                f"(pág. {page+1}/{n_pages})", fontsize=14,
            )
            fig.tight_layout()
            fig.savefig(
                os.path.join(TRAIN_PNG_FOLDER, f"comparacion_original_vs_recon_{page+1:02d}.png"),
                dpi=150,
            )
            plt.close(fig)
            print(f"  Comparación pág. {page+1}/{n_pages} guardada ({cols_this} planos).")

    print("Comparación original vs reconstrucción guardada.")

    # ══════════════════════════════════════════════════════════════════════
    #  8. GUARDAR MODELO
    # ══════════════════════════════════════════════════════════════════════
    torch.save({
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
        "interface_z": INTERFACE_Z,
        "interface_eps": INTERFACE_EPS,
        "output_is_tl": OUTPUT_IS_TL,
        "tl_ref_pressure": TL_REF_PRESSURE,
        "tl_db_sign": TL_DB_SIGN,
        "tl_db_offset": TL_DB_OFFSET,
        "tl_db_clamp_min": TL_DB_CLAMP_MIN,
        "tl_db_clamp_max": TL_DB_CLAMP_MAX,
        **norm.state_dict(),
    }, MODEL_PATH)
    print(f"\nModelo guardado en: {MODEL_PATH}")

    # ══════════════════════════════════════════════════════════════════════
    #  9. VALIDACIÓN CON PLANOS NO VISTOS
    # ══════════════════════════════════════════════════════════════════════
    val_rois, val_angles, val_extents = load_validation_rois(
        VALIDATION_FOLDER, ROI_HEIGHT, ROI_WIDTH,
        roi_mode=ROI_MODE, roi_corner=ROI_CORNER,
    )
    if val_rois is not None and len(val_rois) > 0:
        create_output_dirs(VALIDATION_PNG_FOLDER, VALIDATION_MAT_FOLDER)

        # Normalizar ROIs de validación con el Normalizer del entrenamiento
        val_rois_norm = norm.normalize_tl(val_rois)

        # Ángulos de entrenamiento ordenados para buscar vecinos
        train_angles_sorted = np.sort(np.array(roi_angles))

        # Tensores de entrenamiento para usar como semillas
        base_samples = torch.from_numpy(
            rois_norm[:, np.newaxis, :, :]
        ).to(device)

        print(f"\nValidando con {len(val_rois)} planos no vistos...")

        n_val = len(val_rois)
        n_rows_per_page = min(n_val, 6)
        n_pages = int(np.ceil(n_val / n_rows_per_page))

        all_maes = []
        all_maxes = []
        all_gaps = []
        all_val_angles_plot = []
        all_mapes = []
        all_rmses = []

        roi_angles_arr = np.array(roi_angles)

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
                    ext = val_extents[j]

                    # --- Vecinos de entrenamiento (menor y mayor) ---
                    lower_mask = train_angles_sorted[train_angles_sorted <= val_angle]
                    upper_mask = train_angles_sorted[train_angles_sorted >= val_angle]
                    ang_lower = float(lower_mask[-1]) if len(lower_mask) > 0 else None
                    ang_upper = float(upper_mask[0])  if len(upper_mask) > 0 else None

                    diff_lower = abs(val_angle - ang_lower) if ang_lower is not None else None
                    diff_upper = abs(ang_upper - val_angle) if ang_upper is not None else None

                    if ang_lower is not None and ang_upper is not None:
                        gap = abs(ang_upper - ang_lower)
                    else:
                        gap = None

                    # --- Generar predicción usando semilla más cercana ---
                    dists = np.abs(roi_angles_arr - val_angle)
                    idx_seed = np.argmin(dists)
                    seed = base_samples[idx_seed:idx_seed + 1]

                    gen_tl = generate_from_seed(
                        model, seed, val_angle, norm, NOISE_STD, device,
                    )

                    if enable_infer_inpaint:
                        gen_norm = norm.normalize_tl(gen_tl).astype(np.float32)
                        known_mask_np = generate_inpainting_known_masks(
                            batch_size=1,
                            height=gen_norm.shape[0],
                            width=gen_norm.shape[1],
                            min_holes=INPAINT_MIN_HOLES,
                            max_holes=INPAINT_MAX_HOLES,
                            min_hole_ratio=INPAINT_MIN_HOLE_RATIO,
                            max_hole_ratio=INPAINT_MAX_HOLE_RATIO,
                            keep_full_prob=INPAINT_KEEP_FULL_PROB,
                        )
                        masked_norm = apply_inpainting_mask(
                            gen_norm[np.newaxis, np.newaxis, :, :],
                            known_mask_np,
                            fill_value=INPAINT_FILL_VALUE,
                        )
                        masked_tensor = torch.from_numpy(masked_norm).to(device)
                        known_mask_t = torch.from_numpy(known_mask_np).to(device)
                        cond_val = torch.tensor(
                            [[norm.normalize_angle(val_angle)]],
                            dtype=torch.float32, device=device,
                        )
                        inpaint_norm = model(
                            masked_tensor,
                            cond_val,
                            known_mask=known_mask_t if INPAINT_PRESERVE_KNOWN else None,
                        ).cpu().squeeze().numpy()
                        gen_tl = norm.denormalize_tl(np.clip(inpaint_norm, 0, 1))

                    # --- ROI real del plano de validación ---
                    real_tl = norm.denormalize_tl(val_rois_norm[j])

                    # --- Error ---
                    diff_map = np.abs(gen_tl - real_tl)
                    errors = compute_error_metrics(real_tl, gen_tl)
                    mae_val = errors['mae']
                    max_val = errors['max_error']
                    rmse_val = errors['rmse']
                    mape_val = errors['mape']

                    # Imprimir info de vecinos y métricas
                    lo_str = (f"{ang_lower:+.2f}\u00b0 (dif={diff_lower:.2f}\u00b0)"
                              if ang_lower is not None else "---")
                    hi_str = (f"{ang_upper:+.2f}\u00b0 (dif={diff_upper:.2f}\u00b0)"
                              if ang_upper is not None else "---")
                    gap_str = f"{gap:.2f}\u00b0" if gap is not None else "---"
                    mape_str = f"{mape_val:.2f}%" if mape_val != np.inf else "undef"
                    print(
                        f"  Angulo val {val_angle:+.2f}\u00b0  |  "
                        f"vecino inf: {lo_str}  |  vecino sup: {hi_str}  |  "
                        f"gap: {gap_str}  |  MAE: {mae_val:.2f} dB  |  "
                        f"RMSE: {rmse_val:.2f} dB  |  MAPE: {mape_str}  |  "
                        f"Max: {max_val:.2f} dB"
                    )
                    all_maes.append(mae_val)
                    all_maxes.append(max_val)
                    all_gaps.append(gap)
                    all_val_angles_plot.append(val_angle)
                    all_mapes.append(mape_val)
                    all_rmses.append(rmse_val)

                    # --- Exportar .mat de la generación ---
                    mat_path = os.path.join(
                        VALIDATION_MAT_FOLDER,
                        f"val_generado_{val_angle:+.2f}.mat",
                    )
                    save_mat(mat_path, gen_tl, extent=ext)

                    # --- Texto de vecinos para titulo (claro y legible) ---
                    neigh_parts = []
                    if ang_lower is not None:
                        neigh_parts.append(
                            f"Vecino inf: {ang_lower:+.1f}\u00b0 (\u0394={diff_lower:.1f}\u00b0)"
                        )
                    if ang_upper is not None:
                        neigh_parts.append(
                            f"Vecino sup: {ang_upper:+.1f}\u00b0 (\u0394={diff_upper:.1f}\u00b0)"
                        )
                    if gap is not None:
                        neigh_parts.append(f"Gap total: {gap:.1f}\u00b0")
                    neigh_txt = "  |  ".join(neigh_parts)

                    # --- 3 subplots en horizontal por fila ---
                    angle_str = f"{val_angle:.2f}\u00b0"

                    im0 = axes[r, 0].imshow(
                        real_tl, cmap="jet", aspect="auto", origin="lower",
                        vmin=norm.tl_min, vmax=norm.tl_max, extent=ext,
                    )
                    fig.colorbar(im0, ax=axes[r, 0], shrink=0.6, label="TL (dB)")
                    axes[r, 0].set_title(
                        f"Real -- {angle_str}\n{neigh_txt}", fontsize=8,
                    )
                    axes[r, 0].set_xlabel("X (m)"); axes[r, 0].set_ylabel("Z (m)")

                    im1 = axes[r, 1].imshow(
                        gen_tl, cmap="jet", aspect="auto", origin="lower",
                        vmin=norm.tl_min, vmax=norm.tl_max, extent=ext,
                    )
                    fig.colorbar(im1, ax=axes[r, 1], shrink=0.6, label="TL (dB)")
                    axes[r, 1].set_title(
                        f"Generado -- {angle_str}\n(semilla {roi_angles[idx_seed]:.2f}\u00b0)",
                        fontsize=8,
                    )
                    axes[r, 1].set_xlabel("X (m)"); axes[r, 1].set_ylabel("Z (m)")

                    im2 = axes[r, 2].imshow(
                        diff_map, cmap="turbo", aspect="auto", origin="lower",
                        vmin=0, vmax=max_val or 1e-6, extent=ext,
                    )
                    fig.colorbar(im2, ax=axes[r, 2], shrink=0.6, label="|Error| (dB)")
                    mape_str = f"{mape_val:.1f}%" if mape_val != np.inf else "undef"
                    axes[r, 2].set_title(
                        f"|Error| -- {angle_str}\nMAE={mae_val:.2f} dB  "
                        f"MAPE={mape_str}  RMSE={rmse_val:.2f} dB  Max={max_val:.2f} dB",
                        fontsize=7,
                    )
                    axes[r, 2].set_xlabel("X (m)"); axes[r, 2].set_ylabel("Z (m)")

                fig.suptitle(
                    f"Validacion: Real vs Generado -- UNet AE  "
                    f"(pag. {page+1}/{n_pages})", fontsize=14,
                )
                fig.tight_layout()
                fig.savefig(
                    os.path.join(
                        VALIDATION_PNG_FOLDER,
                        f"validacion_real_vs_gen_{page+1:02d}.png",
                    ),
                    dpi=150,
                )
                plt.close(fig)
                print(f"  Validacion pag. {page+1}/{n_pages} guardada ({rows_this} planos).")

        # --- Grafico: Error vs Gap entre vecinos de entrenamiento ---
        # Solo incluir planos que tienen ambos vecinos (gap definido)
        plot_data = [
            (g, m, mx, r, mp, a)
            for g, m, mx, r, mp, a in zip(all_gaps, all_maes, all_maxes, all_rmses, all_mapes, all_val_angles_plot)
            if g is not None
        ]
        if plot_data:
            # Ordenar por gap para que las lineas conecten de menor a mayor
            plot_data.sort(key=lambda d: d[0])
            gaps_arr   = np.array([d[0] for d in plot_data])
            maes_arr   = np.array([d[1] for d in plot_data])
            maxes_arr  = np.array([d[2] for d in plot_data])
            rmses_arr  = np.array([d[3] for d in plot_data])
            mapes_arr  = np.array([d[4] for d in plot_data])
            labels_arr = [f"{d[5]:+.2f}\u00b0" for d in plot_data]

            fig_err, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # --- Subplot 1: MAE, RMSE, Max vs Gap ---
            color_mae = "#1f77b4"
            color_rmse = "#ff7f0e"
            color_max = "#d62728"

            ax1.plot(gaps_arr, maes_arr, color=color_mae, linewidth=1.2, alpha=0.6, zorder=2)
            ax1.plot(gaps_arr, rmses_arr, color=color_rmse, linewidth=1.2, alpha=0.6, zorder=2)
            ax1.plot(gaps_arr, maxes_arr, color=color_max, linewidth=1.2, alpha=0.6, linestyle="--", zorder=2)

            ax1.scatter(gaps_arr, maes_arr, color=color_mae, s=50, zorder=3, label="MAE (dB)")
            ax1.scatter(gaps_arr, rmses_arr, color=color_rmse, s=50, zorder=3, label="RMSE (dB)")
            ax1.scatter(gaps_arr, maxes_arr, color=color_max, s=50, marker="^", zorder=3, label="Max error (dB)")

            ax1.set_xlabel("Gap entre vecinos de entrenamiento (\u00b0)", fontsize=10)
            ax1.set_ylabel("Error (dB)", fontsize=10)
            ax1.set_title("Errores Absolutos vs Distancia Angular", fontsize=11)
            ax1.legend(fontsize=9, loc="upper left")
            ax1.grid(True, alpha=0.3)

            # --- Subplot 2: MAPE vs Gap ---
            color_mape = "#2ca02c"
            ax2.plot(gaps_arr, mapes_arr, color=color_mape, linewidth=1.2, alpha=0.6, zorder=2)
            ax2.scatter(gaps_arr, mapes_arr, color=color_mape, s=50, zorder=3, label="MAPE (%)")

            # Etiquetar puntos con ángulos
            for xi, yi, lbl in zip(gaps_arr, mapes_arr, labels_arr):
                ax2.annotate(lbl, (xi, yi), textcoords="offset points",
                             xytext=(5, 5), fontsize=7, color=color_mape)

            ax2.set_xlabel("Gap entre vecinos de entrenamiento (\u00b0)", fontsize=10)
            ax2.set_ylabel("MAPE (%)", fontsize=10)
            ax2.set_title("Error Relativo (MAPE) vs Distancia Angular", fontsize=11)
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)
            
            fig_err.suptitle("Análisis de Validación: Error vs Distancia Angular\n"
                             "entre los dos vecinos de entrenamiento más cercanos",
                             fontsize=12)
            fig_err.tight_layout()
            err_path = os.path.join(VALIDATION_PNG_FOLDER, "error_vs_gap.png")
            fig_err.savefig(err_path, dpi=150)
            plt.close(fig_err)
            print(f"  Grafico error vs gap guardado en: {err_path}")

        # Resumen
        mean_mae = np.mean(all_maes)
        mean_rmse = np.mean(all_rmses)
        mean_mape = np.mean([m for m in all_mapes if m != np.inf])
        mean_max = np.mean(all_maxes) if len(all_maxes) > 0 else np.inf
        
        print(f"\n  Validacion completada:")
        print(f"    MAE medio   = {mean_mae:.2f} dB")
        print(f"    RMSE medio  = {mean_rmse:.2f} dB")
        print(f"    MAPE medio  = {mean_mape:.1f}%")
        print(f"    Max medio   = {mean_max:.2f} dB")
        print(f"    sobre {len(all_maes)} planos.")
        print(f"  Resultados en: {VALIDATION_PNG_FOLDER}")
        print(f"                 {VALIDATION_MAT_FOLDER}")
    else:
        print("\n  [i] No se encontraron planos de validacion en "
              f"{VALIDATION_FOLDER}")
        print("      Para usarlo, coloca .mat con 'PlaneAngle' en el nombre.")

    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    else:
        peak_vram_mb = 0.0

    summary = {
        "status": "ok",
        "mean_mae": float(mean_mae) if 'mean_mae' in locals() else float('inf'),
        "mean_rmse": float(mean_rmse) if 'mean_rmse' in locals() else float('inf'),
        "mean_mape": float(mean_mape) if 'mean_mape' in locals() else float('inf'),
        "mean_max_error": float(np.mean(all_maxes)) if 'all_maxes' in locals() and len(all_maxes) > 0 else float('inf'),
        "peak_vram_mb": float(peak_vram_mb),
        "epochs": int(EPOCHS),
        "batch_size": int(BATCH_SIZE),
        "learning_rate": float(LEARNING_RATE),
        "use_augmentation": bool(USE_AUGMENTATION),
        "inpaint_mode": str(INPAINT_MODE),
    }

    print("\n" + "=" * 60)
    print(f"  LISTO! Resultados en:\n  {TRAIN_PNG_FOLDER}\n  {TRAIN_MAT_FOLDER}")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    main()
