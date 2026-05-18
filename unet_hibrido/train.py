import os
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast
 
from model import ConditionalUNetAE, PINN
from data_utils import (
    load_all_rois, load_validation_rois,
    Normalizer, CoordNormalizer, augment, save_mat, adaptive_figsize,
    generate_from_seed, compute_error_metrics,
    generate_inpainting_known_masks, apply_inpainting_mask,
    sample_points_from_rois, get_global_bounds,
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
    POINTS_PER_ROI, PINN_DATA_BATCH_SIZE, PINN_DATA_WEIGHT,
    PHYSICS_BATCH_SIZE, DATA_WEIGHT, PHYSICS_WEIGHT,
    INTERFACE_BATCH_SIZE, INTERFACE_WEIGHT,
    PDE_TYPE, HELMHOLTZ_K,
    FREQUENCY_HZ, AIR_SOUND_SPEED, AIR_DENSITY,
    WATER_SOUND_SPEED, WATER_DENSITY,
    AIR_ABOVE_INTERFACE, INTERFACE_Z, INTERFACE_EPS,
    TL_REF_PRESSURE, OUTPUT_IS_TL,
    TL_DB_SIGN, TL_DB_OFFSET, TL_DB_CLAMP_MIN, TL_DB_CLAMP_MAX,
    PINN_HIDDEN_DIM, PINN_NUM_LAYERS, PINN_ACTIVATION, PINN_DROPOUT,
    DATA_FOLDER, MODEL_PATH,
    VALIDATION_FOLDER,
    TRAIN_PNG_FOLDER, TRAIN_MAT_FOLDER,
    VALIDATION_PNG_FOLDER, VALIDATION_MAT_FOLDER,
    create_output_dirs,
) 
from validation import run_validation


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
    return float(ref_pressure) * torch.exp(-tl_db * scale)


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
    rng = np.random.default_rng(SEED)
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
        rois_aug, angles_aug_norm = augment(rois_norm, angles_norm)
        extents_aug = list(roi_extents)
        angles_aug_raw = list(roi_angles)
        for angle, ext in zip(roi_angles, roi_extents):
            angles_aug_raw.extend([angle, angle, angle])
            extents_aug.extend([ext, ext, ext])
        print(f"Data augmentation activado: {len(rois_aug)} muestras (x{len(rois_aug)//len(rois_norm)})")
    else:
        rois_aug = rois_norm.copy()
        angles_aug_norm = angles_norm.copy()
        angles_aug_raw = list(roi_angles)
        extents_aug = list(roi_extents)
        print(f"Data augmentation desactivado: {len(rois_aug)} muestras (sin aumentar)")
    idx_shuf = np.random.permutation(len(rois_aug))
    rois_aug = rois_aug[idx_shuf]
    angles_aug_norm = angles_aug_norm[idx_shuf]
    angles_aug_raw = np.array(angles_aug_raw, dtype=np.float32)[idx_shuf]
    extents_aug = np.array(extents_aug, dtype=np.float32)[idx_shuf]

    bounds = get_global_bounds(extents_aug)
    interface_z = INTERFACE_Z
    if interface_z is None:
        interface_z = 0.5 * (bounds[2] + bounds[3])
        print(f"Auto interface_z set to {interface_z:.3f}")
    coord_norm = CoordNormalizer(
        bounds[0], bounds[1], bounds[2], bounds[3],
        norm.angle_min, norm.angle_max,
    )

    # Tensores en CPU — cada batch se mueve a GPU en el bucle de entrenamiento
    tensor_imgs = torch.from_numpy(rois_aug[:, np.newaxis, :, :])
    tensor_angs = torch.from_numpy(angles_aug_norm[:, np.newaxis])
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
    print("  ENTRENANDO UNet Autoencoder CONDICIONAL + PINN (hibrido)")
    print("=" * 60)

    unet = ConditionalUNetAE(dropout_prob=MODEL_DROPOUT).to(device)
    pinn = PINN(
        hidden_dim=PINN_HIDDEN_DIM,
        num_layers=PINN_NUM_LAYERS,
        activation=PINN_ACTIVATION,
        dropout=PINN_DROPOUT,
    ).to(device)

    optimizer = optim.AdamW(
        list(unet.parameters()) + list(pinn.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    
    # Scheduler: CosineAnnealingLR for smooth convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    # Loss: L1 + alpha * MSE combination to penalize Max Error strongly
    criterion_l1 = nn.L1Loss()
    criterion_mse = nn.MSELoss()
    def combined_loss(pred, target):
        return criterion_l1(pred, target) + 0.5 * criterion_mse(pred, target)
    
    criterion = combined_loss
    scaler = None
    #scaler = GradScaler("cuda") if device.type == "cuda" else None

    n_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    n_params += sum(p.numel() for p in pinn.parameters() if p.requires_grad)
    print(f"Parámetros entrenables: {n_params:,}")

    history = {"loss": [], "data_loss": [], "physics_loss": [], "interface_loss": []}
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        unet.train()
        pinn.train()
        ep_loss, ep_data, ep_phys, ep_iface, ep_pinn_data, n_batches = 0.0, 0.0, 0.0, 0.0, 0.0, 0

        coords_pool = None
        targets_pool = None
        if PINN_DATA_WEIGHT > 0 or PHYSICS_WEIGHT > 0:
            xs, zs, angs, tls = sample_points_from_rois(
                rois_aug, angles_aug_raw, extents_aug,
                POINTS_PER_ROI, rng,
            )
            x_n, z_n, a_n = coord_norm.normalize(xs, zs, angs)
            coords_pool = np.stack([x_n, z_n, a_n], axis=1).astype(np.float32)
            targets_pool = tls.astype(np.float32).reshape(-1, 1)

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
                    recon = unet(
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

                    pinn_data_loss = torch.tensor(0.0, device=device)
                    if PINN_DATA_WEIGHT > 0 and coords_pool is not None:
                        n_data = min(PINN_DATA_BATCH_SIZE, coords_pool.shape[0])
                        idx = rng.choice(coords_pool.shape[0], size=n_data, replace=False)
                        coords_t = torch.from_numpy(coords_pool[idx]).to(device)
                        targets_t = torch.from_numpy(targets_pool[idx]).to(device)
                        pred = pinn(coords_t)
                        pinn_data_loss = criterion(pred, targets_t)

                    physics_loss = torch.tensor(0.0, device=device)
                    interface_loss = torch.tensor(0.0, device=device)
                    if PHYSICS_WEIGHT > 0 and PDE_TYPE != "none":
                        n_phys = PHYSICS_BATCH_SIZE if PHYSICS_BATCH_SIZE > 0 else PINN_DATA_BATCH_SIZE
                        if n_phys > 0:
                            x_c, z_c, a_c = sample_collocation_points(
                                n_phys, bounds, norm.angle_min, norm.angle_max, rng,
                            )
                            x_cn, z_cn, a_cn = coord_norm.normalize(x_c, z_c, a_c)
                            coords_c = np.stack([x_cn, z_cn, a_cn], axis=1).astype(np.float32)
                            coords_c = torch.from_numpy(coords_c).to(device)
                            coords_c.requires_grad_(True)

                            residual = compute_pde_residual(
                                pinn, coords_c, coord_norm, norm,
                                PDE_TYPE, HELMHOLTZ_K,
                                FREQUENCY_HZ, AIR_SOUND_SPEED, WATER_SOUND_SPEED,
                                bounds, interface_z, AIR_ABOVE_INTERFACE,
                                OUTPUT_IS_TL, TL_REF_PRESSURE,
                                TL_DB_SIGN, TL_DB_OFFSET, TL_DB_CLAMP_MIN, TL_DB_CLAMP_MAX,
                            )
                            physics_loss = torch.mean(residual ** 2)

                        if (PDE_TYPE == "two_layer_helmholtz" and INTERFACE_WEIGHT > 0
                                and INTERFACE_BATCH_SIZE > 0):
                            interface_loss = compute_interface_loss(
                                pinn, coord_norm, norm, bounds,
                                norm.angle_min, norm.angle_max, rng,
                                INTERFACE_BATCH_SIZE, interface_z, INTERFACE_EPS,
                                AIR_ABOVE_INTERFACE, AIR_DENSITY, WATER_DENSITY,
                                OUTPUT_IS_TL, TL_REF_PRESSURE,
                                TL_DB_SIGN, TL_DB_OFFSET, TL_DB_CLAMP_MIN, TL_DB_CLAMP_MAX,
                                device,
                            )

                    physics_total = physics_loss + INTERFACE_WEIGHT * interface_loss
                    loss = (
                        DATA_WEIGHT * data_loss
                        + PINN_DATA_WEIGHT * pinn_data_loss
                        + PHYSICS_WEIGHT * physics_total
                    )

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                recon = unet(
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

                pinn_data_loss = torch.tensor(0.0, device=device)
                if PINN_DATA_WEIGHT > 0 and coords_pool is not None:
                    n_data = min(PINN_DATA_BATCH_SIZE, coords_pool.shape[0])
                    idx = rng.choice(coords_pool.shape[0], size=n_data, replace=False)
                    coords_t = torch.from_numpy(coords_pool[idx]).to(device)
                    targets_t = torch.from_numpy(targets_pool[idx]).to(device)
                    pred = pinn(coords_t)
                    pinn_data_loss = criterion(pred, targets_t)

                physics_loss = torch.tensor(0.0, device=device)
                interface_loss = torch.tensor(0.0, device=device)
                if PHYSICS_WEIGHT > 0 and PDE_TYPE != "none":
                    n_phys = PHYSICS_BATCH_SIZE if PHYSICS_BATCH_SIZE > 0 else PINN_DATA_BATCH_SIZE
                    if n_phys > 0:
                        x_c, z_c, a_c = sample_collocation_points(
                            n_phys, bounds, norm.angle_min, norm.angle_max, rng,
                        )
                        x_cn, z_cn, a_cn = coord_norm.normalize(x_c, z_c, a_c)
                        coords_c = np.stack([x_cn, z_cn, a_cn], axis=1).astype(np.float32)
                        coords_c = torch.from_numpy(coords_c).to(device)
                        coords_c.requires_grad_(True)

                        residual = compute_pde_residual(
                            pinn, coords_c, coord_norm, norm,
                            PDE_TYPE, HELMHOLTZ_K,
                            FREQUENCY_HZ, AIR_SOUND_SPEED, WATER_SOUND_SPEED,
                            bounds, interface_z, AIR_ABOVE_INTERFACE,
                            OUTPUT_IS_TL, TL_REF_PRESSURE,
                            TL_DB_SIGN, TL_DB_OFFSET, TL_DB_CLAMP_MIN, TL_DB_CLAMP_MAX,
                        )
                        physics_loss = torch.mean(residual ** 2)

                    if (PDE_TYPE == "two_layer_helmholtz" and INTERFACE_WEIGHT > 0
                            and INTERFACE_BATCH_SIZE > 0):
                        interface_loss = compute_interface_loss(
                            pinn, coord_norm, norm, bounds,
                            norm.angle_min, norm.angle_max, rng,
                            INTERFACE_BATCH_SIZE, interface_z, INTERFACE_EPS,
                            AIR_ABOVE_INTERFACE, AIR_DENSITY, WATER_DENSITY,
                            OUTPUT_IS_TL, TL_REF_PRESSURE,
                            TL_DB_SIGN, TL_DB_OFFSET, TL_DB_CLAMP_MIN, TL_DB_CLAMP_MAX,
                            device,
                        )

                physics_total = physics_loss + INTERFACE_WEIGHT * interface_loss
                loss = (
                    DATA_WEIGHT * data_loss
                    + PINN_DATA_WEIGHT * pinn_data_loss
                    + PHYSICS_WEIGHT * physics_total
                )
                loss.backward()
                optimizer.step()

            ep_loss += loss.item()
            ep_data += data_loss.item()
            ep_phys += physics_loss.item()
            ep_iface += interface_loss.item()
            ep_pinn_data += pinn_data_loss.item()
            n_batches += 1

        scheduler.step()  # Update Learning Rate at the end of the epoch

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
            avg_pinn_data = ep_pinn_data / n_batches
            print(
                f"  Epoca {epoch:3d}/{EPOCHS} | loss={avg_loss:.6f} "
                f"data={avg_data:.6f} pinn={avg_pinn_data:.6f} "
                f"phys={avg_phys:.6f} iface={avg_iface:.6f} "
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
    ax.set_title("Training loss - UNet Hybrid")
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

    unet.eval()
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
                recon = unet(sample, cond)
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
        "unet_state_dict": unet.state_dict(),
        "pinn_state_dict": pinn.state_dict(),
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
        **coord_norm.state_dict(),
        **norm.state_dict(),
    }, MODEL_PATH)
    print(f"\nModelo guardado en: {MODEL_PATH}")

    # ══════════════════════════════════════════════════════════════════════
    #  9. VALIDACIÓN CON PLANOS NO VISTOS
    # ══════════════════════════════════════════════════════════════════════
    validation_summary = run_validation(
        model_path=MODEL_PATH,
        data_folder=DATA_FOLDER,
        validation_folder=VALIDATION_FOLDER,
        roi_height=ROI_HEIGHT,
        roi_width=ROI_WIDTH,
        rois_per_plane=ROIS_PER_PLANE,
        roi_mode=ROI_MODE,
        roi_corner=ROI_CORNER,
        noise_std=NOISE_STD,
        output_png_folder=VALIDATION_PNG_FOLDER,
        output_mat_folder=VALIDATION_MAT_FOLDER,
        seed=SEED,
        inpaint_mode=INPAINT_MODE,
        inpaint_preserve_known=INPAINT_PRESERVE_KNOWN,
        inpaint_min_holes=INPAINT_MIN_HOLES,
        inpaint_max_holes=INPAINT_MAX_HOLES,
        inpaint_min_hole_ratio=INPAINT_MIN_HOLE_RATIO,
        inpaint_max_hole_ratio=INPAINT_MAX_HOLE_RATIO,
        inpaint_keep_full_prob=INPAINT_KEEP_FULL_PROB,
        inpaint_fill_value=INPAINT_FILL_VALUE,
        inpaint_loss_known_weight=INPAINT_LOSS_KNOWN_WEIGHT,
        print_fn=print,
    )

    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    else:
        peak_vram_mb = 0.0

    summary = {
        "status": "ok",
        "mean_mae": float(validation_summary.get("mean_mae", float('inf'))),
        "mean_rmse": float(validation_summary.get("mean_rmse", float('inf'))),
        "mean_mape": float(validation_summary.get("mean_mape", float('inf'))),
        "mean_max_error": float(validation_summary.get("mean_max_error", float('inf'))),
        "validation_status": validation_summary.get("status", "unknown"),
        "validation_mean_mae": float(validation_summary.get("mean_mae", float('inf'))),
        "validation_mean_rmse": float(validation_summary.get("mean_rmse", float('inf'))),
        "validation_mean_mape": float(validation_summary.get("mean_mape", float('inf'))),
        "validation_mean_max_error": float(validation_summary.get("mean_max_error", float('inf'))),
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
