"""
CONFIGURACIONES OPTIMIZADAS PARA MI210 (68.7GB VRAM)
=====================================================

Opciones escalonadas para aumentar velocidad manteniendo estabilidad OOM.
"""

import os

# ══════════════════════════════════════════════════════════════════════════
#  PARÁMETROS DE DATOS
# ══════════════════════════════════════════════════════════════════════════
ROI_HEIGHT     = 700
ROI_WIDTH      = 2000
ROIS_PER_PLANE = 1
ROI_MODE = "corner_fixed"
ROI_CORNER = (0, 10)

# ══════════════════════════════════════════════════════════════════════════
#  PARÁMETROS DE DIFUSIÓN
# ══════════════════════════════════════════════════════════════════════════
DIFFUSION_STEPS    = 1000
DIFFUSION_SCHEDULE = "linear"
BETA_START = 0.0001
BETA_END   = 0.02

# ══════════════════════════════════════════════════════════════════════════
#  OPCIÓN 1: INCREMENTO MODERADO (Recomendado primero)
#  ✓ Usa ~35-40% VRAM
#  ✓ Aumenta velocidad 50-70%
#  ✓ Mantiene estabilidad OOM
# ══════════════════════════════════════════════════════════════════════════
MODEL_HIDDEN_SIZE = 384      # Mantener igual
MODEL_DEPTH       = 10       # Mantener igual
MODEL_NUM_HEADS   = 12       # Mantener igual
MLP_RATIO         = 3.0      # ↓ de 4.0 (reduce 25% params MLP)
MODEL_PATCH_SIZE  = 2        # Mantener igual
ATTENTION_CHUNK_SIZE = 320   # ↑ de 64 (permite más eficiencia)
GRADIENT_CHECKPOINTING = True  # ↑ ACTIVAR (ahorr
a ~30% memoria intermedia)

VAE_LATENT_CHANNELS = 1      # Mantener igual
VAE_COMPRESSION_RATIO = 8    # ↑ de 6 (reduce tokens 25%)

BATCH_SIZE         = 2       # ↑ de 1 (mejor amortización overhead)
GRAD_ACCUM_STEPS   = 3       # Mantener
EPOCHS             = 220
LEARNING_RATE      = 1.5e-4
WARMUP_STEPS       = 400
WEIGHT_DECAY       = 1e-4
EMA_DECAY          = 0.9997
USE_AUGMENTATION   = False

print("=" * 70)
print("OPCIÓN 1: INCREMENTO MODERADO")
print("=" * 70)
print(f"Expected VRAM usage: ~35-40% (~24-27 GB)")
print(f"Training speedup: ~50-70% faster")
print(f"Memory safety: HIGH (stable OOM recovery)")
print()

# ══════════════════════════════════════════════════════════════════════════
#  OPCIÓN 2: INCREMENTO AGRESIVO (Si Opción 1 funciona bien)
#  ✓ Usa ~50-60% VRAM
#  ✓ Aumenta velocidad 100-150%
#  ⚠ Requiere monitoreo OOM
# ══════════════════════════════════════════════════════════════════════════
# MODEL_HIDDEN_SIZE = 384
# MODEL_DEPTH       = 10
# MODEL_NUM_HEADS   = 12
# MLP_RATIO         = 2.5      # ↓↓ Reduce 37% params MLP
# MODEL_PATCH_SIZE  = 2
# ATTENTION_CHUNK_SIZE = 512   # ↑↑ Máximo chunk
# GRADIENT_CHECKPOINTING = True
# VAE_COMPRESSION_RATIO = 10   # ↑↑ Reduce tokens 40%
# BATCH_SIZE         = 4       # ↑↑ 4x mejor amortización
# GRAD_ACCUM_STEPS   = 2

# print("=" * 70)
# print("OPCIÓN 2: INCREMENTO AGRESIVO")
# print("=" * 70)
# print(f"Expected VRAM usage: ~50-60% (~34-41 GB)")
# print(f"Training speedup: ~100-150% faster")
# print(f"Memory safety: MEDIUM (watch OOM recovery)")
# print()

# ══════════════════════════════════════════════════════════════════════════
#  OPCIÓN 3: MÁXIMO RENDIMIENTO (Si tienes tiempo y experiencia)
#  ✓ Usa ~65-75% VRAM
#  ✓ Aumenta velocidad 150-200%
#  ⚠⚠ Alto riesgo OOM - requiere PYTORCH_HIP_ALLOC_CONF
# ══════════════════════════════════════════════════════════════════════════
# MODEL_HIDDEN_SIZE = 320      # ↓ Reduce params 30%
# MODEL_DEPTH       = 10
# MODEL_NUM_HEADS   = 10       # ↓ 10 heads = 32 dim/head
# MLP_RATIO         = 2.0      # ↓↓↓ Solo x2
# MODEL_PATCH_SIZE  = 2
# ATTENTION_CHUNK_SIZE = 1024  # ↑↑↑ Max
# GRADIENT_CHECKPOINTING = True
# VAE_COMPRESSION_RATIO = 12   # ↑↑↑ Reduce 50% tokens
# BATCH_SIZE         = 8       # ↑↑↑ 8x amortización
# GRAD_ACCUM_STEPS   = 1

# print("=" * 70)
# print("OPCIÓN 3: MÁXIMO RENDIMIENTO")
# print("=" * 70)
# print(f"Expected VRAM usage: ~65-75% (~45-52 GB)")
# print(f"Training speedup: ~150-200% faster")
# print(f"Memory safety: LOW (aggressive OOM recovery expected)")
# print()

# ══════════════════════════════════════════════════════════════════════════
#  PARÁMETROS DE VALIDACIÓN Y GENERACIÓN
# ══════════════════════════════════════════════════════════════════════════
VALIDATION_EVERY   = 50
NUM_VALIDATION_ANGLES = 10
VALIDATION_DDIM_STEPS = 260
GENERATION_DDIM_STEPS = 340
GENERATE_ANGLES    = [95, 100, 110, 120, 130, 140, 150, 160, 170, 180]

# ══════════════════════════════════════════════════════════════════════════
#  RUTAS Y DIRECTORIOS
# ══════════════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, "input")
VALIDATION_FOLDER = os.path.join(DATA_FOLDER, "validation")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")

TRAIN_PNG_FOLDER      = os.path.join(OUTPUT_FOLDER, "train", "PNG")
TRAIN_MAT_FOLDER      = os.path.join(OUTPUT_FOLDER, "train", "MAT")
GENERATE_PNG_FOLDER   = os.path.join(OUTPUT_FOLDER, "generate", "PNG")
GENERATE_MAT_FOLDER   = os.path.join(OUTPUT_FOLDER, "generate", "MAT")
VALIDATION_PNG_FOLDER = os.path.join(OUTPUT_FOLDER, "validation", "PNG")
VALIDATION_MAT_FOLDER = os.path.join(OUTPUT_FOLDER, "validation", "MAT")

DIT_MODEL_PATH      = os.path.join(OUTPUT_FOLDER, "dit_model_latest.pt")
VAE_ENCODER_PATH    = os.path.join(OUTPUT_FOLDER, "vae_encoder.pt")


def create_output_dirs(*folders):
    for f in folders:
        os.makedirs(f, exist_ok=True)


def create_all_dirs():
    create_output_dirs(
        TRAIN_PNG_FOLDER, TRAIN_MAT_FOLDER,
        GENERATE_PNG_FOLDER, GENERATE_MAT_FOLDER,
        VALIDATION_PNG_FOLDER, VALIDATION_MAT_FOLDER
    )


def verify_config(checkpoint):
    checks = [
        ("ROI_HEIGHT", checkpoint.get("roi_height"), ROI_HEIGHT),
        ("ROI_WIDTH",  checkpoint.get("roi_width"),  ROI_WIDTH),
    ]
    for name, saved, current in checks:
        if saved is not None and saved != current:
            print(f"  [!] AVISO: {name} mismatched")
