import math
import os

# ══════════════════════════════════════════════════════════════════════
#  PARÁMETROS DEL MODELO Y DATOS
#  Las ROIs pueden ser rectangulares (alto × ancho).
# ══════════════════════════════════════════════════════════════════
ROI_HEIGHT     = 480        # alto del parche extraído del plano original
ROI_WIDTH      = 1520       # ancho del parche extraído
ROIS_PER_PLANE = 1
# ══════════════════════════════════════════════════════════════════════
#  MODO DE DEFINICIÓN DE LA ROI
#  "center_max"  → busca el punto de mayor valor del plano y lo usa
#                  como centro de la ROI  (comportamiento original).
#  "corner_fixed" → usa las coordenadas físicas ROI_CORNER como esquina
#                   superior izquierda de la ROI.
# ══════════════════════════════════════════════════════════════════════
ROI_MODE = "corner_fixed"          # opciones: "center_max" | "corner_fixed"
ROI_CORNER = (50, -5)             # (X_físico, Z_físico) esquina SUPERIOR IZQUIERDA
                                     # del gráfico (visual, con origin="lower").
                                     # X = borde izquierdo,  Z = borde superior.
                                     # La ROI crece hacia abajo y a la derecha.
                                     # Solo se usa cuando ROI_MODE == "corner_fixed".
USE_BLOCK_VARIABLES = True
# ══════════════════════════════════════════════════════════════════════
#  PARÁMETROS DE ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════
BATCH_SIZE       = 2
EPOCHS           = 500
LEARNING_RATE    = 0.0001225428407544687
WEIGHT_DECAY     = 1.34119606960677e-05
USE_AUGMENTATION = True         # True: x4 (ruido + contraste) | False: solo datos originales
MODEL_DROPOUT    = 0.060945562660182084
SEED             = 42

# =============================
# PHYSICS / PINN PARAMETERS
# =============================
PHYSICS_BATCH_SIZE = 0
DATA_WEIGHT = 0.10698357212442366
PHYSICS_WEIGHT = 0.002166599966311893
INTERFACE_BATCH_SIZE = 2048
INTERFACE_WEIGHT = 0.06044974206707929
PDE_TYPE = "two_layer_helmholtz"  # "none", "laplace", "helmholtz", "two_layer_helmholtz"

# Two-medium acoustics (air/water)
FREQUENCY_HZ = 1000.0
AIR_SOUND_SPEED = 343.0
AIR_DENSITY = 1.225
WATER_SOUND_SPEED = 1480.0
WATER_DENSITY = 1000.0
AIR_ABOVE_INTERFACE = True
INTERFACE_Z = 0.0  # None -> auto midpoint of ROI bounds
INTERFACE_EPS = 0.5
TL_REF_PRESSURE = 1.0
OUTPUT_IS_TL = True
TL_DB_SIGN = -1.0
TL_DB_OFFSET = 0.0
TL_DB_CLAMP_MIN = 0.0
TL_DB_CLAMP_MAX = 100.0

# Single-medium Helmholtz (PDE_TYPE = "helmholtz")
if INTERFACE_Z is None:
    _interface_z_ref = 0.0
else:
    _interface_z_ref = float(INTERFACE_Z)
HELMHOLTZ_K_MEDIUM = "water" if ROI_CORNER[1] < _interface_z_ref else "air"
HELMHOLTZ_K_AIR = 2.0 * math.pi * FREQUENCY_HZ / AIR_SOUND_SPEED
HELMHOLTZ_K_WATER = 2.0 * math.pi * FREQUENCY_HZ / WATER_SOUND_SPEED
HELMHOLTZ_K = HELMHOLTZ_K_AIR if HELMHOLTZ_K_MEDIUM == "air" else HELMHOLTZ_K_WATER

# ══════════════════════════════════════════════════════════════════════
#  INPAINTING
#  INPAINT_MODE:
#    - "none"      -> sin inpainting
#    - "train"     -> inpainting solo en entrenamiento
#    - "inference" -> inpainting solo en generacion/validacion
#    - "both"      -> inpainting en entrenamiento y en generacion/validacion
# ══════════════════════════════════════════════════════════════════════
INPAINT_MODE = "none"
INPAINT_PRESERVE_KNOWN = False

# Mascaras aleatorias de huecos durante entrenamiento/inferencia.
# known_mask = 1 en pixeles conocidos, 0 en huecos.
INPAINT_MIN_HOLES = 1
INPAINT_MAX_HOLES = 4
INPAINT_MIN_HOLE_RATIO = 0.08   # tamano minimo del hueco respecto a H/W
INPAINT_MAX_HOLE_RATIO = 0.30   # tamano maximo del hueco respecto a H/W
INPAINT_KEEP_FULL_PROB = 0.05   # prob. de dejar una muestra sin huecos
INPAINT_FILL_VALUE = 0.0        # valor usado para ocultar huecos
INPAINT_LOSS_KNOWN_WEIGHT = 0.1 # peso de consistencia en zona conocida

# ══════════════════════════════════════════════════════════════════════
#  PARÁMETROS DE GENERACIÓN
# ══════════════════════════════════════════════════════════════════════
NOISE_STD       = 0.15
GENERATE_ANGLES = [-180]
#GENERATE_ANGLES = [-175, -170, -165, -160, -155, -150, -145, -140, -135, -130, -125,
#-120, -115, -110, -105, -100, -95, -90, -85, -80, -75, -70, -65, -60,
#-55, -50, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20,
#25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105,
#110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180]

# ══════════════════════════════════════════════════════════════════════
#  RUTAS
# ══════════════════════════════════════════════════════════════════════

# Carpeta raíz del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Carpeta de datos de entrada (planos .mat originales)
DATA_FOLDER = os.path.join(BASE_DIR, "input")

# Subcarpeta de validación (planos .mat excluidos del entrenamiento,
# se usan al final para verificar la calidad del modelo)
VALIDATION_FOLDER = os.path.join(DATA_FOLDER, "validation")

# Carpeta raíz de salida
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")

# Sub-carpetas por módulo
TRAIN_PNG_FOLDER        = os.path.join(OUTPUT_FOLDER, "train", "PNG")
TRAIN_MAT_FOLDER        = os.path.join(OUTPUT_FOLDER, "train", "MAT")
GENERATE_PNG_FOLDER     = os.path.join(OUTPUT_FOLDER, "generate", "PNG")
GENERATE_MAT_FOLDER     = os.path.join(OUTPUT_FOLDER, "generate", "MAT")
ANALYSIS_PNG_FOLDER     = os.path.join(OUTPUT_FOLDER, "analysis", "PNG")
VALIDATION_PNG_FOLDER   = os.path.join(OUTPUT_FOLDER, "validation", "PNG")
VALIDATION_MAT_FOLDER   = os.path.join(OUTPUT_FOLDER, "validation", "MAT")

MODEL_PATH = os.path.join(OUTPUT_FOLDER, "unet_ae_model.pt")


def create_output_dirs(*folders):
    """Crea las carpetas indicadas si no existen."""
    for f in folders:
        os.makedirs(f, exist_ok=True)


def verify_config(checkpoint):
    """Compara los parámetros guardados en el .pt con los de config.py.
    Avisa si hay discrepancias."""
    checks = [
        ("ROI_HEIGHT", checkpoint.get("roi_height"), ROI_HEIGHT),
        ("ROI_WIDTH",  checkpoint.get("roi_width"),  ROI_WIDTH),
    ]
    for name, saved, current in checks:
        if saved is not None and saved != current:
            print(f"  [!] AVISO: {name} en config ({current}) != "
                  f"{name} en .pt ({saved})")

    pde_type = checkpoint.get("pde_type")
    if pde_type is not None and pde_type != PDE_TYPE:
        print(f"  [!] AVISO: PDE_TYPE en config ({PDE_TYPE}) != PDE_TYPE en .pt ({pde_type})")

    # Verificar rangos de normalización
    for key, label in [("angle_min", "ANGLE_MIN"), ("angle_max", "ANGLE_MAX"),
                       ("tl_min", "TL_MIN"), ("tl_max", "TL_MAX")]:
        val = checkpoint.get(key)
        if val is not None:
            print(f"  [i] {label} guardado en .pt: {val:.4f}")
