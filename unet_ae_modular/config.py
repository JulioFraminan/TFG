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
ROI_CORNER = (7, -2)             # (X_físico, Z_físico) esquina SUPERIOR IZQUIERDA
                                     # del gráfico (visual, con origin="lower").
                                     # X = borde izquierdo,  Z = borde superior.
                                     # La ROI crece hacia abajo y a la derecha.
                                     # Solo se usa cuando ROI_MODE == "corner_fixed".
USE_BLOCK_VARIABLES = True
# ══════════════════════════════════════════════════════════════════════
#  PARÁMETROS DE ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════
BATCH_SIZE       = 8
EPOCHS           = 500
LEARNING_RATE    = 5.380920824447353e-05
WEIGHT_DECAY     = 4.898620823554843e-05
USE_AUGMENTATION = True         # True: x4 (ruido + contraste) | False: solo datos originales
MODEL_DROPOUT    = 0.18809363235942

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

    # Verificar rangos de normalización
    for key, label in [("angle_min", "ANGLE_MIN"), ("angle_max", "ANGLE_MAX"),
                       ("tl_min", "TL_MIN"), ("tl_max", "TL_MAX")]:
        val = checkpoint.get(key)
        if val is not None:
            print(f"  [i] {label} guardado en .pt: {val:.4f}")
