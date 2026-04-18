# Estructura de Exportación de Archivos

## Resumen de Carpetas

```
output/
├── train/                          ← Resultados del ENTRENAMIENTO (train.py)
│   ├── PNG/
│   │   ├── training_curve_dit.png       (curva de pérdida + learning rate)
│   │   └── validation_real_vs_gen.png   (validación durante training: 3 paneles por plano)
│   └── MAT/
│       └── (vacío por ahora; futuro: checkpoints intermedios)
│
├── validation/                     ← Resultados de VALIDACIÓN (durante train.py)
│   ├── PNG/
│   │   └── (para futuras extensiones)
│   └── MAT/
│       └── val_gen_PlaneAngle*.mat     (planos generados durante validación)
│
├── generate/                       ← Resultados de GENERACIÓN (generate.py)
│   ├── PNG/
│   │   └── plano_angulo_*.png          (comparativa: semilla vs generado vs error)
│   └── MAT/
│       └── plano_angulo_*.mat          (planos generados con DDIM)
│
├── dit_model_latest.pt             ← Checkpoint del modelo DiT (train.py)
└── vae_encoder.pt                  ← Checkpoint del VAE encoder (futuro)
```

---

## Detalle por Script

### `train.py` → Qué exporta y DÓNDE

| Qué | Carpeta | Archivo | Línea | Notas |
|-----|---------|---------|-------|-------|
| **Curva de entrenamiento** | `TRAIN_PNG_FOLDER` | `training_curve_dit.png` | 265 | 2 subplots: MSE loss + LR schedule |
| **Validación: PNG** | `TRAIN_PNG_FOLDER` | `validation_real_vs_gen.png` | 397 | 3 columnas × N filas (real, generado, error) |
| **Validación: MAT** | `VALIDATION_MAT_FOLDER` | `val_gen_PlaneAngle*.mat` | 386 | **CORREGIDO**: era `GENERATE_MAT_FOLDER` ❌ |
| **Modelo entrenado** | `OUTPUT_FOLDER` | `dit_model_latest.pt` | 419 | Checkpoint con state_dict + hyperparams |

**Validación** sólo se ejecuta si existe `input/validation/` con archivos `.mat`.

---

### `generate.py` → Qué exporta y DÓNDE

| Qué | Carpeta | Archivo | Línea | Notas |
|-----|---------|---------|-------|-------|
| **Generación: PNG** | `GENERATE_PNG_FOLDER` | `plano_angulo_*.png` | ~290 | 3 subplots: semilla vs generado vs error |
| **Generación: MAT** | `GENERATE_MAT_FOLDER` | `plano_angulo_*.mat` | ~260 | Plano sintetizado para ángulo objetivo |

**Entrada**: ángulos especificados en `GENERATE_ANGLES` (config.py).

---

## Definiciones en `config.py`

```python
# Base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")

# Sub-carpetas
TRAIN_PNG_FOLDER      = os.path.join(OUTPUT_FOLDER, "train", "PNG")
TRAIN_MAT_FOLDER      = os.path.join(OUTPUT_FOLDER, "train", "MAT")
GENERATE_PNG_FOLDER   = os.path.join(OUTPUT_FOLDER, "generate", "PNG")
GENERATE_MAT_FOLDER   = os.path.join(OUTPUT_FOLDER, "generate", "MAT")
VALIDATION_PNG_FOLDER = os.path.join(OUTPUT_FOLDER, "validation", "PNG")
VALIDATION_MAT_FOLDER = os.path.join(OUTPUT_FOLDER, "validation", "MAT")

# Checkpoints
DIT_MODEL_PATH        = os.path.join(OUTPUT_FOLDER, "dit_model_latest.pt")
VAE_ENCODER_PATH      = os.path.join(OUTPUT_FOLDER, "vae_encoder.pt")

def create_all_dirs():
    """Crea TODAS las carpetas de salida."""
    create_output_dirs(
        TRAIN_PNG_FOLDER, TRAIN_MAT_FOLDER,
        GENERATE_PNG_FOLDER, GENERATE_MAT_FOLDER,
        VALIDATION_PNG_FOLDER, VALIDATION_MAT_FOLDER
    )
```

---

## Checklist de Correcciones (9 de Abril de 2026)

- ✅ **Línea 386 en `train.py`**: Cambié `GENERATE_MAT_FOLDER` → `VALIDATION_MAT_FOLDER`
  - **Causa original**: NameError al ejecutar `train.py` con validación.
  - **Impacto**: Los `.mat` de validación ahora se guardan en `output/validation/MAT/` (correcto).
  
- ✅ **Importaciones en `train.py`**: Añadidas `GENERATE_PNG_FOLDER, GENERATE_MAT_FOLDER` a imports (aunque no se usan en entrenamiento).
  - **Razón**: Evitar confusiones futuras; tener todas las constantes disponibles.

- ✅ **Limpieza de `train.py`**: Eliminada definición duplicada de `def main()`.
  - **Impacto**: Código más limpio, sin ambigüedades.

---

## Recomendaciones para Evitar Errores Similares

1. **Auditoría de carpetas**: Antes de `savefig()` o `save_mat()`, preguntarse:
   - ¿Es validación o generación?
   - ¿Dónde debe ir este archivo?
   - ¿Está definida la carpeta en `config.py`?
   - ¿Se llama `create_all_dirs()` para crearla?

2. **Nombres consistentes**: Usar `*_PNG_FOLDER`, `*_MAT_FOLDER`, `*_MODEL_PATH` en `config.py`.

3. **Assertions defensivos** (opcional): Antes de guardar, validar:
   ```python
   assert os.path.exists(VALIDATION_MAT_FOLDER), f"Carpeta {VALIDATION_MAT_FOLDER} no existe"
   ```

4. **Documentación inline**: Siempre comentar qué carpeta se usa y por qué.

---

**Última revisión**: 9 de Abril de 2026  
**Autor**: Auditoría completa de exportación  
**Estado**: ✅ CORREGIDO
