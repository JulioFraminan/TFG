# 🔧 INSTRUCCIONES DE FIX — Training con VAE Pre-entrenado

## 📋 Cambios Realizados

### 1. Carga de VAE Pre-entrenado (CRÍTICO FIX)
- **Archivo**: `train.py`, después de crear modelo DiT
- **Acción**: Carga automáticamente el VAE entrenado desde `../unet_ae_modular/output/unet_ae_model_latest.pt`
- **Resultado**: DiT ahora entrena en espacio latent significativo (no ruido puro)
- **Impacto**: Loss debería converger de 1.0 → ~0.1-0.5

### 2. Validación PNG en Carpeta Correcta (ESTRUCTURA FIX)
- **Archivo**: `train.py`, línea ~385
- **Cambio**: `TRAIN_PNG_FOLDER` → `VALIDATION_PNG_FOLDER`
- **Resultado**: `validation_real_vs_gen.png` ahora va a `output/validation/PNG/`

### 3. VAE Congelado (OPTIMIZACIÓN)
- **Archivo**: `train.py`, después de cargar VAE
- **Acción**: Congela parámetros del VAE (no entrena, sólo usa)
- **Resultado**: Sólo entrena DiT blocks (64M parámetros, no 300M+)

---

## 🚀 CÓMO RE-EJECUTAR

### Prerequisito: Verificar que existe VAE pre-entrenado
```bash
ls -la /home/j.framinan/unet_ae_modular/output/unet_ae_model_latest.pt
```

Si NO existe:
1. Primero entrena UNet_AE:
```bash
cd /home/j.framinan/unet_ae_modular
python train.py
# Espera ~1-2 horas
```

2. Luego vuelve a Intento_Transformers

### Ejecutar training con VAE pre-entrenado
```bash
conda activate test_env
cd /home/j.framinan/Intento_Transformers

# Borrar checkpoint anterior (opcional, o simplemente sobrescribe)
# rm output/dit_model_latest.pt

python train.py
```

**Tiempo estimado**: 27 minutos (igual que antes, porque EPOCHS=100)

### Monitores el progreso
Deberías ver:
```
✓ Cargando VAE pre-entrenado desde: /home/j.framinan/unet_ae_modular/output/unet_ae_model_latest.pt
  ✓ VAE encoder/decoder cargados correctamente
  ✓ VAE congelado (no se entrena, sólo se usa como encoder/decoder)

  Época   1/100  |  MSE loss = 0.xxx  |  LR = 1.10e-06  |  ...
  Época  10/100  |  MSE loss = 0.xxx  |  LR = 1.10e-05  |  ...
  ...
  (Loss debería BAJAR con cada época, no quedarse en 1.0)
```

---

## 📊 Cambios Esperados en Output

### Antes (Sin VAE pre-entrenado):
```
  Época 100/100  |  MSE loss = 1.000133
  
  Validación:
    MAE medio   = nan dB
    RMSE medio  = nan dB
  
  validation_real_vs_gen.png → TRAIN_PNG_FOLDER (INCORRECTO)
```

### Después (Con VAE pre-entrenado + fixes):
```
  Época 100/100  |  MSE loss = 0.15-0.30 (CONVERGE!)
  
  Validación:
    MAE medio   = 1-5 dB (RAZONABLE)
    RMSE medio  = 2-8 dB (RAZONABLE)
  
  validation_real_vs_gen.png → VALIDATION_PNG_FOLDER (CORRECTO)
  validation_real_vs_gen.png → Imágenes con contenido (no blancos)
```

---

## 🧪 Verificación Post-Training

```bash
# 1. Verificar estructura de outputs
ls -la output/train/PNG/
ls -la output/validation/PNG/
ls -la output/validation/MAT/

# 2. Verificar que validation PNG tiene contenido (no está vacío)
file output/validation/PNG/validation_real_vs_gen.png

# 3. Verificar modelo guardado
ls -lh output/dit_model_latest.pt
# Debería ser ~1.5 GB (sin VAE adentro)

# 4. Inspeccionar loss en gráfica
# Abre output/train/PNG/training_curve_dit.png
# - Gráfica 1: Loss debería bajar (no plana en 1.0)
# - Gráfica 2: Learning rate debería seguir schedule
```

---

## ⚠️ Troubleshooting

### Problema: "VAE pre-entrenado NO encontrado"
```
⚠️  VAE pre-entrenado NO encontrado en: /home/j.framinan/unet_ae_modular/output/unet_ae_model_latest.pt
```

**Solución**:
1. Verifica que `unet_ae_modular/` existe en `/home/j.framinan/`
2. Si no existe, clone el repo o crea un stub VAE
3. O entrena UNet primero: `cd ../unet_ae_modular && python train.py`

### Problema: Loss sigue siendo ~1.0
```
Época 100/100  |  MSE loss = 1.000xxx
```

**Posibles causas**:
1. VAE no se cargó correctamente (revisar logs)
2. Datos de entrada son puros ruido (revisar data_utils.py)
3. VAE no fue entrenado bien (el UNet_AE tiene pérdida baja?)

**Solución**:
1. Ejecutar `bash verify_structure.sh` para diagnosticar
2. Revisar que UNet_AE loss converge correctamente
3. Si no, entrenar UNet_AE más epochs: `EPOCHS = 200` en config

### Problema: Memory error (CUDA out of memory)
```
RuntimeError: CUDA out of memory
```

**Solución** (en `config.py`):
```python
BATCH_SIZE = 1  # (en lugar de 2)
MODEL_DEPTH = 8  # (en lugar de 12)
MODEL_HIDDEN_SIZE = 256  # (en lugar de 512)
```

---

## 🎯 Resumen

| Aspecto | Antes | Después |
|--------|-------|---------|
| **Loss** | ~1.0 (no converge) | ~0.2-0.5 (converge) |
| **Validación** | NaN (no funciona) | MAE ~1-5 dB (funciona) |
| **Outputs PNG** | Blancos/vacíos | Imágenes significativas |
| **Estructura** | validation_real_vs_gen.png en TRAIN_PNG | En VALIDATION_PNG (correcto) |
| **VAE** | Aleatorio (sin entrenar) | Pre-entrenado (robusto) |

---

## 🚀 Next Steps

1. **Verificar UNet_AE** (si no existe):
   - `ls -la /home/j.framinan/unet_ae_modular/output/`
   - Si no existe `unet_ae_model_latest.pt`, entrenar primero

2. **Re-ejecutar training**:
   - `python train.py`

3. **Inspeccionar resultados**:
   - Revisar loss curve (debería bajar)
   - Revisar validation PNG (debería tener contenido)

4. **Si sigue fallando**:
   - Revisar `ANALYSIS_PROBLEMS.md` para otras causas
   - Contactar para debugging avanzado

---

**Versión**: 1.0 (9 de Abril 2026)  
**Estado**: ✅ Listo para re-ejecutar con VAE pre-entrenado
