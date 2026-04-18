# 🚀 GUÍA RÁPIDA PRE-EJECUCIÓN

## ✅ Antes de Ejecutar train.py o generate.py

### Paso 1: Verificar que todo está correcto (1 minuto)
```bash
cd /home/j.framinan/Intento_Transformers
bash verify_structure.sh
```

Si ves ✅ en todas las líneas, ¡puedes continuar!

---

### Paso 2: Activar el entorno conda
```bash
conda activate test_env
```

---

### Paso 3: Ejecutar training
```bash
python train.py
```

**Espera**: El script tardará:
- Carga de datos: ~30-60 seg
- Entrenamiento: depende de EPOCHS (config.py)
- Validación: ~2-5 min por cada 50 epochs
- Total estimado: 24-72 horas (si EPOCHS=100-500)

**Genera**:
- `output/train/PNG/training_curve_dit.png` — Gráfica de pérdida
- `output/validation/MAT/val_gen_*.mat` — Planos de validación (✅ CARPETA CORRECTA)
- `output/validation/PNG/` — (futuro, por ahora va a TRAIN_PNG_FOLDER)
- `output/dit_model_latest.pt` — Modelo entrenado

---

### Paso 4: Ejecutar generación (opcional)
```bash
python generate.py
```

**Requiere**: Modelo entrenado (`output/dit_model_latest.pt`)

**Espera**: ~20-30 segundos por ángulo (50 pasos DDIM)

**Genera**:
- `output/generate/PNG/plano_angulo_*.png` — Comparativas visuales
- `output/generate/MAT/plano_angulo_*.mat` — Planos sintetizados

---

## 📂 Estructura de Salida Esperada

Después de ejecutar ambos scripts:

```
output/
├── train/
│   └── PNG/
│       ├── training_curve_dit.png           ✅ (train.py)
│       └── validation_real_vs_gen.png       ✅ (train.py, si hay validation/)
├── validation/
│   ├── PNG/                                 (vacío)
│   └── MAT/
│       ├── val_gen_PlaneAngle95.00_*.mat    ✅ FIXED: va aquí (no en GENERATE_MAT)
│       ├── val_gen_PlaneAngle100.00_*.mat   ✅ FIXED
│       └── ...
├── generate/
│   ├── PNG/
│   │   ├── plano_angulo_+95.0.png           ✅ (generate.py)
│   │   ├── plano_angulo_+100.0.png          ✅ (generate.py)
│   │   └── ...
│   └── MAT/
│       ├── plano_angulo_+95.0.mat           ✅ (generate.py)
│       ├── plano_angulo_+100.0.mat          ✅ (generate.py)
│       └── ...
├── dit_model_latest.pt                      ✅ (train.py)
└── vae_encoder.pt                           (futuro, no generado aún)
```

---

## ⚠️ Problemas Comunes y Soluciones

### Problema: `ModuleNotFoundError: No module named 'einops'`
```bash
conda activate test_env
pip install einops
```

### Problema: `ModuleNotFoundError: No module named 'torch'`
```bash
conda activate test_env
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Problema: Error `CUDA out of memory`
Edita `config.py`:
```python
BATCH_SIZE = 1  # (ó 2, en lugar de 8)
MODEL_DEPTH = 10  # (en lugar de 12)
MODEL_HIDDEN_SIZE = 256  # (en lugar de 512)
```

### Problema: Entrenamiento muy lento
```python
# En config.py
DIFFUSION_STEPS = 500  # (en lugar de 1000, sólo para testing)
EPOCHS = 10  # (en lugar de 100, sólo para testing)
```

### Problema: `NameError: name 'GENERATE_MAT_FOLDER' is not defined`
✅ **RESUELTO**: Ya no debería ocurrir. Si ocurre, revisa:
1. Que estés usando el train.py corregido (línea 374)
2. Que hayas activado el entorno conda correcto

---

## 📊 Monitoreo Mientras se Ejecuta

### Train.py
- Busca líneas con: `Época XXX/YYY | MSE loss = ...`
- Verifica que el loss está bajando gradualmente

### Generate.py
- Busca líneas con: `θ=+95.0° ... MAE=...`
- Verifica que los MAE son razonables (<10 dB)

---

## 🔍 Verificación Post-Ejecución

```bash
# Verificar que los archivos se crearon correctamente
ls -la output/train/PNG/
ls -la output/validation/MAT/
ls -la output/generate/PNG/
ls -la output/generate/MAT/

# Contar archivos generados
echo "Archivos PNG de validación:" ; ls output/train/PNG/ | wc -l
echo "Archivos MAT de validación:" ; ls output/validation/MAT/ | wc -l
echo "Archivos PNG de generación:" ; ls output/generate/PNG/ | wc -l
echo "Archivos MAT de generación:" ; ls output/generate/MAT/ | wc -l
```

---

## ✨ Resumen (TL;DR)

```bash
# 1. Verificar
bash verify_structure.sh

# 2. Entrenar
conda activate test_env
python train.py

# 3. Generar (opcional)
python generate.py

# 4. Revisar salidas
ls output/
```

**Nota**: El error `NameError: name 'GENERATE_MAT_FOLDER' is not defined` está **RESUELTO** ✅

---

**Última actualización**: 9 de Abril de 2026  
**Versión**: 1.0 (Post-Correcciones)  
**Estado**: ✅ LISTO PARA CLUSTER
