# Resumen de Cambios - Debugging Gráficos Vacíos

## 🔍 Problema Identificado

El entrenamiento corría sin errores pero los gráficos de validación salían vacíos/blancos. Causas identificadas:

### 1. **Hyperparameter Crítico: WARMUP_STEPS = 1000**
- **Dataset pequeño:** Solo 22 planos de entrenamiento
- **Batches:** 11 batches/época × 100 épocas = **1100 pasos totales**
- **Problema:** Warmup de 1000 pasos = **91% del entrenamiento en warmup**
- **Consecuencia:** LR comienza en 1.10e-06, muy bajo para aprender

✅ **Solución:** Reducir WARMUP_STEPS a 50 (5% del entrenamiento)

### 2. **Modelo Demasiado Pequeño: 178k parámetros**
- Capacidad insuficiente para modelar la complejidad del espacio latente
- 22,000 tokens × 64 hidden = demasiada compresión de información

✅ **Solución:** Aumentar a 128 hidden, 3 depth, 2 heads (~500-600k parámetros)

## 📝 Cambios Realizados

### config.py
```python
# ANTES:
LEARNING_RATE = 1e-4
WARMUP_STEPS = 1000
MODEL_HIDDEN_SIZE = 64
MODEL_DEPTH = 2
MODEL_NUM_HEADS = 1

# DESPUÉS:
LEARNING_RATE = 5e-4       # 5× más alto
WARMUP_STEPS = 50          # 20× menor
MODEL_HIDDEN_SIZE = 128    # 2× más grande
MODEL_DEPTH = 3            # 1.5× más profundo
MODEL_NUM_HEADS = 2        # 2× más cabezas
```

### generate.py
Añadidos mensajes de DEBUG para rastrear:
- `z_0` valores antes de VAE decode
- `gen_np` salida del VAE decoder
- `gen_norm_01` después de conversión [-1,1] → [0,1]
- `gen_tl` valores finales en dB

### train.py
Añadidos mensajes de DEBUG en validación para comparar:
- Rango y forma de imágenes generadas
- Rango y forma de imágenes reales

## 🧪 Cómo Verificar

### Opción A: Test Rápido (sin entrenar)
```bash
cd /home/j.framinan/Intento_Transformers
python quick_test.py
```
Genera una imagen con pesos aleatorios para verificar pipeline.

### Opción B: Entrenar con nuevos parámetros
```bash
cd /home/j.framinan/Intento_Transformers
python train.py 2>&1 | tee logs/training_run_latest.log
```

Monitorear que:
1. **Pérdida baje:** Época 1: ~1.3 → Época 50: ~0.4-0.5
2. **LR suba rápido:** Hasta epoch ~50, luego descienda
3. **Mensajes [DEBUG] tenga valores:**
   - `z_0`: distribuido gaussiano (mean≈0, std≈1)
   - `gen_tl`: rango -100 a -4 (TL valores típicos oceanografía)
4. **Gráficos tengan contenido:** No uniformes

## 📊 Expectativas

| Aspecto | Antes | Después |
|---------|-------|---------|
| Parámetros | 178k | ~600k |
| Capacidad | Muy baja | Razonable |
| Learning Rate | 1e-4 (siempre bajo) | 5e-4 (competente) |
| Warmup % | 91% de entrenamiento | 5% de entrenamiento |
| Pérdida esperada | Plana ~1.0 | Bajando a 0.2-0.5 |
| Gráficos | Vacíos/blancos | Con estructura |

## 🚨 Si aún hay problemas

Si después del entrenamiento los gráficos siguen vacíos:

1. **Revisar mensajes [DEBUG]** para ver dónde se pierden los valores
2. **Aumentar más el modelo:**
   ```python
   MODEL_HIDDEN_SIZE = 256
   MODEL_DEPTH = 4
   MODEL_NUM_HEADS = 4
   ```
3. **Ajustar LR:**
   ```python
   LEARNING_RATE = 1e-3  # 10× actual
   ```
4. **Aumentar épocas:**
   ```python
   EPOCHS = 200
   ```

## 📂 Archivos Modificados
- ✅ config.py - Hyperparámetros
- ✅ generate.py - Debug messages (161-180)
- ✅ train.py - Debug messages (validación)
- ✅ DEBUG_PLAN.md - Este plan de debugging
- ✅ quick_test.py - Script de test rápido
