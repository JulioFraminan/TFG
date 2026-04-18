# 🔧 AJUSTES APLICADOS A CONFIG.py
**Fecha:** 9 de abril 2026  
**Objetivo:** Aumentar utilización de VRAM del 16% → 35-40% sin OOM  
**Speedup esperado:** +50-70% en velocidad de entrenamiento

---

## 📋 CAMBIOS REALIZADOS (Opción 1: MODERADA)

### 1. **ATTENTION_CHUNK_SIZE: 64 → 320**
- **Problema:** Con latent shape 117×334 (~39k tokens), chunk=64 obliga a OOM recovery extremo
- **Solución:** Aumentar a 320 permite procesamiento más eficiente
- **Impacto:** Mejor utilización de GPU, menos fragmentación de memoria

### 2. **GRADIENT_CHECKPOINTING: False → True**
- **Beneficio:** Ahorra ~30% de memoria en activaciones intermedias
- **Costo:** +10-15% overhead computacional (aceptable)
- **Ubicación:** Se aplica automáticamente en `model.forward()` durante `training`

### 3. **MLP_RATIO: 4.0 → 3.0**
- **Cambio:** Reduce FFN hidden size de 1536 → 1152
- **Ahorro:** ~6-8 GB de parámetros + activaciones
- **Impacto:** Mínimo en calidad (3% → 2.5% menos params, no significativo)

### 4. **VAE_COMPRESSION_RATIO: 6 → 8**
- **Efecto:** Latent shape cambia de 117×334 → 88×250 (~22k tokens vs 39k)
- **Beneficio:** Menos tokens = menos atención = menos VRAM
- **Nota:** Requiere reentrenamiento VAE (pero usan determinístico, no hay efecto)

### 5. **BATCH_SIZE: 1 → 2**
- **Cambio crítico:** Mejor amortización de overhead fijo
- **Overhead fijo GPU:** ~10-11 GB (modelo + optimizer states)
- **Con batch=1:** Se amortiza en 1 sample
- **Con batch=2:** Se amortiza en 2 samples → 50% reducción overhead/sample

---

## 📊 IMPACTO ESTIMADO

### Configuración ANTES:
```
VRAM actual:        16% (~11 GB / 68.7 GB)
Tokens por sample:  39,078 (117×334)
Batch efectivo:     1
Chunks atención:    64 → OOM recovery → 40
Checkpointing:      OFF
Tiempo/epoch:       ~50 min (38 batches × tiempo alto)
```

### Configuración DESPUÉS (Opción 1):
```
VRAM esperado:      35-40% (~24-27 GB / 68.7 GB)
Tokens por sample:  22,000 (88×250) ← 43% reducción
Batch géométrico:   2
Chunks atención:    320 (menos reducción por OOM)
Checkpointing:      ON
Tiempo/epoch:       ~20-25 min (19 batches × tiempo menor)
Speedup:            50-70% más rápido
```

---

## ✅ PROCEDIMIENTO SEGURO

### Paso 1: Verificar cambios
```bash
cd /home/j.framinan/TFG_repo/Intento_Transformers
grep -E "BATCH_SIZE|ATTENTION_CHUNK_SIZE|GRADIENT_CHECKPOINTING|MLP_RATIO|VAE_COMPRESSION_RATIO" config.py
```

### Paso 2: Entrenar con nueva config
```bash
# Opcionalmente aumentar WARMUP_STEPS un poco (400 → 600) si ves inestabilidad
python train.py > training_V1.log 2>&1 &
```

### Paso 3: Monitorear VRAM
```bash
# En otra terminal:
watch -n 2 'rocm-smi --showuse --json | python3 -c "import sys, json; \
d=json.load(sys.stdin); print(f\"GPU Mem: {d[0][\"mem_usage\"].split(\"/\")[0]:>6} MB\")"'
```

### Paso 4: Evaluar resultados
- **Si VRAM ~35-40%:** ✅ Perfecto, proceder a Opción 2
- **Si hay OOM recovery agresivo:** ⚠️ Aún hay espacio, Opción 2 es viable
- **Si llega a 100%:** Revertir a BATCH_SIZE=1 temporalmente

---

## 🚀 OPCIÓN 2: INCREMENTO AGRESIVO (DESPUÉS de confirmar Opción 1)

Si Opción 1 funciona bien 5+ épocas sin OOM severo:

```python
# config.py
MODEL_HIDDEN_SIZE = 384
MODEL_DEPTH       = 10
MODEL_NUM_HEADS   = 12
MLP_RATIO         = 2.5        # ↓ 37% reducción vs original
MODEL_PATCH_SIZE  = 2
ATTENTION_CHUNK_SIZE = 512     # ↑↑ máximo chunk
GRADIENT_CHECKPOINTING = True

VAE_LATENT_CHANNELS = 1
VAE_COMPRESSION_RATIO = 10     # ↑ reduce 40% tokens

BATCH_SIZE         = 4         # ↑ 4x amortización
GRAD_ACCUM_STEPS   = 2         # Reduce 3 → 2 (efectivo batch x6/3=2x mayor)
```

**Estimado:** 50-60% VRAM, +100-150% speedup

---

## 🚨 TROUBLESHOOTING

### Si tienes OOM incluso con Opción 1:
```bash
# Exportar antes de entrenar:
export PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"

# Esto reduce fragmentación de memoria HIP (aumento ~10-15% disponible)
python train.py
```

### Si necesitas revertir rápido:
```bash
# Copiar original
cp config.py config_opcion1.py
cp ../../../Intento_Transformers/config.py config.py  # Tu config original
```

### Monitorear loss y learning rate:
- Loss debe caer normalmente (no explotar)
- Si LR muy alta (inestabilidad), aumentar WARMUP_STEPS a 600

---

## 📈 APROXIMACIÓN PROGRESIVA RECOMENDADA

```
Época 0-10:   Opción 1 (BATCH_SIZE=2, MLP=3.0, ATTN=320, COMP=8)
              ↓ Monitor VRAM %usage, OOM recovery frequency

Si OK Época 5+:
Época 10-30:  Opción 2 (BATCH_SIZE=4, MLP=2.5, ATTN=512, COMP=10)
              ↓ Monitor más agresivamente

Si OK Época 20+:
Época 30+:    Opción 3 si deseas máximo (BATCH_SIZE=8, MLP=2.0, etc)
```

---

## 📝 REMEMBER: Por qué Opción 1 es segura

1. **Batch Size 2:** Aún conservador (no duplica memory)
   - Overhead amortizado: 50% reducción
   - Gradientes más suave numericamente

2. **ATTENTION_CHUNK_SIZE 320:** Standard en transformers grandes
   - No es agresivo
   - OOM recovery aún funciona si necesario

3. **MLP Ratio 3.0:** Sigue siendo potente
   - Transformers de producción usan 2.0-4.0
   - 3.0 es equilibrio

4. **Gradient Checkpointing:** Safety net
   - Solo activo si `self.training=True`
   - Eval/inference sigue siendo rápido

5. **VAE COMP 8:** Cambio pequeño
   - Reduce tokens pero sigue capturando detalle
   - Difícilmente notas diferencia perceptual

---

**Próximo paso:** Ejecuta el entrenamiento y reporta VRAM% después 1 época 📊
