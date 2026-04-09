# Plan de Debugging para DiT - Gráficos Vacíos

## Problema Actual
Los gráficos generados durante validación salen vacíos/blancos. El entrenamiento está corriendo pero:
1. El modelo es muy pequeño (178k parámetros antes)
2. Los gráficos no tienen contenido

## Cambios Realizados

### 1. Hyperparameters Ajustados (config.py)
```python
LEARNING_RATE = 5e-4      # Fue 1e-4 (5x más alto)
WARMUP_STEPS = 50         # Fue 1000 (20x menor - problema de warmup)
```
**Razón:** Con solo 11 batches/época × 100 épocas = 1100 pasos totales, un warmup de 1000 pasos es 91% del entrenamiento. El modelo nunca tenía LR suficientemente alto para aprender.

### 2. Modelo Aumentado (config.py)
```python
# Configuración anterior (NO FUNCIONÓ):
MODEL_HIDDEN_SIZE = 64    # Muy pequeño
MODEL_DEPTH = 2
MODEL_NUM_HEADS = 1
# Parámetros: 178,176

# Nueva configuración:
MODEL_HIDDEN_SIZE = 128   # Tamaño razonable
MODEL_DEPTH = 3           # Más profundidad sin romper memoria
MODEL_NUM_HEADS = 2       # Mejor atención multi-cabeza
# Parámetros estimados: ~500k-600k (todavía seguro)
```

### 3. Mensajes de Debug Añadidos

#### En generate.py (líneas 161-180):
```python
print(f"[DEBUG] z_0 shape: {z_0.shape}, min: {z_0.min():.4f}, max: {z_0.max():.4f}")
print(f"[DEBUG] gen_np (VAE output): min: {gen_np.min():.4f}, max: {gen_np.max():.4f}")
print(f"[DEBUG] gen_norm_01 (after conversion): min: {gen_norm_01.min():.4f}, max: {gen_norm_01.max():.4f}")
print(f"[DEBUG] gen_tl (final dB values): min: {gen_tl.min():.2f}, max: {gen_tl.max():.2f}")
```

#### En train.py (en validación):
```python
print(f"[DEBUG] gen_tl shape: {gen_tl.shape}, min: {gen_tl.min():.2f}, max: {gen_tl.max():.2f}")
print(f"[DEBUG] val_tl shape: {val_tl.shape}, min: {val_tl.min():.2f}, max: {val_tl.max():.2f}")
```

## Problemas Potenciales a Detectar

### Escenario 1: Ruido sin estructura
**Síntoma:** 
```
[DEBUG] z_0: min: -X, max: X, mean: ≈0  (gaussiano puro)
[DEBUG] gen_tl: min: -100 (o extremos), max: -4, mean: -50
```
**Causa:** El modelo no está aprendiendo a denoiser, solo deja ruido gaussiano
**Solución:** Aumentar capacidad del modelo o tiempo de entrenamiento

### Escenario 2: Valores constantes (plateau)
**Síntoma:**
```
[DEBUG] gen_tl: min: -52.1, max: -51.9, mean: -52.0
```
**Causa:** El modelo colapsó a un valor medio constante
**Solución:** Revisar inicialización, normalización o task

### Escenario 3: Valores fuera de rango
**Síntoma:**
```
[DEBUG] gen_norm_01: min: -0.5, max: 1.5  (fuera de [0,1])
```
**Causa:** VAE está dando salidas inesperadas
**Solución:** Verificar que VAE está congelado, cargar correctamente

## Próximos Pasos

1. **Ejecutar entrenamiento** con nuevos hyperparámetros
2. **Monitorear logs** para ver:
   - ¿Baja la pérdida? (debe bajar de 1.3 a ~0.2-0.5)
   - ¿Los mensajes [DEBUG] muestran valores razonables?
3. **Analizar gráficos resultantes**
4. Si todavía hay problemas:
   - Aumentar LEARNING_RATE a 1e-3
   - Aumentar MODEL_DEPTH a 4
   - Reducir WARMUP_STEPS a 20

## Configuración a Testear Próximamente

| Config | Hidden | Depth | Heads | Est. Params | Peak Mem | Notas |
|--------|--------|-------|-------|------------|----------|--------|
| Ultra-small (actual) | 64 | 2 | 1 | ~178k | 8.2 GB | Demasiado pequeño |
| Small (NEW) | 128 | 3 | 2 | ~500-600k | <20 GB | Target actual |
| Medium (v2) | 256 | 4 | 4 | ~2.5M | >30 GB | Si cabe |
| Large (v3) | 512 | 6 | 8 | ~10M | OOM | Probable overflow |
