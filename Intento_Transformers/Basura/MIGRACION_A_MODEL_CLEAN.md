# PLAN DE MIGRACIÓN: model_clean.py

## 🎯 Objetivo
Reemplazar el código redundante y confuso de `model.py` con `model_clean.py` que es:
- ✅ Basado en el DiT original de Meta
- ✅ Limpio, sin código muerto
- ✅ Una única responsabilidad: predecir ruido en latent space
- ✅ Sin VAE encoder/decoder incorporado

---

## 📝 Cambios Necesarios

### 1. **Crear el nuevo modelo**
Ya hecho: `/home/j.framinan/Intento_Transformers/model_clean.py`

### 2. **Reemplazar model.py**
```bash
cp model_clean.py model.py
```

Esto elimina:
- ❌ Método `forward()` antiguo (computaciones innecesarias)
- ❌ Método `forward_latent()` (ahora `forward()` lo hace)
- ❌ VAE encoder/decoder (responsabilidad de train.py)
- ❌ `encode()` método muerto
- ❌ `decode()` método muerto
- ❌ `sample_from_noise()` método muerto
- ❌ Padding manual de ceros
- ❌ Código duplicado de pos_embed

### 3. **Actualizar train.py (línea ~223)**

**CAMBIO ÚNICO:**
```python
# ANTES:
noise_pred_z = model.forward_latent(z_t, t, batch_ang)

# DESPUÉS:
noise_pred_z = model(z_t, t, batch_ang)
```

Eso es TODO. El resto del código de train.py sigue igual.

### 4. **Actualizar generate.py (línea ~32)**

**CAMBIO ÚNICO en ddim_sample_latent():**
```python
# ANTES:
eps_pred = model.forward_latent(z_t, t_tensor, conditioning)

# DESPUÉS:
eps_pred = model(z_t, t_tensor, conditioning)
```

---

## ✅ Verificación Post-Migración

**Paso 1: Verificar que el modelo se importa**
```python
from model import DiT
```
Debe funcionar sin errores.

**Paso 2: Verificar instantiación**
```python
model = DiT(
    latent_channels=128,
    hidden_size=128,
    depth=3,
    num_heads=2,
    num_diffusion_steps=1000,
)
```

**Paso 3: Verificar forward pass**
```python
z = torch.randn(2, 128, 88, 250)
t = torch.randint(0, 1000, (2,))
cond = torch.randn(2, 1)
output = model(z, t, cond)
# Output shape debe ser: torch.Size([2, 128, 88, 250])
```

---

## 🔧 Diferencias Clave

| Aspecto | Antes (model.py) | Después (model_clean.py) |
|---------|------------------|--------------------------|
| **forward()** | Codifica + predice + decodifica | Solo predice ruido |
| **forward_latent()** | Método duplicado | No existe (forward() lo hace) |
| **VAE** | Dentro del modelo | Fuera (train.py/generate.py) |
| **Pos embeddings** | Inconsistente | Método único `_get_pos_embed()` |
| **Padding** | Sí, ceros dañinos | No necesario |
| **Líneas de código** | 639 | ~380 |
| **Mantenibilidad** | Baja | Alta |

---

## ⚠️ IMPORTANTE: No hay cambios en VAE

El VAE sigue siendo cargado en train.py desde:
```
../unet_ae_modular/output/unet_ae_model.pt
```

La diferencia es que:
- **ANTES:** model.py cargaba VAE (confuso)
- **DESPUÉS:** train.py carga VAE y lo usa explícitamente (claro)

---

## 🚀 Después de migrar

Puedes entrenar con:
```bash
cd /home/j.framinan/Intento_Transformers
python train.py
```

**Cambios esperados:**
1. Código más limpio en los logs
2. Sin confusión sobre responsabilidades
3. Mejor debuggeable si hay problemas
4. Estructura idéntica al DiT original de Meta

---

## 📚 Archivos Relacionados

| Archivo | Estado |
|---------|--------|
| `model_clean.py` | ✅ Nuevo, listo para usar |
| `model.py` | ❌ A reemplazar |
| `train.py` | ⚠️ Necesita 1 línea de cambio (223) |
| `generate.py` | ⚠️ Necesita 1 línea de cambio (32) |
| `config.py` | ✅ Sin cambios |
| `data_utils.py` | ✅ Sin cambios |

---

## 💡 Resumen

El problema raíz era que `model.py` tenía demasiadas responsabilidades:
- Codificar imagen
- Predecir ruido
- Decodificar predicción
- Albergar VAE

Ahora `model_clean.py` hace UNA COSA BIEN:
- Predecir ruido en latent space

Y `train.py`/`generate.py` manejan VAE explícitamente.

Esto es arquitectura clara, mantenible y basada en el original de Meta.
