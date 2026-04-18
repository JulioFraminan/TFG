# CAMBIOS EXACTOS A REALIZAR

## 📝 Cambio 1: train.py (línea 223)

**Ubicación:** `/home/j.framinan/Intento_Transformers/train.py`

**ANTES:**
```python
            # 3. Modelo predice ruido en espacio latente
            # Pero para esto, necesitamos modificar el forward pass del modelo
            # Temporalmente, pasar z_t directo en lugar de imagen completa
            noise_pred_z = model.forward_latent(z_t, t, batch_ang)
            loss = criterion(noise_pred_z, noise)
```

**DESPUÉS:**
```python
            # 3. Modelo predice ruido en espacio latente
            noise_pred_z = model(z_t, t, batch_ang)
            loss = criterion(noise_pred_z, noise)
```

**Explicación:** `forward_latent()` deja de existir en `model_clean.py`. El método `forward()` ahora hace exactamente eso.

---

## 📝 Cambio 2: generate.py (línea ~32)

**Ubicación:** `/home/j.framinan/Intento_Transformers/generate.py`
**Función:** `ddim_sample_latent()`

**ANTES:**
```python
def ddim_sample_latent(model, z_T, timesteps, conditioning=None, eta=0.0, device="cpu"):
    """DDIM sampling loop en latent space.
    ...
    """
    z_t = z_T.clone()
    
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            t_tensor = torch.tensor([t], dtype=torch.long, device=device)
            
            # Predict noise en latent space
            eps_pred = model.forward_latent(z_t, t_tensor, conditioning)
```

**DESPUÉS:**
```python
def ddim_sample_latent(model, z_T, timesteps, conditioning=None, eta=0.0, device="cpu"):
    """DDIM sampling loop en latent space.
    ...
    """
    z_t = z_T.clone()
    
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            t_tensor = torch.tensor([t], dtype=torch.long, device=device)
            
            # Predict noise en latent space
            eps_pred = model(z_t, t_tensor, conditioning)
```

**Explicación:** Mismo que anterior - `forward()` reemplaza a `forward_latent()`.

---

## 📝 Cambio 3: model.py (REEMPLAZAR TODO)

**Ubicación:** `/home/j.framinan/Intento_Transformers/model.py`

**ACCIÓN:** 
```bash
cp model_clean.py model.py
```

**Qué se elimina:**
- ❌ Método `forward()` antiguo (líneas 465-530)
- ❌ Método `forward_latent()` (líneas 539-600)
- ❌ Método `encode()` (línea 533)
- ❌ Método `decode()` (línea 537)
- ❌ Método `sample_from_noise()` (línea 603)
- ❌ Método privado `_initialize_pos_embed()` (línea 622)
- ❌ VAE encoder/decoder en `__init__` (líneas 413-423)
- ❌ Todo el código de inicialización de VAE (líneas 155-423)
- ❌ Clase `VAEEncoder` completa (líneas 26-56)
- ❌ Clase `VAEDecoder` completa (líneas 65-100)

**Qué se añade:**
- ✅ Nuevo método `forward()` limpio (línea ~242 en model_clean.py)
- ✅ Método `_get_pos_embed()` consistente (línea ~219)
- ✅ Sin VAE encoder/decoder

**Tamaño del cambio:**
- Antes: 639 líneas
- Después: ~380 líneas
- Reducción: 40%

---

## 🔄 Orden de Cambios Recomendado

### Paso 1: Copiar el nuevo modelo
```bash
cp /home/j.framinan/Intento_Transformers/model_clean.py \
   /home/j.framinan/Intento_Transformers/model.py
```

### Paso 2: Actualizar train.py
Reemplazar en línea ~223:
```python
noise_pred_z = model.forward_latent(z_t, t, batch_ang)
```
por:
```python
noise_pred_z = model(z_t, t, batch_ang)
```

### Paso 3: Actualizar generate.py
Reemplazar en línea ~32 (función `ddim_sample_latent`):
```python
eps_pred = model.forward_latent(z_t, t_tensor, conditioning)
```
por:
```python
eps_pred = model(z_t, t_tensor, conditioning)
```

---

## ✅ Verificación Post-Cambios

### Test 1: Compilación
```python
import model
```
Debe importar sin errores.

### Test 2: Instantiación
```python
from model import DiT
import torch

model = DiT(
    latent_channels=128,
    hidden_size=128,
    depth=3,
    num_heads=2,
)
print(f"✓ Modelo instantiado")
```

### Test 3: Forward Pass
```python
import torch
from model import DiT

model = DiT(latent_channels=128, hidden_size=128, depth=3, num_heads=2)
model.eval()

z = torch.randn(2, 128, 88, 250)
t = torch.randint(0, 1000, (2,))
cond = torch.randn(2, 1)

with torch.no_grad():
    output = model(z, t, cond)

assert output.shape == z.shape, f"Shape mismatch: {output.shape} != {z.shape}"
print(f"✓ Forward pass: {z.shape} → {output.shape}")
```

---

## ⚠️ Notas Importantes

1. **Importar desde model.py**: Solo cambia el contenido, no el nombre
2. **VAE sigue siendo cargado**: En `train.py`, línea ~163
3. **No hay cambios en config.py**: Los hyperparámetros siguen igual
4. **No hay cambios en data_utils.py**: Los datos siguen igual

---

## 🎯 Resultado Esperado

Después de los cambios:
1. **Code cleanup**: Eliminación de 259 líneas de código muerto
2. **Claridad**: Una responsabilidad por componente
3. **Compatibility**: Código sigue siendo compatible con train.py y generate.py
4. **Functionality**: El comportamiento es IDÉNTICO pero código es más limpio

---

## 💡 Reflexión

El código viejo (`model.py`) tenía:
- VAE encoder/decoder (no debería)
- Dos métodos forward incompatibles (confuso)
- Código muerto (mantener)
- Padding manual (buggy)

El código nuevo (`model_clean.py`) tiene:
- Solo predicción de ruido (responsabilidad única)
- Un método forward claro (mantenible)
- Sin código muerto (limpio)
- Sin padding manual (correcto)

Es el mismo modelo, pero escrito mejor. Como dicen: "Simple is better than complex."

---

## 📚 Documentación Generada

He creado varios documentos:
- `RESUMEN_EJECUTIVO.md` ← Lee esto primero
- `ANALISIS_PROBLEMAS.md` - Detalle de cada problema
- `MIGRACION_A_MODEL_CLEAN.md` - Guía de migración
- `model_clean.py` - El nuevo código
- Este documento - Cambios exactos

Todos están en `/home/j.framinan/Intento_Transformers/`
