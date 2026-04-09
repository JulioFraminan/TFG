# 🔴 ANÁLISIS DE PROBLEMAS — Último Training

## Problema 1: Loss No Converge (Siempre ~1.0)

### Síntomas
```
Época   1/100  |  MSE loss = 1.001017
Época  10/100  |  MSE loss = 1.000083
Época  20/100  |  MSE loss = 1.000394
...
Época 100/100  |  MSE loss = 1.000133
```

**Análisis**: La pérdida nunca baja de 1.0 → el modelo NO está aprendiendo.

### Causa Probable: VAE No Entrena

En `train.py` línea ~315:
```python
# El VAE viene con modelo.vae (inicializado aleatoriamente)
# Pero NO hay loss que entrene el VAE, sólo el DiT
noise_pred = model(x_t, t, batch_ang)  # Predice ruido
loss = criterion(noise_pred, noise_target)  # MSE vs ruido
```

**Problema**: 
- El VAE encoder convierte imagen → latent (con pesos ALEATORIOS)
- El VAE decoder convierte latent → imagen (con pesos ALEATORIOS)
- DiT predice en espacio latent, pero los latents NO son significativos
- Loss = ruido puro (Gaussiano) ≈ 1.0 siempre

### Causa Secundaria: Forward Diffusion en Espacio Latente

```python
noise = torch.randn_like(batch_img)  # Ruido [1, 700, 2000]
x_t, noise_target = model.diffusion_schedule.q_sample(batch_img, t, noise)
```

Si `batch_img` es ruido puro (entrada), entonces `x_t` también será ruido puro → loss ≈ 1.0 siempre.

---

## Problema 2: Validación Genera NaN

```
PlaneAngle10.00_Length499.00.mat: MAE=nan, RMSE=nan, Max=nan, MAPE=nan%
```

**Causa**: `gen_tl` son todos zeros o NaN (porque el modelo no aprendió nada).

---

## Problema 3: Estructura de Outputs Incorrecta

Actualmente:
```
output/
├── train/PNG/
│   ├── training_curve_dit.png       ✅ OK
│   └── validation_real_vs_gen.png   ❌ DEBERÍA ESTAR EN validation/PNG
├── validation/MAT/
│   └── val_gen_*.mat                ✅ OK
└── ...
```

Debería ser:
```
output/
├── train/PNG/
│   ├── training_curve_dit.png       ✅ Curva de pérdida
│   └── (comparación real vs recon del training - NO IMPLEMENTADO)
├── train/MAT/
│   └── (ROIs originales durante training - NO IMPLEMENTADO)
├── validation/PNG/
│   ├── error_vs_gap.png             ❌ NO EXISTE
│   └── validation_real_vs_gen.png   ❌ AQUÍ (no en TRAIN_PNG)
└── validation/MAT/
    └── val_gen_*.mat                ✅ OK
```

---

## 🔧 Soluciones Recomendadas

### Fix 1: Entrenar el VAE (CRÍTICO)

Opción A: Entrenar VAE separadamente antes de DiT
```python
# Añadir a train.py (antes del loop de DiT)
vae_loss_fn = nn.MSELoss()
vae_optimizer = optim.Adam(model.vae.parameters(), lr=1e-3)

for vae_epoch in range(50):
    for batch_img, _ in loader:
        batch_img = batch_img.to(device)
        
        # Forward VAE
        latent = model.vae_encoder(batch_img)
        recon = model.vae_decoder(latent)
        
        # Loss: reconstrucción
        vae_loss = vae_loss_fn(recon, batch_img)
        
        vae_optimizer.zero_grad()
        vae_loss.backward()
        vae_optimizer.step()
    
    if vae_epoch % 10 == 0:
        print(f"VAE Epoch {vae_epoch}, Loss: {vae_loss.item():.4f}")
```

Opción B: Usar VAE pre-entrenado (mejor)
```python
# Cargar VAE entrenado en UNet_AE
vae_state = torch.load("../unet_ae_modular/output/vae_encoder.pt")
model.vae_encoder.load_state_dict(vae_state['encoder'])
model.vae_decoder.load_state_dict(vae_state['decoder'])

# Congelar VAE (no entrenar, sólo usar)
for param in model.vae_encoder.parameters():
    param.requires_grad = False
for param in model.vae_decoder.parameters():
    param.requires_grad = False
```

### Fix 2: Reorganizar Outputs (ESTRUCTURA)

Cambios en `train.py`:

1. **Mover validation PNG a carpeta correcta** (línea ~385):
```python
# ❌ ANTES
fig_val.savefig(os.path.join(TRAIN_PNG_FOLDER, "validation_real_vs_gen.png"), dpi=150)

# ✅ DESPUÉS
fig_val.savefig(os.path.join(VALIDATION_PNG_FOLDER, "validation_real_vs_gen.png"), dpi=150)
```

2. **Añadir error vs gap plot** (en validation):
```python
# Después de calcular todos los errors en la sección de validación
fig_err, ax = plt.subplots(figsize=(10, 6))
ax.scatter(all_gaps, all_maes, alpha=0.6, s=100)
ax.set_xlabel("Gap entre ángulos (°)")
ax.set_ylabel("MAE (dB)")
ax.set_title("Error vs Gap — Validación")
ax.grid(True, alpha=0.3)
fig_err.tight_layout()
fig_err.savefig(os.path.join(VALIDATION_PNG_FOLDER, "error_vs_gap.png"), dpi=150)
plt.close(fig_err)
```

3. **Añadir comparación training PNG** (durante epochs):
```python
# En el loop de training (cada 50 epochs)
if epoch % 50 == 0:
    with torch.no_grad():
        sample_batch = next(iter(loader))[0][:4].to(device)
        # Generar reconstrucciones
        # Plotear y guardar en TRAIN_PNG_FOLDER
```

---

## 📋 Checklist de Fixes

### Priority 1 (CRÍTICO — Sin esto no funciona)
- [ ] Entrenar VAE (Opción A) O cargar VAE pre-entrenado (Opción B)
- [ ] Verificar que loss baja en cada época (debería estar < 0.5 en época 50+)

### Priority 2 (Estructura)
- [ ] Mover validation PNG a `VALIDATION_PNG_FOLDER`
- [ ] Añadir error vs gap plot en `VALIDATION_PNG_FOLDER`
- [ ] Verificar que `TRAIN_MAT_FOLDER` y `VALIDATION_MAT_FOLDER` se usan correctamente

### Priority 3 (Opcional pero mejora)
- [ ] Añadir comparación training PNG (muestras cada 50 epochs)
- [ ] Añadir histogramas de loss/LR

---

## 🎯 Recomendación Inmediata

Creo que el mejor fix es:

**Opción B: Cargar VAE Pre-entrenado de UNet**

Porque:
1. El VAE ya fue entrenado en UNet_AE (está en `../unet_ae_modular/`)
2. Es más rápido (no perder 50 epochs entrenando VAE)
3. El VAE ya conoce bien los datos oceanográficos
4. DiT sólo entrena en espacio latent (mucho más rápido)

---

## 📊 Impacto Esperado Tras Fix

**Sin fix**:
- Loss: ~1.0 (no converge)
- Validación: NaN (no funciona)
- Outputs: Blancos/vacíos

**Con fix VAE + estructura corregida**:
- Loss: 0.1-0.5 (converge bien)
- Validación: MAE ~1-5 dB (razonable)
- Outputs: PNG en carpetas correctas, imágenes significativas

---

**Próximo paso**: ¿Quieres que implemente la Opción B (cargar VAE pre-entrenado)?
