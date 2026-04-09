# ✅ CHECKLIST EJECUTABLE - Migración a model_clean.py

## 📋 Pre-Migración

- [ ] **Leer RESUMEN_EJECUTIVO.md** (5 min)
  - [ ] Entender el problema de gráficos vacíos
  - [ ] Entender por qué ocurrió
  - [ ] Entender cómo se arregla

- [ ] **Leer CAMBIOS_EXACTOS.md** (10 min)
  - [ ] Ver cambios exactos requeridos
  - [ ] Ver orden de cambios
  - [ ] Ver verificaciones post-cambios

- [ ] **Verificar que model_clean.py existe**
  ```bash
  ls -la /home/j.framinan/Intento_Transformers/model_clean.py
  # Debe mostrar el archivo
  ```

- [ ] **Crear backup de archivos originales**
  ```bash
  cd /home/j.framinan/Intento_Transformers
  cp model.py model_old.py
  cp train.py train_old.py
  cp generate.py generate_old.py
  ```

---

## 🔄 Migración (Paso a Paso)

### Paso 1: Reemplazar model.py
```bash
cd /home/j.framinan/Intento_Transformers
cp model_clean.py model.py
echo "✓ model.py reemplazado"
```

**Verificación:**
```bash
ls -la model.py model_clean.py
# Ambos deben existir
diff model.py model_clean.py | head -5
# Deben ser idénticos
```

- [ ] model.py reemplazado correctamente

### Paso 2: Modificar train.py (línea 223)

**Encontrar la línea:**
```bash
grep -n "model.forward_latent" /home/j.framinan/Intento_Transformers/train.py
# Debe mostrar una línea ~223
```

**Cambiar manualmente o con sed:**
```bash
# Opción 1: sed (automático)
sed -i 's/noise_pred_z = model\.forward_latent/noise_pred_z = model/' \
  /home/j.framinan/Intento_Transformers/train.py

# Opción 2: Abrir en editor y cambiar manualmente
# Line 223: cambiar "model.forward_latent(z_t, t, batch_ang)"
#           a "model(z_t, t, batch_ang)"
```

**Verificación:**
```bash
grep -A2 -B2 "noise_pred_z = model(" /home/j.framinan/Intento_Transformers/train.py | head -10
# Debe mostrar la línea sin ".forward_latent"
```

- [ ] train.py modificado correctamente

### Paso 3: Modificar generate.py (línea 32)

**Encontrar la línea:**
```bash
grep -n "model.forward_latent" /home/j.framinan/Intento_Transformers/generate.py
# Debe mostrar una línea en ddim_sample_latent()
```

**Cambiar manualmente o con sed:**
```bash
# Opción 1: sed (automático)
sed -i 's/eps_pred = model\.forward_latent/eps_pred = model/' \
  /home/j.framinan/Intento_Transformers/generate.py

# Opción 2: Abrir en editor y cambiar manualmente
# Buscar: "eps_pred = model.forward_latent(z_t, t_tensor, conditioning)"
# Cambiar a: "eps_pred = model(z_t, t_tensor, conditioning)"
```

**Verificación:**
```bash
grep "eps_pred = model(" /home/j.framinan/Intento_Transformers/generate.py
# Debe mostrar la línea sin ".forward_latent"
```

- [ ] generate.py modificado correctamente

---

## ✅ Post-Migración (Verificaciones)

### Verificación 1: Compilación
```bash
cd /home/j.framinan/Intento_Transformers
python -c "from model import DiT; print('✓ model imports OK')"
```
- [ ] Importación exitosa

### Verificación 2: Instantiación
```bash
python << 'EOF'
from model import DiT

try:
    model = DiT(
        latent_channels=128,
        hidden_size=128,
        depth=3,
        num_heads=2,
        num_diffusion_steps=1000,
    )
    print("✓ DiT instantiates OK")
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)
EOF
```
- [ ] Instantiación exitosa

### Verificación 3: Forward Pass
```bash
python << 'EOF'
import torch
from model import DiT

model = DiT(
    latent_channels=128,
    hidden_size=128,
    depth=3,
    num_heads=2,
    num_diffusion_steps=1000,
)
model.eval()

z = torch.randn(2, 128, 88, 250)
t = torch.randint(0, 1000, (2,))
cond = torch.randn(2, 1)

with torch.no_grad():
    output = model(z, t, cond)

if output.shape == z.shape:
    print(f"✓ Forward pass OK: {z.shape} → {output.shape}")
else:
    print(f"✗ Shape mismatch: {output.shape} != {z.shape}")
    exit(1)
EOF
```
- [ ] Forward pass exitoso

### Verificación 4: No hay forward_latent()
```bash
python << 'EOF'
from model import DiT

model = DiT()

if hasattr(model, 'forward_latent'):
    print("✗ forward_latent() still exists!")
    exit(1)
else:
    print("✓ forward_latent() eliminated as expected")
EOF
```
- [ ] forward_latent() no existe

### Verificación 5: Código sin VAE encoder/decoder
```bash
python << 'EOF'
from model import DiT

model = DiT()

if hasattr(model, 'vae_encoder') or hasattr(model, 'vae_decoder'):
    print("✗ VAE still in model!")
    exit(1)
else:
    print("✓ VAE removed from model as expected")
EOF
```
- [ ] VAE no en modelo

### Verificación 6: No cambios en train.py/generate.py (excepto líneas)
```bash
# Contar diferencias
diff -u train_old.py train.py | grep "^[+-]" | wc -l
# Debe ser pequeño (~2 líneas)

diff -u generate_old.py generate.py | grep "^[+-]" | wc -l
# Debe ser pequeño (~2 líneas)
```
- [ ] Solo cambios mínimos

---

## 🚀 Ejecución de Prueba

### Opción 1: Quick Test (sin entrenar)
```bash
cd /home/j.framinan/Intento_Transformers

# Ver que todo importa
python << 'EOF'
from model import DiT
from train import main as train_main
from generate import generate_samples_ddim

print("✓ All imports successful")
print("✓ Ready for training")
EOF
```
- [ ] Quick test pasado

### Opción 2: Entrenar Completo (opcional)
```bash
cd /home/j.framinan/Intento_Transformers

# Entrenar (este toma tiempo)
python train.py 2>&1 | tee training_log.txt

# Verificar resultados
tail -50 training_log.txt
# Buscar: "Entrenamiento completado"
# Buscar: "Validación visual guardada"
```
- [ ] Entrenamiento completado
- [ ] Gráficos generados

---

## 🔍 Verificación de Resultados

Después de entrenar, verificar:

```bash
# 1. ¿Existe la curva de entrenamiento?
ls -la output/train/PNG/training_curve_dit.png

# 2. ¿Existe validación?
ls -la output/validation/PNG/validation_real_vs_gen.png

# 3. ¿Tiene contenido el gráfico (no está vacío)?
# Abrir: output/validation/PNG/validation_real_vs_gen.png
# ✓ Debe tener colores/patrones oceanográficos
# ✗ NO debe estar completamente blanco/vacío
```

- [ ] training_curve_dit.png existe
- [ ] validation_real_vs_gen.png existe
- [ ] Los gráficos tienen contenido (no vacíos)

---

## 🆘 Si Algo Sale Mal

### Error: "No module named 'forward_latent'"
```bash
# Verificar cambios en train.py y generate.py
grep "forward_latent" /home/j.framinan/Intento_Transformers/train.py
grep "forward_latent" /home/j.framinan/Intento_Transformers/generate.py

# Ambos deben estar vacíos. Si no:
# Volver a hacer los cambios en línea 223 de train.py
# y línea 32 de generate.py
```

### Error: "model doesn't have attribute 'vae_encoder'"
```bash
# En train.py, revisar línea ~163
# Debe cargar VAE: vae_checkpoint = torch.load(vae_path)
# Eso es normal - train.py carga VAE, no model

# Si hay error aquí, verificar que VAE existe:
ls -la ../unet_ae_modular/output/unet_ae_model.pt
```

### Error: "forward_latent is not a valid method"
```bash
# Significa que los cambios en train.py o generate.py no se aplicaron
# Repetir Paso 2 y Paso 3 de la migración
```

### Revertir si es necesario
```bash
cd /home/j.framinan/Intento_Transformers
cp model_old.py model.py
cp train_old.py train.py
cp generate_old.py generate.py
```

---

## 📊 Resumen de Cambios

| Archivo | Cambio | Líneas |
|---------|--------|--------|
| model.py | Reemplazar completo | 639 → ~380 |
| train.py | 1 línea (223) | +0 -1 |
| generate.py | 1 línea (32) | +0 -1 |
| config.py | No cambios | - |
| data_utils.py | No cambios | - |

**Total:** 2 líneas de cambio + 1 archivo reemplazado = 5 minutos

---

## ✅ Checklist Final

- [ ] Pre-migración leída y entendida
- [ ] Backups creados
- [ ] model.py reemplazado
- [ ] train.py modificado (1 línea)
- [ ] generate.py modificado (1 línea)
- [ ] Todas las verificaciones pasadas
- [ ] Quick test exitoso (opcional)
- [ ] Entrenamiento completado (opcional)
- [ ] Gráficos tienen contenido (no vacíos)

---

## 🎉 Listo!

Si completaste todas las verificaciones, **la migración fue exitosa**.

El código ahora es:
- ✅ Más limpio
- ✅ Más mantenible
- ✅ Menos confuso
- ✅ Genera gráficos correctos

**Próximo paso:** `python train.py` y observar que los gráficos tienen contenido 🚀
