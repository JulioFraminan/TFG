# CHECKLIST DE AUDITORÍA — Estructura de Exportación

## 🔍 Audit realizado: 9 de Abril de 2026

### ✅ Errores Encontrados y Corregidos

| ID | Archivo | Línea | Error | Fix | Estado |
|----|---------|----|-------|-----|--------|
| E1 | `train.py` | 386 | `os.path.join(GENERATE_MAT_FOLDER, ...)` en sección de validación | Cambiar a `VALIDATION_MAT_FOLDER` | ✅ FIXED |
| E2 | `train.py` | 31-33 | Importaciones incompletas (faltaba `GENERATE_*` y `VALIDATION_*`) | Añadir a import desde config | ✅ FIXED |
| E3 | `train.py` | 38-50 | Función `def main()` duplicada | Eliminar la primera definición vacía | ✅ FIXED |

---

### ✅ Auditoría de Exportaciones (Current State)

#### `train.py` — Exportaciones

```python
# ENTRENAMIENTO
fig.savefig(os.path.join(TRAIN_PNG_FOLDER, "training_curve_dit.png"), dpi=150)
# ✅ Correcto: TRAIN_PNG_FOLDER es para curva de training

# VALIDACIÓN (si existe input/validation/)
fig_val.savefig(os.path.join(TRAIN_PNG_FOLDER, "validation_real_vs_gen.png"), dpi=150)
# ✅ Correcto: PNG de validación va a TRAIN_PNG_FOLDER (son comparativas visuales del training)

gen_mat_path = os.path.join(VALIDATION_MAT_FOLDER, f"val_gen_{val_file}")
save_mat(gen_mat_path, gen_tl)
# ✅ Correcto (FIXED): MAT de validación va a VALIDATION_MAT_FOLDER

# MODELO
torch.save({...}, DIT_MODEL_PATH)
# ✅ Correcto: Checkpoint va a OUTPUT_FOLDER/dit_model_latest.pt
```

#### `generate.py` — Exportaciones

```python
# GENERACIÓN
mat_path = os.path.join(GENERATE_MAT_FOLDER, f"plano_angulo_{target_angle:+.1f}.mat")
save_mat(mat_path, gen_tl)
# ✅ Correcto: MAT generado va a GENERATE_MAT_FOLDER

png_path = os.path.join(GENERATE_PNG_FOLDER, f"plano_angulo_{target_angle:+.1f}.png")
fig.savefig(png_path, dpi=150, bbox_inches="tight")
# ✅ Correcto: PNG de generación va a GENERATE_PNG_FOLDER
```

---

## 📋 Matriz de Responsabilidades (Quién Exporta Dónde)

| Script | Tipo Exp. | Carpeta | Notas |
|--------|-----------|---------|-------|
| `train.py` | PNG training | `TRAIN_PNG_FOLDER` | Curva loss + LR |
| `train.py` | PNG validation | `TRAIN_PNG_FOLDER` | Real vs Gen vs Error (durante training) |
| `train.py` | MAT validation | `VALIDATION_MAT_FOLDER` | Planos generados en validación |
| `train.py` | Checkpoint | `OUTPUT_FOLDER` | `dit_model_latest.pt` |
| `generate.py` | PNG generation | `GENERATE_PNG_FOLDER` | Semilla vs Gen vs Error |
| `generate.py` | MAT generation | `GENERATE_MAT_FOLDER` | Planos sintetizados |
| (futuro) | VAE checkpoint | `OUTPUT_FOLDER` | `vae_encoder.pt` |

---

## 🛡️ Medidas Preventivas Implementadas

1. ✅ **Importaciones explícitas en `train.py`**:
   - Líneas 31-33: Todos los `*_FOLDER` y `*_MODEL_PATH` importados desde `config.py`.
   - Evita NameError por variables no definidas.

2. ✅ **Comentarios inline**:
   - Cada `savefig()` y `save_mat()` tiene comentario indicando qué se guarda y por qué.

3. ✅ **Función `create_all_dirs()` centralizada**:
   - En `config.py`, crea TODAS las carpetas de una vez.
   - Llamada al inicio de `main()` en `train.py` y `generate.py`.

4. ✅ **Documentación**: Archivo `EXPORT_STRUCTURE.md` con matriz de exportación.

---

## 🔧 Procedimiento de Validación

Para verificar que TODO funciona correctamente:

```bash
# 1. Verificar que config.py define todas las carpetas
python -c "from config import *; import os; paths = [TRAIN_PNG_FOLDER, TRAIN_MAT_FOLDER, VALIDATION_PNG_FOLDER, VALIDATION_MAT_FOLDER, GENERATE_PNG_FOLDER, GENERATE_MAT_FOLDER]; print('✓ All paths defined'); [print(f'  {p}') for p in paths]"

# 2. Verificar que create_all_dirs() funciona
python -c "from config import create_all_dirs; create_all_dirs(); import os; folders = [d for d in os.walk('output')]; print(f'✓ Created {len(list(os.walk(\"output\")))} folders')"

# 3. Ejecutar train.py (si tienes datos)
# python train.py

# 4. Ejecutar generate.py (requiere model entrenado)
# python generate.py
```

---

## ⚠️ Errores Comunes a Evitar

| Error | Síntoma | Prevención |
|-------|---------|-----------|
| `NameError: name 'GENERATE_MAT_FOLDER' is not defined` | Falta importar variable desde config | Revisar imports al inicio del archivo |
| Archivo guardado en carpeta incorrecta | Datos dispersos, confusión en outputs | Usar matriz de responsabilidades (arriba) |
| Carpeta no existe (`FileNotFoundError`) | Error al guardar | Llamar `create_all_dirs()` antes de cualquier save |
| Path relativo vs absoluto | Inconsistencias entre entornos | Usar `os.path.join()` siempre, nunca paths hardcoded |

---

## 📊 Estado General

| Aspecto | Estado | Detalles |
|--------|--------|----------|
| **Imports en train.py** | ✅ OK | Todas las constantes importadas |
| **Imports en generate.py** | ✅ OK | Todas las constantes importadas |
| **Exportaciones train.py** | ✅ OK | PNG y MAT van a carpetas correctas |
| **Exportaciones generate.py** | ✅ OK | PNG y MAT van a carpetas correctas |
| **Carpetas creadas** | ✅ OK | `create_all_dirs()` en config.py |
| **Documentación** | ✅ OK | EXPORT_STRUCTURE.md actualizado |
| **Código limpio** | ✅ OK | Duplicados eliminados, comentarios claros |

---

## 🚀 Próximas Mejoras Recomendadas

1. Añadir assertions defensivos antes de cada `save_mat()`:
   ```python
   assert os.path.exists(VALIDATION_MAT_FOLDER), f"Missing {VALIDATION_MAT_FOLDER}"
   ```

2. Crear logs de auditoría que registren **dónde** se guardó cada archivo:
   ```python
   with open(os.path.join(OUTPUT_FOLDER, "save_log.txt"), "a") as f:
       f.write(f"{timestamp} | VALIDATION_MAT | {gen_mat_path}\n")
   ```

3. Script de validación post-training que verifique integridad:
   - Contar archivos en cada carpeta
   - Verificar que no hay archivos huérfanos

---

**Responsable de auditoría**: Sistema automático  
**Fecha**: 9 de Abril de 2026  
**Próxima revisión**: Antes de próximo training en cluster  
**Entorno**: test_env (conda)
