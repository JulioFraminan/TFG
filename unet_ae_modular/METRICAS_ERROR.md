# 📊 Métricas de Error Implementadas

## Resumen de cambios
Se han añadido múltiples métricas de error para complementar el análisis actual:

### Métricas Implementadas

| Métrica | Fórmula | Utilidad | Rango |
|---------|---------|---------|-------|
| **MAE** (Mean Absolute Error) | `mean(\|actual - pred\|)` | Error promedio absoluto. Fácil de interpretar, simetría de errores. | [0, ∞) dB |
| **RMSE** (Root Mean Square Error) | `sqrt(mean((actual - pred)²))` | **Penaliza más los errores grandes** (outliers). Útil para detectar predicciones malas. | [0, ∞) dB |
| **MAPE** (Mean Absolute Percentage Error) | `100 × mean(\|actual - pred\| / \|actual\|)` | **Error relativo en %**. Muestra qué % de error cometemos respecto al valor real. Escala independiente. | [0, ∞) % |
| **Max Error** | `max(\|actual - pred\|)` | Peor caso. Importante para aplicaciones que requieren garantías. | [0, ∞) dB |

---

## 📌 Interpretación por métrica

### 1. MAE (Error Absoluto Medio)
```
MAE = 3.5 dB → En promedio, nuestro modelo se equivoca 3.5 dB
```
**Ventajas:**
- Simple e intuitivo
- Una unidad es una unidad de error
- No penaliza outliers

**Desventajas:**
- No refleja si hay errores grandes dispersos
- No indica proporción relativa de error

---

### 2. RMSE (Error Cuadrático Medio)
```
RMSE = 4.2 dB → Hay errores significativos (RMSE > MAE indica variabilidad)
```
**Ventajas:**
- **Penaliza errores grandes más que el MAE**
- Si `RMSE >> MAE` → hay outliers (predicciones muy malas en algunos píxeles)
- Métrica estándar en ML

**Desventajas:**
- Unidades al cuadrado (menos intuitivo)
- Puede ser dominado por pocos píxeles / outliers

**Relación RMSE vs MAE:**
- `RMSE ≈ MAE` → errores uniformes
- `RMSE >> MAE` → existen errores grandes concentrados (problemas locales)

---

### 3. MAPE (Error Absoluto Porcentual Medio)
```
MAPE = 15.3% → Nuestro modelo predice con ~15.3% de error relativo
```
**Ventajas:**
- **Métrica normalizada**, independiente de escala
- Útil cuando los valores varían mucho
- Fácil de comunicar ("error relativo del 15%")

**Desventajas:**
- **Indefinido si actual ≈ 0** (dividimos por ~0)
- Implementación actual: ignoramos píxeles con |actual| < 1e-6
- Puede dar resultados confusos si hay muchos ceros

**Caso de uso:**
- Campo sonoro con valores entre -50 dB y +10 dB
- MAPE muestra el % de desviación respecto al valor real
- Especialmente útil para comparar calidad entre diferentes rangos

---

### 4. Max Error
```
Max Error = 8.9 dB → El peor píxel tiene un error de 8.9 dB
```
Útil para verificar que **no hay catástrofes locales**.

---

## 🔍 Cómo usar estas métricas juntas

### Scenario 1: Modelo bueno uniforme
```
MAE = 1.5 dB
RMSE = 1.6 dB  ✓ RMSE ≈ MAE
MAPE = 3.2%
Max = 3.2 dB
```
→ Errores uniformes y pequeños. **Modelo estable.**

### Scenario 2: Modelo con problemas locales
```
MAE = 2.0 dB
RMSE = 5.5 dB  ⚠️ RMSE >> MAE (variabilidad alta)
MAPE = 8.1%
Max = 15.2 dB
```
→ Hay zonas donde falla mucho. **Investigar dónde fallan (zonas de baja SNR, bordes, etc.)**

### Scenario 3: Errores relativos problemáticos
```
MAE = 1.0 dB
RMSE = 1.2 dB
MAPE = 45%  ⚠️ MAPE muy alto
Max = 2.0 dB
```
→ Los valores reales son muy pequeños. **El error relativo es grande aunque el absoluto es manejable.**

---

## 📈 Visualizaciones generadas

### En PNGs de Entrenamiento (train/PNG/)
**Comparación Original vs Reconstrucción:**
```
[Original] [Reconstrución] [|Error|]
                            MAE=2.1 dB  MAPE=5.3%  RMSE=2.4 dB  Max=4.8 dB
```

### En PNGs de Validación (validation/PNG/)
**Real vs Generado:**
```
[Real] [Generado] [|Error|]
                  MAE=2.8 dB  MAPE=6.1%  RMSE=3.5 dB  Max=7.2 dB
```

### Gráfico Error vs Gap (validation/PNG/error_vs_gap.png)
**Dos subplots:**
1. **MAE, RMSE, Max vs Gap entre vecinos**
   - Muestra cómo crece el error con la distancia angular entre vecinos de entrenamiento
   - RMSE > MAE → indica cuando el modelo se desestabiliza

2. **MAPE vs Gap**
   - Error relativo
   - Complementa el gráfico anterior

---

## 🎯 Recomendaciones para análisis

### Si encuentras:
1. **MAPE indefinido (undef)** en algunos planos
   - Significa que ese plano tiene píxeles con valores muy cercanos a cero
   - Considera revisar el rango dinámico del plano
   - Posiblemente ruido de fondo muy bajo en esa zona

2. **RMSE >> MAE**
   - Hay errores concentrados en zonas específicas
   - Investiga si son bordes, zonas de baja SNR, etc.

3. **MAPE consistentemente alto pero MAE bajo**
   - Los valores reales en esa zona son muy pequeños
   - Error absoluto pequeño pero proporcionalmente grande
   - Puede ser problemático si necesitas precisión relativa

---

## 🔧 Otras métricas a considerar en futuro

### SSIM (Structural Similarity Index)
```python
# Mide similitud estructural de imágenes (no solo píxel a píxel)
from skimage.metrics import structural_similarity as ssim
ssim_val = ssim(real_tl, gen_tl, data_range=max_val)
```
- **Rango:** [-1, 1] (1 = idéntico)
- **Útil para:** Validar que la estructura acústica (cambios graduales) es correcta
- **Ventaja:** Menos sensible a pequeños desplazamientos que MSE/MAE

### R² (Coeficiente de Determinación)
```python
# Proporción de varianza explicada
ss_res = np.sum((real_tl - gen_tl)**2)
ss_tot = np.sum((real_tl - real_tl.mean())**2)
r2 = 1 - (ss_res / ss_tot)
```
- **Rango:** [0, 1+] (1 = perfecto, <0 = peor que simplemente usar la media)
- **Útil para:** Entender si el modelo captura la variabilidad de los datos

---

## 📝 Archivos modificados

- **data_utils.py**
  - Nueva función `compute_error_metrics(actual, predicted)` → devuelve dict con todas las métricas

- **train.py**
  - Importa `compute_error_metrics`
  - Usa la función en comparación original vs reconstrucción
  - Usa la función en validación
  - Gráfico error_vs_gap.png ahora tiene 2 subplots (absoluto + relativo)
  - Resumen final imprime MAE, RMSE, MAPE

