# 🔬 Análisis Avanzado - Métricas Adicionales Recomendadas

## Resumen Ejecutivo

He implementado **MAE, RMSE, MAPE** como solicitaste. Además, identifico 4 **análisis adicionales** que merecerían la pena estudiar para entender mejor la calidad del modelo:

---

## 1️⃣ SSIM (Structural Similarity Index) - ⭐ RECOMENDADO

### ¿Qué es?
Mide cuánto se parece la estructura visual de una imagen a otra, no solo píxel a píxel.

### Por qué es útil
- **Tu problema:** Reconstruir campos acústicos complejos
- **MAE mide:** Cada píxel independientemente
- **SSIM mide:** ¿El mapa de isolíneas acústicas tiene la misma forma? ¿Los gradientes están en los sitios correctos?

### Ejemplo
```
Caso 1: Desplazamiento pequeño (píxeles correctos pero 1-2 px desplazados)
  MAE: ALTO (cada píxel difiere)
  SSIM: ALTO (estructura idéntica)
  → Indica problema de alineación, no de contenido

Caso 2: Ruido aleatorio distribuido
  MAE: MEDIO
  SSIM: BAJO
  → Indica que la estructura no se mantiene
```

### Implementación (fácil)
```python
from skimage.metrics import structural_similarity as ssim

def compute_ssim(actual, predicted):
    """SSIM: 1 = idéntico, -1 = totalmente diferente, típico 0.7-0.99."""
    return ssim(actual, predicted, data_range=actual.max() - actual.min())

# En tu código:
errors['ssim'] = compute_ssim(real_tl, gen_tl)  # Rango [-1, 1]
```

**Rango esperado:** 0.70 - 0.99 (0.99 = casi perfecto)

---

## 2️⃣ Error en zonas de interés (ROI error) - ⭐ RECOMENDADO

### ¿Por qué?
Actualmente calculas error en toda la imagen. Pero:
- Los bordes pueden tener artefactos por el padding/extrapolación
- El centro puede ser más importante que los bordes
- Zonas de baja SNR pueden aumentar error falsamente

### Qué hacer
Calcular errores por **zonas de la imagen:**
```python
def compute_zonal_errors(actual, predicted):
    """Divide imagen en 9 zonas (3x3) y calcula MAE de cada una."""
    h, w = actual.shape
    h_third = h // 3
    w_third = w // 3
    
    zones = {}
    for i in range(3):
        for j in range(3):
            zone = actual[i*h_third:(i+1)*h_third, j*w_third:(j+1)*w_third]
            pred_zone = predicted[i*h_third:(i+1)*h_third, j*w_third:(j+1)*w_third]
            zones[f'zone_{i}_{j}'] = np.mean(np.abs(zone - pred_zone))
    
    return zones
```

**Ventaja:** Detectaría si el modelo tiene problemas en bordes, centros, arriba/abajo.

---

## 3️⃣ Análisis espectral (FFT) - ⭐ RECOMENDADO

### ¿Qué es?
Transformada de Fourier para ver si el modelo reproduce bien las frecuencias espaciales.

### Por qué es útil
- Detecta si el modelo genera suavizado artificial
- Muestra si faltan detalles de alta frecuencia
- Diagnostica degradación de resolución

### Implementación
```python
import numpy as np
from scipy import signal

def compare_spectra(actual, predicted):
    """Compara energía espectral en diferentes bandas de frecuencia."""
    
    # FFT 2D
    fft_actual = np.abs(np.fft.fft2(actual))
    fft_pred = np.abs(np.fft.fft2(predicted))
    
    # Energía por anillo de frecuencia
    h, w = actual.shape
    cy, cx = h//2, w//2
    
    results = {}
    for radius in [5, 10, 20, 50]:
        y, x = np.ogrid[:h, :w]
        mask = (x-cx)**2 + (y-cy)**2 <= radius**2
        energy_actual = np.sum(fft_actual[mask])
        energy_pred = np.sum(fft_pred[mask])
        results[f'freq_r{radius}'] = energy_pred / (energy_actual + 1e-8)
    
    return results  # Ratio pred/actual (1 = correcto)
```

**Interpretación:**
- Ratio < 0.8 → Pérdida de energía en esas frecuencias (suavizado)
- Ratio > 1.2 → Ruido o artefactos añadidos

---

## 4️⃣ Curva de confianza (percentiles de error) - ⭐ RECOMENDADO

### ¿Por qué?
MAE es media, RMSE es desviación. Pero ¿cuál es la distribución real?

### Implementación
```python
def error_percentiles(actual, predicted):
    """Calcula percentiles del error."""
    error_distribution = np.abs(actual.flatten() - predicted.flatten())
    
    return {
        'p25': np.percentile(error_distribution, 25),
        'p50': np.percentile(error_distribution, 50),  # mediana
        'p75': np.percentile(error_distribution, 75),
        'p90': np.percentile(error_distribution, 90),
        'p99': np.percentile(error_distribution, 99),
    }
```

**Ventaja:** Ver si 90% de píxeles tienen error < 1 dB pero 10% tienen > 5 dB

---

## 5️⃣ Correlación espacial del error

### ¿Qué es?
¿Los errores están aleatoriamente distribuidos o están concentrados en zonas?

### Por qué importa
- Error aleatorio → problema estadístico del modelo
- Error concentrado → problema sistemático (ex: bordes, baja SNR)

### Implementación (simple)
```python
diff = np.abs(actual - predicted)

# Autocorrelación espacial
spatial_corr = np.correlate(diff.flatten(), diff.flatten(), mode='full')

# Si alta → errores correlacionados espacialmente (problema sistemático)
# Si baja → errores distribuidos (mejor, más aleatorio)
```

---

## 🎯 Prioridad de implementación

| Métrica | Esfuerzo | Impacto | Prioridad |
|---------|----------|--------|-----------|
| SSIM    | ⭐       | ⭐⭐⭐ | **ALTA** |
| Zonal errors | ⭐ | ⭐⭐  | **ALTA** |
| FFT spectrum | ⭐⭐ | ⭐⭐  | MEDIA |
| Error percentiles | ⭐ | ⭐⭐  | **ALTA** |
| Spatial correlation | ⭐⭐ | ⭐ | BAJA |

**Recomendación:** Implementa SSIM + Zonal Errors + Percentiles. Son simples y muy informativos.

---

## 📋 Plan de Acción Sugerido

### Fase 1 (ya hecho)
✅ MAE, RMSE, MAPE

### Fase 2 (recomendado corto plazo)
1. SSIM para validar estructura visual
2. Zonal errors para identificar problemas geográficos
3. Error percentiles para entender distribución

### Fase 3 (opcional)
4. Análisis espectral FFT
5. Validación de suavizado artificial

---

## 💡 Pregunta de diagnóstico que responden

| Pregunta | Métrica |
|----------|---------|
| ¿Es el error pequeño? | MAE, RMSE |
| ¿El error es relativo? | MAPE |
| ¿Hay outliers? | RMSE vs MAE, percentiles |
| ¿La estructura se mantiene? | SSIM |
| ¿Dónde está el error? | Zonal errors |
| ¿Hay suavizado artificial? | FFT spectrum |
| ¿Es error aleatorio o sistemático? | Spatial correlation |

---

## 🔗 Referencias rápidas

### SSIM
```
Range: [-1, 1]
Target: > 0.95 (excelente), > 0.80 (aceptable)
```

### MAPE
```
Range: [0, ∞)
Target: < 5-10% (excelente), < 20% (aceptable)
Cuidado: Puede ser infinito si hay ceros
```

### RMSE vs MAE
```
RMSE = MAE → Errores uniformes
RMSE > 1.5 × MAE → Hay variabilidad (outliers)
```

---

**Recomendación:** Después de esta validación actual con MAE/RMSE/MAPE, considera adicionar SSIM y análisis zonal. Servirán para entender mejor dónde y por qué falla el modelo.

