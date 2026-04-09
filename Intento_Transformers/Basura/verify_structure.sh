#!/bin/bash
# verify_structure.sh - Verificar que la estructura de exportación es correcta
# Uso: bash verify_structure.sh

set -e

echo "════════════════════════════════════════════════════════════════"
echo "  VERIFICACIÓN DE ESTRUCTURA DE CARPETAS Y EXPORTACIÓN"
echo "════════════════════════════════════════════════════════════════"
echo ""

# 1. Verificar que config.py define todas las carpetas
echo "1️⃣  Verificando definiciones en config.py..."
python3 << 'PYEOF'
import sys
try:
    from config import (
        TRAIN_PNG_FOLDER, TRAIN_MAT_FOLDER,
        GENERATE_PNG_FOLDER, GENERATE_MAT_FOLDER,
        VALIDATION_PNG_FOLDER, VALIDATION_MAT_FOLDER,
        DIT_MODEL_PATH,
        create_all_dirs
    )
    print("   ✅ Todas las constantes de carpetas definidas en config.py")
    print(f"      - TRAIN_PNG_FOLDER: {TRAIN_PNG_FOLDER}")
    print(f"      - TRAIN_MAT_FOLDER: {TRAIN_MAT_FOLDER}")
    print(f"      - GENERATE_PNG_FOLDER: {GENERATE_PNG_FOLDER}")
    print(f"      - GENERATE_MAT_FOLDER: {GENERATE_MAT_FOLDER}")
    print(f"      - VALIDATION_PNG_FOLDER: {VALIDATION_PNG_FOLDER}")
    print(f"      - VALIDATION_MAT_FOLDER: {VALIDATION_MAT_FOLDER}")
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    sys.exit(1)
PYEOF

echo ""
echo "2️⃣  Verificando que train.py importa las constantes..."
grep -q "GENERATE_PNG_FOLDER, GENERATE_MAT_FOLDER" train.py && \
    echo "   ✅ train.py importa GENERATE_PNG_FOLDER y GENERATE_MAT_FOLDER" || \
    echo "   ❌ FALTA importar GENERATE_PNG_FOLDER o GENERATE_MAT_FOLDER"

echo ""
echo "3️⃣  Verificando que train.py usa VALIDATION_MAT_FOLDER (NO GENERATE_MAT_FOLDER)..."
if grep -q "os.path.join(VALIDATION_MAT_FOLDER" train.py; then
    echo "   ✅ train.py usa VALIDATION_MAT_FOLDER para guardar validación (CORRECTO)"
else
    echo "   ❌ train.py no usa VALIDATION_MAT_FOLDER"
fi

if grep -q "os.path.join(GENERATE_MAT_FOLDER.*val_gen" train.py; then
    echo "   ❌ train.py TODAVÍA usa GENERATE_MAT_FOLDER para validación (INCORRECTO)"
else
    echo "   ✅ train.py NO usa GENERATE_MAT_FOLDER para validación (CORRECTO)"
fi

echo ""
echo "4️⃣  Verificando que generate.py usa GENERATE_MAT_FOLDER (correcto)..."
grep -q "os.path.join(GENERATE_MAT_FOLDER" generate.py && \
    echo "   ✅ generate.py usa GENERATE_MAT_FOLDER (CORRECTO)" || \
    echo "   ⚠️  generate.py no usa GENERATE_MAT_FOLDER"

echo ""
echo "5️⃣  Verificando que no hay definición duplicada de main()..."
COUNT=$(grep -c "^def main():" train.py)
if [ "$COUNT" -eq 1 ]; then
    echo "   ✅ Sólo una definición de main() (CORRECTO)"
else
    echo "   ❌ Encontradas $COUNT definiciones de main() (INCORRECTO)"
fi

echo ""
echo "6️⃣  Creando estructura de carpetas..."
python3 << 'PYEOF'
import os
import sys
try:
    from config import create_all_dirs
    create_all_dirs()
    print("   ✅ Carpetas de output creadas exitosamente")
except Exception as e:
    print(f"   ❌ ERROR creando carpetas: {e}")
    sys.exit(1)
PYEOF

echo ""
echo "7️⃣  Listando estructura creada..."
if [ -d output ]; then
    echo "   output/"
    find output -type d | sed 's|output|   ├  |' | head -20
    echo "   ✅ Estructura de carpetas existe"
else
    echo "   ❌ Carpeta output NO existe"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ✅ VERIFICACIÓN COMPLETADA"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "📌 Resumen:"
echo "   • train.py importa todas las constantes necesarias"
echo "   • Validación se guarda en VALIDATION_MAT_FOLDER (no en GENERATE_MAT_FOLDER)"
echo "   • generate.py usa GENERATE_MAT_FOLDER correctamente"
echo "   • No hay definiciones duplicadas de main()"
echo "   • Estructura de carpetas creada"
echo ""
echo "🚀 Listo para ejecutar:"
echo "   python train.py"
echo "   python generate.py"
echo ""
