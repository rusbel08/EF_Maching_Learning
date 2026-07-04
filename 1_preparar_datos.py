# ============================================================
# 1_preparar_datos.py
# Descarga dataset de enfermedades en arándanos desde Roboflow
# Proyecto: PlantSeg Disease Detection
# Clases: sano | botrytis | antracnosis
# ============================================================

import os
import shutil
import csv
from roboflow import Roboflow

API_KEY    = "MmUpyJ8o5JAV1h4DD5Ew"
DATASET_DIR = "dataset"

# Mapeo nombre en CSV → carpeta local
CLASES = {
    "Blueberry healthy"         : "sano",
    "blueberry botrytis blight" : "botrytis",
    "blueberry anthracnose"     : "antracnosis",
}

# ── Descarga ─────────────────────────────────────────────────
print("Conectando con Roboflow...")
rf      = Roboflow(api_key=API_KEY)
project = rf.workspace("iqra-national-university-hayatabad-peshawar").project("plantseg-disease-detection")
dataset = project.version(1).download("multiclass")
print(f"Dataset descargado en: {dataset.location}")

# ── Filtrar imágenes de arándano usando _classes.csv ────────
splits_map = {"train": "train", "valid": "val", "test": "test"}

for split_origen, split_dest in splits_map.items():
    carpeta = os.path.join(dataset.location, split_origen)
    csv_path = os.path.join(carpeta, "_classes.csv")

    if not os.path.exists(csv_path):
        print(f"⚠ No se encontró _classes.csv en {split_origen}")
        continue

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames

        # Buscar columnas que correspondan a nuestras clases
        col_map = {}
        for clase_rf in CLASES:
            for h in headers:
                if clase_rf.lower() in h.lower():
                    col_map[h] = CLASES[clase_rf]
                    break

        if not col_map:
            print(f"⚠ No se encontraron clases de arándano en {split_origen}")
            print(f"  Columnas disponibles: {', '.join(headers[:10])}...")
            continue

        print(f"\n{split_dest.upper()} — clases encontradas: {list(col_map.values())}")

        contadores = {c: 0 for c in CLASES.values()}

        for row in reader:
            for col, clase_local in col_map.items():
                if row.get(col, "0").strip() == "1":
                    destino = os.path.join(DATASET_DIR, split_dest, clase_local)
                    os.makedirs(destino, exist_ok=True)
                    img_src = os.path.join(carpeta, row["filename"])
                    if os.path.exists(img_src):
                        shutil.copy2(img_src, os.path.join(destino, row["filename"]))
                        contadores[clase_local] += 1

        for clase, n in contadores.items():
            print(f"  {clase:15s}: {n:4d} imágenes")

# ── Resumen ──────────────────────────────────────────────────
print("\n" + "="*50)
print("RESUMEN FINAL")
print("="*50)
total = 0
for split in ["train", "val", "test"]:
    print(f"\n{split.upper()}:")
    for clase in ["sano", "botrytis", "antracnosis"]:
        ruta = os.path.join(DATASET_DIR, split, clase)
        if os.path.exists(ruta):
            n = len([f for f in os.listdir(ruta) if f.endswith((".jpg",".jpeg",".png"))])
            total += n
            print(f"  {clase:15s}: {n:4d} imágenes")

print(f"\nTotal: {total} imágenes")
print("✓ Listo. Ejecuta 2_entrenar_modelo.py")