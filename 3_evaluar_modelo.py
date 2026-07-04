# ============================================================
# 3_evaluar_modelo.py
# Evalúa el modelo entrenado usando el set de TEST
# (imágenes que el modelo nunca vio durante el entrenamiento)
# Genera: matriz de confusión, accuracy, precision, recall, F1
# ============================================================

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

# ── Configuración ────────────────────────────────────────────
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 8
DATASET_DIR = "dataset"
MODEL_PATH  = "models/modelo_arandano.keras"

# ── 1. Cargar el modelo entrenado y el mapeo de clases ───────
print("Cargando modelo...")
model = keras.models.load_model(MODEL_PATH)

with open("models/clases.json") as f:
    class_indices = json.load(f)

# Invertir el diccionario: {0: "antracnosis", 1: "botrytis", 2: "sano"}
idx_to_class = {v: k for k, v in class_indices.items()}
nombres_clases = [idx_to_class[i] for i in range(len(idx_to_class))]
print("Clases:", nombres_clases)

# ── 2. Cargar el set de TEST (nunca visto por el modelo) ────
test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,  # importante: para que coincidan predicciones con etiquetas
)

# ── 3. Predecir sobre el set de test ─────────────────────────
print("\nEvaluando sobre el set de TEST...")
predicciones_prob = model.predict(test_gen)
y_pred = np.argmax(predicciones_prob, axis=1)
y_true = test_gen.classes

# ── 4. Métricas generales ─────────────────────────────────────
acc = accuracy_score(y_true, y_pred)
print(f"\nAccuracy en TEST: {acc:.2%}")

print("\n" + "="*60)
print("REPORTE DE CLASIFICACIÓN (Precision, Recall, F1-score)")
print("="*60)
reporte = classification_report(
    y_true, y_pred, target_names=nombres_clases, digits=3
)
print(reporte)

# Guardar el reporte en un archivo de texto para el informe
with open("models/reporte_metricas.txt", "w", encoding="utf-8") as f:
    f.write(f"Accuracy en TEST: {acc:.2%}\n\n")
    f.write(reporte)
print("✓ Reporte guardado en: models/reporte_metricas.txt")

# ── 5. Matriz de confusión ────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=nombres_clases,
    yticklabels=nombres_clases,
    cbar_kws={"label": "Cantidad de imágenes"},
)
plt.xlabel("Predicción del modelo")
plt.ylabel("Clase real")
plt.title(f"Matriz de Confusión — Accuracy: {acc:.2%}")
plt.tight_layout()
plt.savefig("models/matriz_confusion.png", dpi=150)
print("✓ Matriz de confusión guardada en: models/matriz_confusion.png")

# ── 6. Resumen final ───────────────────────────────────────────
print("\n" + "="*60)
print("EVALUACIÓN COMPLETA")
print("="*60)
print(f"Imágenes evaluadas: {len(y_true)}")
print(f"Accuracy final en TEST: {acc:.2%}")
print("\nArchivos generados en models/:")
print("  - reporte_metricas.txt")
print("  - matriz_confusion.png")