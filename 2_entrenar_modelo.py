# ============================================================
# 2_entrenar_modelo.py
# Entrena un modelo de clasificación de enfermedades en
# arándanos usando MobileNetV2 + Transfer Learning
# Clases: sano | botrytis | antracnosis
# ============================================================

import os
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Configuración ────────────────────────────────────────────
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 8          # bajo porque tenemos pocas imágenes
EPOCHS      = 15
DATASET_DIR = "dataset"
MODEL_PATH  = "models/modelo_arandano.keras"

os.makedirs("models", exist_ok=True)

# ── 1. Data Augmentation para TRAIN ──────────────────────────
# Como tenemos pocas imágenes reales, generamos variaciones
# (rotación, espejo, zoom, brillo) para que el modelo vea
# más diversidad sin necesitar más fotos.
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
)

# Val y Test SOLO se normalizan, nunca se aumentan
val_test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
)

val_gen = val_test_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)

print("\nClases detectadas:", train_gen.class_indices)
NUM_CLASES = len(train_gen.class_indices)

# ── 2. Construir el modelo (arquitectura del diagrama) ──────

# Base: MobileNetV2 preentrenado en ImageNet, sin la capa final
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False  # congelar los ~1.4M parámetros

# Capas entrenables (Transfer Learning)
inputs = keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASES, activation="softmax")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ── 3. Entrenamiento ──────────────────────────────────────────
print("\nIniciando entrenamiento...\n")

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3
    ),
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# ── 4. Guardar el modelo entrenado ───────────────────────────
model.save(MODEL_PATH)
print(f"\n✓ Modelo guardado en: {MODEL_PATH}")

# Guardar el mapeo de clases
import json
with open("models/clases.json", "w") as f:
    json.dump(train_gen.class_indices, f)

# ── 5. Graficar curvas de entrenamiento ──────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history["accuracy"], label="Train")
axes[0].plot(history.history["val_accuracy"], label="Validación")
axes[0].set_title("Accuracy por época")
axes[0].set_xlabel("Época")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(history.history["loss"], label="Train")
axes[1].plot(history.history["val_loss"], label="Validación")
axes[1].set_title("Loss por época")
axes[1].set_xlabel("Época")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("models/curvas_entrenamiento.png", dpi=150)
print("✓ Gráfica guardada en: models/curvas_entrenamiento.png")

print("\n" + "="*50)
print("ENTRENAMIENTO COMPLETO")
print("="*50)
print(f"Accuracy final (train):      {history.history['accuracy'][-1]:.2%}")
print(f"Accuracy final (validación): {history.history['val_accuracy'][-1]:.2%}")
print("\nSiguiente paso: ejecuta 3_evaluar_modelo.py")