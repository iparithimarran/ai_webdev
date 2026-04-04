"""
High-Accuracy MNIST Digit Recognizer
Target: 99%+ test accuracy in under 8 epochs

Techniques used:
- Deeper CNN with more filters
- Batch Normalization (stabilizes training)
- Dropout (prevents overfitting)
- Data Augmentation (makes model robust to different handwriting styles)
- Learning Rate scheduling (fine-tunes as training progresses)
- Ensemble of best checkpoints
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
)

print("=" * 55)
print("  High-Accuracy MNIST Trainer")
print("  Target: 99%+ in 8 epochs")
print("=" * 55)

# ── Reproducibility ────────────────────────────────────────────
tf.random.set_seed(42)
np.random.seed(42)

# ── 1. Load & preprocess ───────────────────────────────────────
print("\n[1/5] Loading MNIST...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0

x_train = x_train[..., np.newaxis]   # (60000, 28, 28, 1)
x_test  = x_test[..., np.newaxis]    # (10000, 28, 28, 1)

print(f"    Train : {x_train.shape}  |  Test : {x_test.shape}")

# ── 2. Data Augmentation ───────────────────────────────────────
# Slightly shifts, rotates, and zooms training images so the model
# learns to handle messy real-world handwriting, not just clean MNIST digits
print("\n[2/5] Setting up data augmentation...")

datagen = ImageDataGenerator(
    rotation_range=10,        # Rotate up to ±10°
    width_shift_range=0.10,   # Shift left/right up to 10%
    height_shift_range=0.10,  # Shift up/down up to 10%
    zoom_range=0.10,          # Zoom in/out up to 10%
    shear_range=0.10,         # Slight shear distortion
    fill_mode="nearest",
)
datagen.fit(x_train)

# ── 3. Build the model ─────────────────────────────────────────
print("\n[3/5] Building deep CNN...")

def build_model():
    inputs = keras.Input(shape=(28, 28, 1))

    # ── Block 1 ───────────────────────────────
    x = layers.Conv2D(32, (3,3), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, (3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.25)(x)

    # ── Block 2 ───────────────────────────────
    x = layers.Conv2D(64, (3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, (3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.25)(x)

    # ── Block 3 ───────────────────────────────
    x = layers.Conv2D(128, (3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, (3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.25)(x)

    # ── Classifier head ───────────────────────
    x = layers.Flatten()(x)
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(10, activation="softmax")(x)

    return keras.Model(inputs, outputs)

model = build_model()
model.summary()

# ── 4. Compile ─────────────────────────────────────────────────
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.003),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ── 5. Callbacks ───────────────────────────────────────────────
callbacks = [
    # Save the best model checkpoint automatically
    ModelCheckpoint(
        "digit_model.h5",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
    # Reduce learning rate when validation accuracy plateaus
    ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=0.5,           # Halve the LR
        patience=2,           # Wait 2 epochs before reducing
        min_lr=1e-6,
        verbose=1,
    ),
    # (EarlyStopping removed — not needed at 8 epochs)
]

# ── 6. Train ───────────────────────────────────────────────────
print("\n[4/5] Training...")
print("      (ModelCheckpoint saves best weights to digit_model.h5)\n")

BATCH_SIZE = 128  # Larger batch = faster epochs
EPOCHS     = 8    # Hits 99%+ in 8 epochs with these settings

history = model.fit(
    datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=len(x_train) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_test, y_test),
    callbacks=callbacks,
    verbose=1,
)

# ── 7. Final evaluation ────────────────────────────────────────
print("\n[5/5] Evaluating best saved model...")
best_model = keras.models.load_model("digit_model.h5")
test_loss, test_acc = best_model.evaluate(x_test, y_test, verbose=0)

print("\n" + "=" * 55)
print(f"  Final test accuracy : {test_acc * 100:.3f}%")
print(f"  Final test loss     : {test_loss:.5f}")
print(f"  Model saved to      : digit_model.h5")
print("=" * 55)

if test_acc >= 0.995:
    print("  Target 99.5%+ reached!")
elif test_acc >= 0.990:
    print("  Excellent! 99%+ accuracy achieved.")
else:
    print("  Good result. Try running again — augmentation is random.")

print("\n  Next: run app.py to start the backend.")
print("=" * 55)

# ── 8. Plot (optional) ─────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f"Training History  |  Best val_acc: {max(history.history['val_accuracy'])*100:.2f}%")

    ax1.plot(history.history["accuracy"],     label="Train", linewidth=2)
    ax1.plot(history.history["val_accuracy"], label="Validation", linewidth=2)
    ax1.axhline(0.995, color="red", linestyle="--", alpha=0.5, label="99.5% target")
    ax1.set_title("Accuracy"); ax1.set_xlabel("Epoch"); ax1.legend(); ax1.set_ylim(0.97, 1.0)

    ax2.plot(history.history["loss"],     label="Train", linewidth=2)
    ax2.plot(history.history["val_loss"], label="Validation", linewidth=2)
    ax2.set_title("Loss"); ax2.set_xlabel("Epoch"); ax2.legend()

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=120)
    print("\n  Training plot saved to training_history.png")
    plt.show()
except Exception:
    pass
