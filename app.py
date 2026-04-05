"""
Flask Backend — Digit Recognizer API
Uses a lightweight model that trains within Render's 512MB free tier.
"""

import os
import numpy as np
import base64
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "digit_model.keras")

# ── Train a lightweight model that fits in 512MB ───────
def train_and_save():
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    # Limit TF memory usage
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    print("[train] Loading MNIST...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Use only 30k samples to save memory
    x_train = x_train[:30000].astype("float32") / 255.0
    y_train = y_train[:30000]
    x_test  = x_test.astype("float32") / 255.0

    x_train = x_train[..., None]
    x_test  = x_test[..., None]

    print("[train] Building model...")
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(16, (3,3), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, (3,3), activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("[train] Training (3 epochs)...")
    model.fit(
        x_train, y_train,
        epochs=3,
        batch_size=256,
        validation_data=(x_test, y_test),
        verbose=1,
    )

    _, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"[train] Done! Accuracy: {acc*100:.2f}%")
    model.save(MODEL_PATH)
    print(f"[train] Saved to {MODEL_PATH}")
    return model

# ── Load or train ──────────────────────────────────────
print("\n[startup] Initializing...")
try:
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    if os.path.exists(MODEL_PATH):
        print(f"[startup] Loading model from {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("[startup] Model loaded!")
    else:
        print("[startup] No model found — training now...")
        model = train_and_save()

    print(f"[startup] Ready! {model.input_shape} → {model.output_shape}")
except Exception as e:
    print(f"[startup] ERROR: {e}")
    model = None

# ── Preprocess canvas image ────────────────────────────
def preprocess(data_url):
    _, encoded = data_url.split(",", 1)
    img = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("L")
    img = img.resize((28, 28), Image.LANCZOS)
    arr = np.array(img, dtype="float32") / 255.0
    return arr.reshape(1, 28, 28, 1)

# ── Routes ─────────────────────────────────────────────
@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "message": "Digit recognizer API is running. POST /predict to use it."
    })

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded."}), 503

    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "Missing 'image' field."}), 400

    try:
        tensor     = preprocess(data["image"])
        preds      = model.predict(tensor, verbose=0)[0].tolist()
        digit      = int(np.argmax(preds))
        confidence = float(max(preds))
        print(f"[predict] digit={digit}  conf={confidence*100:.1f}%")
        return jsonify({
            "digit":       digit,
            "confidence":  round(confidence, 4),
            "confidences": [round(c, 4) for c in preds],
        })
    except Exception as e:
        print(f"[predict] ERROR: {e}")
        return jsonify({"error": str(e)}), 500

# ── Run ────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  Listening on port {port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
