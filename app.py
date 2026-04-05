"""
Flask Backend — Digit Recognizer API
Trains model automatically on first run if no model file is found.
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

# ── Train model from scratch ───────────────────────────
def train_and_save():
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    print("[train] No model found — training now (~5 mins)...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0
    x_train = x_train[..., None]
    x_test  = x_test[..., None]

    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3,3), activation="relu", padding="same"),
        layers.Conv2D(32, (3,3), activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(x_train, y_train, epochs=5, batch_size=128,
              validation_data=(x_test, y_test), verbose=1)

    _, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"[train] Done! Accuracy: {acc*100:.2f}%")
    model.save(MODEL_PATH)
    print(f"[train] Model saved to {MODEL_PATH}")
    return model

# ── Load or train model ────────────────────────────────
print("\n[startup] Initializing model...")
try:
    import tensorflow as tf
    if os.path.exists(MODEL_PATH):
        print(f"[startup] Loading existing model from {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("[startup] Model loaded successfully!")
    else:
        print("[startup] No model file found — training from scratch...")
        model = train_and_save()
    print(f"[startup] Ready! Input: {model.input_shape} Output: {model.output_shape}")
except Exception as e:
    print(f"[startup] FATAL ERROR: {e}")
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
        tensor      = preprocess(data["image"])
        preds       = model.predict(tensor, verbose=0)[0].tolist()
        digit       = int(np.argmax(preds))
        confidence  = float(max(preds))
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
