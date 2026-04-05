"""
Flask Backend — Digit Recognizer API
Downloads pre-trained model from Hugging Face on first run.
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

# ── PASTE YOUR HUGGING FACE URL HERE ──────────────────
HF_URL = "https://huggingface.co/iparithimarran/digit-model/resolve/main/digit_model.keras"

# ── Download model from Hugging Face ──────────────────
def download_model():
    import urllib.request
    print(f"[startup] Downloading model from Hugging Face...")
    urllib.request.urlretrieve(HF_URL, MODEL_PATH)
    print(f"[startup] Download complete!")

# ── Load model ─────────────────────────────────────────
print("\n[startup] Initializing...")
try:
    import tensorflow as tf
    if not os.path.exists(MODEL_PATH):
        download_model()
    print(f"[startup] Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"[startup] Ready!")
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
