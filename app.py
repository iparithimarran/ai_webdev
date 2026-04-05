"""
Phase 2 — Flask Backend API
Run: python app.py
Listens on: http://localhost:5000
Endpoint: POST /predict
"""

import numpy as np
import base64
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageOps
import tensorflow as tf

# ── App setup ──────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Allow requests from your frontend (index.html)

# ── Load model once at startup ─────────────────────────
print("[startup] Loading model...")
try:
    import os
    import subprocess
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "digit_model.keras")

    # If no model exists, train one on the fly
    if not os.path.exists(model_path):
        print("[startup] No model found — training now (takes ~5 mins)...")
        subprocess.run(["python", os.path.join(BASE_DIR, "ai_model.py")], check=True)
        print("[startup] Training complete!")

    model = tf.keras.models.load_model(model_path)
    print("[startup] Model loaded successfully!")
except Exception as e:
    print(f"[startup] ERROR: Could not load model — {e}")
    model = None


# ── Helper: preprocess canvas image → 28x28 tensor ────
def preprocess_image(image_data_url: str) -> np.ndarray:
    """
    Takes a base64 PNG from the canvas (white digit on black bg),
    resizes to 28x28, normalizes, and returns shape (1, 28, 28, 1).

    MNIST images are: white digit on BLACK background.
    Canvas images are: white digit on BLACK background. ✓ (already matches)
    """

    # Strip the data URL prefix  →  raw base64 string
    header, encoded = image_data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)

    # Open with Pillow
    img = Image.open(io.BytesIO(img_bytes)).convert("L")  # Grayscale

    # Resize to 28x28 (MNIST format) using high-quality resampling
    img = img.resize((28, 28), Image.LANCZOS)

    # Convert to numpy array and normalize 0–255 → 0.0–1.0
    arr = np.array(img, dtype="float32") / 255.0

    # Add batch and channel dims: (28,28) → (1, 28, 28, 1)
    arr = arr.reshape(1, 28, 28, 1)

    return arr


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
    """
    Expects JSON body: { "image": "data:image/png;base64,..." }
    Returns JSON:      { "digit": 7, "confidence": 0.98, "confidences": [...10 floats...] }
    """

    # ── Guard: model must be loaded ────────────────────
    if model is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503

    # ── Guard: request must have JSON + image field ────
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "Missing 'image' field in JSON body."}), 400

    try:
        # ── Preprocess ─────────────────────────────────
        tensor = preprocess_image(data["image"])

        # ── Run inference ──────────────────────────────
        predictions = model.predict(tensor, verbose=0)  # shape: (1, 10)
        confidences = predictions[0].tolist()           # list of 10 floats

        predicted_digit = int(np.argmax(confidences))
        confidence      = float(max(confidences))

        # ── Log to terminal ────────────────────────────
        bar = "█" * int(confidence * 20)
        print(f"[predict] digit={predicted_digit}  conf={confidence*100:.1f}%  {bar}")

        return jsonify({
            "digit":       predicted_digit,
            "confidence":  round(confidence, 4),
            "confidences": [round(c, 4) for c in confidences],
        })

    except Exception as e:
        print(f"[predict] ERROR: {e}")
        return jsonify({"error": str(e)}), 500


# ── Run ────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Digit Recognizer — Backend Server")
    print("=" * 50)
    print("  Listening on : http://localhost:5000")
    print("  Health check : GET  /")
    print("  Predict      : POST /predict")
    print("=" * 50 + "\n")

    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
