import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from PIL import Image
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "final_eye_model.keras"
UPLOAD_DIR = BASE_DIR / "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}
IMG_SIZE = (300, 300)
CLASS_NAMES = ["Cataract", "Diabetic Retinopathy", "Glaucoma", "Normal"]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload limit
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)

UPLOAD_DIR.mkdir(exist_ok=True)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

# Load once at startup for fast predictions.
model = tf.keras.models.load_model(MODEL_PATH)


def allowed_file(filename: str) -> bool:
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def preprocess_image(image_path: Path) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMG_SIZE)
    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = tf.keras.applications.efficientnet.preprocess_input(image_array)
    return image_array


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file found in request."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Please select an image file."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Use png/jpg/jpeg/bmp/webp."}), 400

    safe_name = secure_filename(file.filename)
    saved_path = UPLOAD_DIR / safe_name
    file.save(saved_path)

    try:
        img_array = preprocess_image(saved_path)
        pred = model.predict(img_array, verbose=0)[0]
        top_idx = int(np.argmax(pred))
        confidence = float(pred[top_idx])

        ranked = np.argsort(pred)[::-1]
        probabilities = [
            {
                "class_name": CLASS_NAMES[int(i)],
                "probability": round(float(pred[int(i)]) * 100, 2),
            }
            for i in ranked
        ]

        return jsonify(
            {
                "predicted_class": CLASS_NAMES[top_idx],
                "confidence": round(confidence * 100, 2),
                "probabilities": probabilities,
                "image_name": safe_name,
            }
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Prediction failed: {str(exc)}"}), 500
    finally:
        if saved_path.exists():
            saved_path.unlink(missing_ok=True)


@app.errorhandler(413)
def request_entity_too_large(_error):
    return jsonify({"error": "File is too large. Max size is 8 MB."}), 413


if __name__ == "__main__":
    app.run(debug=True)
