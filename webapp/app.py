"""
Flask web app — CVAE Plant Stress Detection Demo
Run: python app.py
"""
import os
import io
import json
import base64

import numpy as np
from PIL import Image

import torch
from torchvision import transforms

from flask import Flask, request, jsonify, render_template, send_from_directory

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.model import load_model, predict_image, generate_image, IMG_SIZE

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR   = os.path.join(BASE_DIR, '..', 'models')
RESULTS_DIR  = os.path.join(BASE_DIR, '..', 'results')
UPLOAD_DIR   = os.path.join(BASE_DIR, 'static', 'uploads')
WEIGHTS_PATH = os.path.join(MODELS_DIR, 'cvae_weights.pth')
METRICS_PATH = os.path.join(RESULTS_DIR, 'metrics.json')

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8 MB

# ── Load model once at startup ─────────────────────────────────────────────────
print("Loading CVAE model...")
if os.path.exists(WEIGHTS_PATH):
    model, device = load_model(WEIGHTS_PATH)
    print(f"Model loaded on {device}")
else:
    model, device = None, None
    print("WARNING: weights not found — run the notebook first.")

# Load metrics
metrics = {}
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH) as f:
        metrics = json.load(f)

THRESHOLD  = metrics.get('threshold', 0.01)
CLASSES    = metrics.get('classes', ['Healthy', 'Stressed'])

# ── Image preprocessing ────────────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

def tensor_to_b64(tensor):
    """Convert (1,3,H,W) float tensor → base64 PNG string."""
    arr  = tensor[0].permute(1, 2, 0).clamp(0, 1).numpy()
    arr  = (arr * 255).astype(np.uint8)
    img  = Image.fromarray(arr)
    buf  = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', metrics=metrics, classes=CLASSES)


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Run the notebook first.'}), 503

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename.'}), 400

    try:
        img  = Image.open(file.stream).convert('RGB')
        orig_b64  = tensor_to_b64(preprocess(img).unsqueeze(0))
        img_tensor = preprocess(img).unsqueeze(0)

        # Try both class conditionings, report the one with lower error
        results = {}
        for idx, name in enumerate(CLASSES):
            results[name] = predict_image(model, img_tensor, idx, THRESHOLD, device)

        # Pick conditioning that matches final prediction
        best = min(results.values(), key=lambda r: r['error'])

        return jsonify({
            'original_b64'     : orig_b64,
            'recon_b64'        : tensor_to_b64(best['recon_img']),
            'prediction'       : best['prediction'],
            'confidence'       : best['confidence'],
            'error'            : best['error'],
            'threshold'        : best['threshold'],
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/generate', methods=['POST'])
def generate():
    if model is None:
        return jsonify({'error': 'Model not loaded.'}), 503

    data      = request.get_json(silent=True) or {}
    cls_name  = data.get('class', CLASSES[0])
    cls_idx   = CLASSES.index(cls_name) if cls_name in CLASSES else 0

    try:
        gen_tensor = generate_image(model, cls_idx, device)
        return jsonify({
            'image_b64': tensor_to_b64(gen_tensor),
            'class'    : cls_name,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/metrics')
def get_metrics():
    return jsonify(metrics)


@app.route('/results/<path:filename>')
def serve_result(filename):
    return send_from_directory(RESULTS_DIR, filename)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
