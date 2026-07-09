import os
import sys
from flask import Flask, request, jsonify

# Ensure the workspace directory is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predict import load_model, predict as run_prediction

app = Flask(__name__)

# Load model and config on startup
print("=" * 60)
print("  Initializing ADIP Model Service (Port 5001)...")
print("  Loading trained Multi-Task DistilBERT model and tokenizer...")
print("=" * 60)

try:
    model, tokenizer, cfg = load_model()
    print("\n[OK] Model & tokenizer loaded successfully and ready for inference!\n")
except Exception as e:
    print(f"\n[ERROR] Failed to load model files: {e}")
    print("Please make sure you have run 'python train_classifier.py' to train the model first.\n")
    model, tokenizer, cfg = None, None, None


@app.route("/predict", methods=["POST"])
def api_predict():
    """Predict category and rating for the provided review text."""
    if model is None:
        return jsonify({"error": "Model not loaded. Please train the model first."}), 503

    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No 'text' field provided in request body."}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Review text cannot be empty."}), 400

    try:
        result = run_prediction(text, model, tokenizer, cfg)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    # Host on localhost, port 5001, run with reloader disabled to avoid double model loading
    app.run(host="127.0.0.1", port=5001, debug=False, use_reloader=False)
