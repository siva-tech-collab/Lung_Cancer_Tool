from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from predict import predict

# =====================================
# APP CONFIG
# =====================================
app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static"
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# =====================================
# HOME
# =====================================
@app.route('/')
def index():
    return render_template('index.html')

# =====================================
# PREDICT API
# =====================================
@app.route('/predict', methods=['POST'])
def predict_api():

    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400

        # SAFE SAVE
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # MODEL PREDICTION
        result, confidence, heatmap, original = predict(filepath)

        # HEATMAP SAVE
        heatmap_img = cv2.applyColorMap(
            np.uint8(255 * heatmap),
            cv2.COLORMAP_JET
        )

        heatmap_filename = "heatmap.png"
        heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], heatmap_filename)
        cv2.imwrite(heatmap_path, heatmap_img)

        return jsonify({
            "result": result,
            "confidence": float(confidence),
            "heatmap": f"/uploads/{heatmap_filename}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =====================================
# UPLOAD FILE ACCESS ROUTE
# =====================================
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# =====================================
# RUN
# =====================================
if __name__ == '__main__':
    app.run(debug=True)