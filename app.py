from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import joblib
import cv2
import os
import gdown
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'


# -----------------------------
# Load all required models
# -----------------------------

# ===== IMAGE MODEL (CNN) =====
IMAGE_MODEL_PATH = "image_model.h5"
IMAGE_FILE_ID = "1oGJOXkSVakBCLJio1ZfF9w8pYQLtjaoP"
IMAGE_URL = f"https://drive.google.com/uc?export=download&id={IMAGE_FILE_ID}"

if not os.path.exists(IMAGE_MODEL_PATH):
    print("Downloading image model from Google Drive...")
    gdown.download(IMAGE_URL, IMAGE_MODEL_PATH, quiet=False)

image_model = load_model(IMAGE_MODEL_PATH, compile=False)


# ===== MEDICAL MODEL =====
MEDICAL_MODEL_PATH = "medical_model.pkl"
MEDICAL_FILE_ID = "1ENkUXXKllCi02M2ri72qfhC1Shuaz5ZB"
MEDICAL_URL = f"https://drive.google.com/uc?export=download&id={MEDICAL_FILE_ID}"

if not os.path.exists(MEDICAL_MODEL_PATH):
    print("Downloading medical model from Google Drive...")
    gdown.download(MEDICAL_URL, MEDICAL_MODEL_PATH, quiet=False)

medical_model = joblib.load(MEDICAL_MODEL_PATH)


# ===== MEDICAL SCALER =====
SCALER_PATH = "medical_scaler.pkl"
SCALER_FILE_ID = "1frkGHa74b1Qm-gx_3HmzCaJWIxhWSLuz"
SCALER_URL = f"https://drive.google.com/uc?export=download&id={SCALER_FILE_ID}"

if not os.path.exists(SCALER_PATH):
    print("Downloading medical scaler from Google Drive...")
    gdown.download(SCALER_URL, SCALER_PATH, quiet=False)

medical_scaler = joblib.load(SCALER_PATH)




# (Optional) Fusion model if you really use it
# fusion_model = joblib.load("model.pkl")



# fusion_model = joblib.load("model.pkl")                    # Optional (for late fusion, not used here)
# medical_scaler = joblib.load("medical_scaler.pkl")         # Scaler used during training

# -----------------------------
# Image preprocessing function
# -----------------------------
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))  # adjust as per model input size
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# -----------------------------
# Flask routes
# -----------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # --- 1️⃣ Handle image input ---
    image_file = request.files['image']
    filename = secure_filename(image_file.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image_file.save(image_path)

    img_input = preprocess_image(image_path)
    preds_img = image_model.predict(img_input)  # e.g. [[0.02, 0.98]]

    # --- 2️⃣ Handle medical inputs ---
    temp = float(request.form['Temperature'])
    swell = 1.0 if request.form['Swelling'].lower() == "yes" else 0.0
    nasal = 1.0 if request.form['Nasal_Discharge'].lower() == "yes" else 0.0
    skin = 1.0 if request.form['Skin_Nodules'].lower() == "yes" else 0.0
    eye = 1.0 if request.form['Eye_Discharge'].lower() == "yes" else 0.0
    app_high = 1.0 if request.form['Appetite_Level'].lower() == "high" else 0.0
    app_low = 1.0 if request.form['Appetite_Level'].lower() == "low" else 0.0
    app_norm = 1.0 if request.form['Appetite_Level'].lower() == "normal" else 0.0

    med_input = np.array([[temp, swell, nasal, skin, eye, app_high, app_low, app_norm]])

    # Scale the medical data before prediction
    med_input_scaled = medical_scaler.transform(med_input)
    preds_med = medical_model.predict(med_input_scaled)  # ensure you use predict_proba()

    # --- 3️⃣ Late Fusion (Weighted average of probabilities) ---
    fused_preds = (0.7 * preds_img + 0.3 * preds_med)  # Weighted fusion

    # Get max probability and label
    prob = float(np.max(fused_preds))
    label = int(np.argmax(fused_preds))  # 0 = Negative, 1 = Positive

    print("Image Model:", preds_img)
    print("Medical Model:", preds_med)
    print("Fused Prediction:", fused_preds)
    print("Label:", label)

    # --- 4️⃣ Result Interpretation ---
    if label == 1 and prob >= 0.65:
        result = f"LSD Positive (Confidence: {prob:.2f})"
    else:
        result = f"LSD Negative (Confidence: {prob:.2f})"

    return render_template('index.html', prediction=result, image_path=image_path)

# -----------------------------
# Run the app
# -----------------------------
# if __name__ == '__main__':
#     app.run(debug=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)



