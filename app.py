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



IMAGE_MODEL_PATH = "image_model.h5"
IMAGE_URL = "https://drive.google.com/uc?export=download&id=1oGJOXkSVakBCLJio1ZfF9w8pYQLtjaoP"

MEDICAL_MODEL_PATH = "medical_model.pkl"
MEDICAL_URL = "https://drive.google.com/uc?export=download&id=1ENkUXXKllCi02M2ri72qfhC1Shuaz5ZB"

SCALER_PATH = "medical_scaler.pkl"
SCALER_URL = "https://drive.google.com/uc?export=download&id=1frkGHa74b1Qm-gx_3HmzCaJWIxhWSLuz"


image_model = None
medical_model = None
medical_scaler = None


def load_models():
    global image_model, medical_model, medical_scaler

    if image_model is None:
        print("Loading models...")

        if not os.path.exists(IMAGE_MODEL_PATH):
            gdown.download(IMAGE_URL, IMAGE_MODEL_PATH, quiet=False)
        image_model = load_model(IMAGE_MODEL_PATH, compile=False)

        if not os.path.exists(MEDICAL_MODEL_PATH):
            gdown.download(MEDICAL_URL, MEDICAL_MODEL_PATH, quiet=False)
        medical_model = joblib.load(MEDICAL_MODEL_PATH)

        if not os.path.exists(SCALER_PATH):
            gdown.download(SCALER_URL, SCALER_PATH, quiet=False)
        medical_scaler = joblib.load(SCALER_PATH)


@app.route("/health")
def health():
    return "OK", 200


@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict', methods=['POST'])
def predict():
    load_models()  # <-- THIS IS THE KEY LINE

    image_file = request.files['image']

    filename = secure_filename(image_file.filename)

    # 1️⃣ Filesystem path (for OpenCV)
    save_path = os.path.join("static", "uploads", filename)
    image_file.save(save_path)

    # 2️⃣ URL path (for browser)
    image_url = f"/static/uploads/{filename}"

    # Read image correctly
    img = cv2.imread(save_path)



    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    preds_img = image_model.predict(img)

    temp = float(request.form['Temperature'])

    swell = 1.0 if request.form['Swelling'].lower() == "yes" else 0.0
    nasal = 1.0 if request.form['Nasal_Discharge'].lower() == "yes" else 0.0
    skin  = 1.0 if request.form['Skin_Nodules'].lower() == "yes" else 0.0
    eye   = 1.0 if request.form['Eye_Discharge'].lower() == "yes" else 0.0

    appetite = request.form['Appetite_Level'].lower()

    app_high = 1.0 if appetite == "high" else 0.0
    app_low = 1.0 if appetite == "low" else 0.0
    app_norm = 1.0 if appetite == "normal" else 0.0

    med_input = np.array([[
        temp,
        swell,
        nasal,
        skin,
        eye,
        app_high,
        app_low,
        app_norm
    ]])


    
    med_scaled = medical_scaler.transform(med_input)
    preds_med = medical_model.predict(med_scaled)

    fused = 0.3 * preds_img + 0.7 * preds_med
    prob = float(np.max(fused))
    label = int(np.argmax(fused))

    result = "LSD Positive" if label == 1 else "LSD Negative"

    return render_template(
        "index.html",
        prediction=f"{result} ({prob:.2f})",
        image_path=image_url
    )




if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
