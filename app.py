# ===================== Standard imports =====================
from pathlib import Path
from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

# ===================== Set base directory =====================
BASE_DIR = Path(__file__).resolve().parent

# ===================== Load CSV files =====================
disease_info = pd.read_csv(BASE_DIR / "disease_info.csv", encoding="cp1252")
supplement_info = pd.read_csv(BASE_DIR / "supplement_info.csv", encoding="cp1252")

# ===================== CNN Model =====================
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

def build_cnn_model(input_shape=(128, 128, 3), num_classes=5):
    model = Sequential([
        layers.Input(shape=input_shape),
        layers.Rescaling(1./255),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

cnn_model = build_cnn_model()
cnn_model.load_weights(BASE_DIR / "model" / "cnn_rice_leaf_weights.h5")
print("✅ CNN model weights loaded successfully!")

# ===================== MobileNetV2 Model =====================
mobilenet_model = load_model(BASE_DIR / "model" / "mobilenet_rice_leaf_full.h5")
print("✅ Full MobileNetV2 model loaded successfully!")

# ===================== Model dictionary for dynamic switching =====================
models = {
    "cnn": cnn_model,
    "mobilenet": mobilenet_model
}
default_model_name = "mobilenet"

# ===================== Prediction function =====================
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def prediction(image_path, model_name=default_model_name):
    image = Image.open(image_path).convert("RGB")
    if model_name == "mobilenet":
        image = image.resize((224, 224))
    else:  # CNN expects 128x128
        image = image.resize((128, 128))

    input_data = np.array(image)
    if model_name == "mobilenet":
        input_data = preprocess_input(input_data)
    else:
        input_data = input_data / 255.0  # CNN normalization

    input_data = np.expand_dims(input_data, axis=0)
    selected_model = models.get(model_name, models[default_model_name])
    preds = selected_model.predict(input_data)
    index = np.argmax(preds)
    return index

# ===================== Initialize Flask app =====================
app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename

        upload_folder = BASE_DIR / "static" / "uploads"
        upload_folder.mkdir(parents=True, exist_ok=True)
        file_path = upload_folder / filename
        image.save(file_path)

        model_choice = request.form.get("model_choice", default_model_name)
        pred = prediction(file_path, model_name=model_choice)

        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]

        return render_template(
            'submit.html',
            title=title,
            desc=description,
            prevent=prevent,
            image_url=image_url,
            pred=pred,
            sname=supplement_name,
            simage=supplement_image_url,
            buy_link=supplement_buy_link,
            selected_model=model_choice
        )

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template(
        'market.html',
        supplement_image=list(supplement_info['supplement image']),
        supplement_name=list(supplement_info['supplement name']),
        disease=list(disease_info['disease_name']),
        buy=list(supplement_info['buy link'])
    )

# ===================== Handle 404 error =====================
@app.errorhandler(404)
def handle_404(e):
    if request.path.startswith('/hybridaction/'):
        return '', 204
    return render_template('404.html'), 404

# ===================== Run app =====================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
