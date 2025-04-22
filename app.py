from flask import Flask, render_template, request # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np # type: ignore
from PIL import Image

import os
from sass import compile # type: ignore
from watchdog.observers import Observer # type: ignore
from watchdog.events import FileSystemEventHandler # type: ignore

app = Flask(__name__)
model = load_model('saved_model')  # Replace with the actual path to your saved_model folder

# Class names corresponding to the predicted class indices
class_names = ["Backterialblight1", "BrownSpot", "Healthy", "LeafBlast", "Tungro"]

def preprocess_image(img_path, target_size=(224, 224)):
    img = Image.open(img_path)
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def compile_scss():
    input_file = "static/styles.scss"
    output_file = "static/styles.css"
    
    with open(input_file, "r") as f:
        scss_code = f.read()

    compiled_css = compile(string=scss_code)
    with open(output_file, "w") as f:
        f.write(compiled_css)

class MyHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith(".scss"):
            compile_scss()

# Initialize the observer outside of __main__
observer = Observer()
event_handler = MyHandler()
observer.schedule(event_handler, path="static", recursive=False)
observer.start()

# Cache version to force stylesheet refresh
cache_version = 1

@app.route('/')
def index():
    return render_template('index.html', prediction=None, cache_version=cache_version)

@app.route('/predict', methods=['POST'])
def predict():
    global cache_version
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']
        if file:
            # Save the uploaded file to a temporary location
            file_path = 'temp_image.jpg'
            file.save(file_path)

            # Preprocess the image
            img_array = preprocess_image(file_path)

            # Make predictions using the model
            predictions = model.predict(img_array)

            # Get the predicted class
            predicted_class_index = np.argmax(predictions, axis=1)
            
            # Map the predicted class index to class name
            predicted_class_name = class_names[predicted_class_index[0]]

            # Clean up: Remove the temporary file
            os.remove(file_path)

            # Update cache version to force stylesheet refresh
            cache_version += 1

            # Return the prediction only
            return render_template('index.html', prediction=predicted_class_name, cache_version=cache_version)

    # Return an empty prediction if no file or prediction
    return render_template('index.html', prediction=None, cache_version=cache_version)

if __name__ == '__main__':
    # Remove the infinite loop, and run the app
    app.run(debug=True)

# Stop the observer when the app is stopped
observer.stop()
observer.join()










# from flask import Flask, render_template, request # type: ignore
# from tensorflow.keras.models import load_model, Sequential # type: ignore
# from tensorflow.keras.layers import Dense # type: ignore
# from tensorflow.keras.preprocessing import image # type: ignore
# import numpy as np # type: ignore
# from PIL import Image
# import os
# from sass import compile # type: ignore
# from watchdog.observers import Observer # type: ignore
# from watchdog.events import FileSystemEventHandler # type: ignore

# app = Flask(__name__)

# # If training is required, define and train your model below
# # Uncomment and modify this section to train a new model
# """
# def train_model():
#     # Example model architecture
#     model = Sequential([
#         Dense(64, activation='relu', input_shape=(224 * 224 * 3,)),
#         Dense(32, activation='relu'),
#         Dense(5, activation='softmax')  # Assuming 5 classes
#     ])

#     # Compile the model
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#     # Placeholder for training data (replace with actual dataset)
#     # X_train, y_train, X_val, y_val = ...

#     # Train the model
#     model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

#     # Save the model
#     model.save('model.h5')

#     return model

# # Uncomment the next line to train a new model
# # model = train_model()
# """

# # Load the pre-trained model
# model = load_model('saved_model')  # Replace with the actual path to your saved_model folder

# # Class names corresponding to the predicted class indices
# class_names = ["Backterialblight1", "BrownSpot", "Healthy", "LeafBlast", "Tungro"]

# def preprocess_image(img_path, target_size=(224, 224)):
#     """Preprocesses the image for model prediction."""
#     img = Image.open(img_path)
#     img = img.resize(target_size)
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0
#     return img_array

# def compile_scss():
#     """Compiles SCSS to CSS."""
#     input_file = "static/styles.scss"
#     output_file = "static/styles.css"

#     with open(input_file, "r") as f:
#         scss_code = f.read()

#     compiled_css = compile(string=scss_code)
#     with open(output_file, "w") as f:
#         f.write(compiled_css)

# class MyHandler(FileSystemEventHandler):
#     """Watches for SCSS changes and recompiles."""
#     def on_modified(self, event):
#         if event.src_path.endswith(".scss"):
#             compile_scss()

# # Initialize the observer
# observer = Observer()
# event_handler = MyHandler()
# observer.schedule(event_handler, path="static", recursive=False)
# observer.start()

# # Cache version to force stylesheet refresh
# cache_version = 1

# @app.route('/')
# def index():
#     return render_template('index.html', prediction=None, cache_version=cache_version)

# @app.route('/predict', methods=['POST'])
# def predict():
#     global cache_version
#     if request.method == 'POST':
#         # Get the uploaded file
#         file = request.files['file']
#         if file:
#             # Save the uploaded file to a temporary location
#             file_path = 'temp_image.jpg'
#             file.save(file_path)

#             # Preprocess the image
#             img_array = preprocess_image(file_path)

#             # Make predictions using the model
#             predictions = model.predict(img_array)

#             # Get the predicted class
#             predicted_class_index = np.argmax(predictions, axis=1)

#             # Map the predicted class index to class name
#             predicted_class_name = class_names[predicted_class_index[0]]

#             # Clean up: Remove the temporary file
#             os.remove(file_path)

#             # Update cache version to force stylesheet refresh
#             cache_version += 1

#             # Return the prediction
#             return render_template('index.html', prediction=predicted_class_name, cache_version=cache_version)

#     # Return an empty prediction if no file or prediction
#     return render_template('index.html', prediction=None, cache_version=cache_version)

# if __name__ == '__main__':
#     # Run the app
#     try:
#         app.run(debug=True)
#     finally:
#         # Stop the observer when the app is stopped
#         observer.stop()
#         observer.join()





# from flask import Flask, render_template, request
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np
# from PIL import Image
# import os
# from sass import compile
# from watchdog.observers import Observer
# from watchdog.events import FileSystemEventHandler
# import psutil

# # Example: Get CPU usage percentage
# print(psutil.cpu_percent(interval=1))


# app = Flask(__name__)

# # Load the pre-trained model in .h5 format (or .keras format)
# model = load_model('model.h5')  # Use the appropriate file extension

# # Class names corresponding to the predicted class indices
# class_names = ["Backterialblight1", "BrownSpot", "Healthy", "LeafBlast", "Tungro"]

# def preprocess_image(img_path, target_size=(224, 224)):
#     """Preprocesses the image for model prediction."""
#     img = Image.open(img_path)
#     img = img.resize(target_size)
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0
#     return img_array

# def compile_scss():
#     """Compiles SCSS to CSS."""
#     input_file = "static/styles.scss"
#     output_file = "static/styles.css"

#     with open(input_file, "r") as f:
#         scss_code = f.read()

#     compiled_css = compile(string=scss_code)
#     with open(output_file, "w") as f:
#         f.write(compiled_css)

# class MyHandler(FileSystemEventHandler):
#     """Watches for SCSS changes and recompiles."""
#     def on_modified(self, event):
#         if event.src_path.endswith(".scss"):
#             compile_scss()

# # Initialize the observer
# observer = Observer()
# event_handler = MyHandler()
# observer.schedule(event_handler, path="static", recursive=False)
# observer.start()

# # Cache version to force stylesheet refresh
# cache_version = 1

# @app.route('/')
# def index():
#     return render_template('index.html', prediction=None, cache_version=cache_version)

# @app.route('/predict', methods=['POST'])
# def predict():
#     global cache_version
#     if request.method == 'POST':
#         # Get the uploaded file
#         file = request.files['file']
#         if file:
#             # Save the uploaded file to a temporary location
#             file_path = 'temp_image.jpg'
#             file.save(file_path)

#             # Preprocess the image
#             img_array = preprocess_image(file_path)

#             # Make predictions using the model
#             predictions = model.predict(img_array)

#             # Get the predicted class
#             predicted_class_index = np.argmax(predictions, axis=1)

#             # Map the predicted class index to class name
#             predicted_class_name = class_names[predicted_class_index[0]]

#             # Clean up: Remove the temporary file
#             os.remove(file_path)

#             # Update cache version to force stylesheet refresh
#             cache_version += 1

#             # Return the prediction
#             return render_template('index.html', prediction=predicted_class_name, cache_version=cache_version)

#     # Return an empty prediction if no file or prediction
#     return render_template('index.html', prediction=None, cache_version=cache_version)

# if __name__ == '__main__':
#     # Run the app
#     try:
#         app.run(debug=True)
#     finally:
#         # Stop the observer when the app is stopped
#         observer.stop()
#         observer.join()
