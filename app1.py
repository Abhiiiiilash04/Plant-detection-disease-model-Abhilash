import pickle
from io import BytesIO

import numpy as np
import tensorflow as tf
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app, supports_credentials=True, origins="*")

with open('model3.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
    
MODEL = tf.keras.models.load_model("./saved_models/2")


CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.route('/')
def home():
    return jsonify({'message': 'Hello World'})

import random
@app.route('/predictleaf', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        image = read_file_as_image(file.read())
        img_batch = np.expand_dims(image, 0)
        
        predictions = MODEL.predict(img_batch)

        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))
        return jsonify({
            'class': predicted_class,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)})
    
def predict(model, img):
    img_array = img.astype(np.float32) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  

    predictions = model.predict(img_array)

    predicted_class_index = np.argmax(predictions)
    confidence = round(100 * np.max(predictions), 2)
    return predicted_class_index, confidence

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
