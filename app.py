import io
from flask import Flask
from flask_restful import Api, Resource, reqparse,request
import tkinter as tk
from tkinter import filedialog
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from PIL import Image, ImageTk
import base64
import numpy as np
app = Flask(__name__)
api = Api(app)
class_mapping = {
    0: "Abscess",
    1: "Ards",
    2: "Atelectasis",
    3: "Atherosclerosis of the aorta",
    4: "Cardiomegaly",
    5: "Emphysema",
    6: "Fracture",
    7: "Hydropneumothorax",
    8: "Hydrothorax",
    9: "Pneumonia",
    10: "Pneumosclerosis",
    11: "Post inflammatory changes",
    12: "Post traumatic ribs deformation",
    13: "Sarcoidosis",
    14: "Scoliosis",
    15: "Tuberculosis",
    16: "Venous congestion"
}
model = load_model('model_VGG16.h5')
def predict_image(imgByBase64):
    # Open a file dialog to select an image

    # Load and preprocess the image
    img = Image.open(io.BytesIO(base64.decodebytes(bytes(imgByBase64, "utf-8"))))
    img = img.resize((64,64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make a prediction
    predictions = model.predict(img_array)

    # Get the predicted class number
    predicted_class_number = np.argmax(predictions)

    # Get the corresponding class name
    predicted_class_name = class_mapping.get(predicted_class_number, "Unknown")
    return predicted_class_name


class ModelController(Resource):
    @app.route("/api/Model/PostImage/",methods=["POST"])
    def handleImage():
        return  predict_image(request.json["content"]), 200

if __name__ == '__main__':
    app.run(debug=True)
    