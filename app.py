import os
import time
import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request, redirect
from PIL import Image
from io import BytesIO
import base64
from flask import send_from_directory


app = Flask(__name__)

# Load the pre-trained MobileNetV2 and ResNet-50 models
MobileNet = tf.keras.applications.MobileNetV2(weights='imagenet')
ResNet50 = tf.keras.applications.ResNet50(weights='imagenet')


# a function to preprocess the image for the model
def preprocess_image(image):
    try:
        image = tf.image.resize(image, (224, 224))
    except Exception as e:
        return render_template('index.html', cant=True)
    
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image

#read the image file, encode it to base64
def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

# a function to classify the image
def classify_image_using_MobileNet(image):
    image = preprocess_image(image)
    image = tf.expand_dims(image, axis=0)
    predictions = MobileNet.predict(image)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions)
    label = decoded_predictions[0][0][1]
    return label

def classify_image_using_ResNet50(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = tf.keras.applications.resnet50.preprocess_input(image_array)

    prediction = ResNet50.predict(image_array)
    decoded_prediction = tf.keras.applications.resnet50.decode_predictions(prediction, top=1)[0]
    label = decoded_prediction[0][1]

    return label

@app.route('/', methods=['GET', 'POST'])
def home():
        return render_template('index.html', model="mobile")

@app.route('/classify-using-MobileNetV2', methods=['GET', 'POST'])
def classifyUsingMobileNetV2():
    global image
    global label
    global image_data_base64
    image = None
    label = None
    image_data_base64 = None
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', nopart=True, model="mobile")
        
        image = request.files['image']
        if image.filename == '':
            return render_template('index.html', nofile=True, model="mobile")
        
        try:
            image_data = image.read()
            image_data_base64 = base64.b64encode(image_data).decode('utf-8')
            img = Image.open(BytesIO(image_data))
        except Exception as e:
            return render_template('index.html', cant=True, model="mobile")
        
        try:
            label = classify_image_using_MobileNet(img)
        except Exception as e:
            return render_template('index.html', cant=True, model="mobile")

        return render_template('index.html', result=label, data=image_data_base64, model="mobile")
    

@app.route('/classify-using-ResNet50', methods=['GET', 'POST'])
def classifyUsingResNet50():
    global image
    global label
    global image_data_base64
    image = None
    label = None
    image_data_base64 = None
    
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', nopart=True, model="resnet")
        
        image = request.files['image']
        if image.filename == '':
            return render_template('index.html', nofile=True, model="resnet")

        try:
            image_path = "static/uploaded_image.jpg"
            image.save(image_path)
        except Exception as e:
            return render_template('index.html', cant=True, model="resnet")
        
        try:
            label = classify_image_using_ResNet50(image_path)
            image_data_base64 = get_base64_encoded_image(image_path)
        except Exception as e:
            return render_template('index.html', cant=True, model="resnet")

        return render_template('index.html', result=label, data=image_data_base64, model="resnet")

    

@app.route('/feedback/incorrect', methods=['GET', 'POST'])
def thanks_for_feedback_incorrect():
    if request.method == 'POST':
        return render_template('index.html', result=label, data=image_data_base64, valid=True, incorrect=True)

@app.route('/feedback/correct', methods=['GET', 'POST'])
def thanks_for_feedback_correct():
    if request.method == 'POST':
        return render_template('index.html', result=label, data=image_data_base64, valid=True, correct=True)

if __name__ == "__main__":
    app.run(debug=True)

def create_app():
    return app