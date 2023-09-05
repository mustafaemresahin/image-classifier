# Import necessary libraries
import os
import time
import tensorflow as tf  
# For deep learning tasks
import numpy as np  
# For numerical operations
from flask import Flask, render_template, request, redirect  
# For web framework
from PIL import Image  
# For image processing
from io import BytesIO  
# For byte stream handling
import base64  
# For encoding and decoding
from flask import send_from_directory  
# For serving files

# Initialize the Flask application
app = Flask(__name__)

# Load pre-trained deep learning models for image classification
# MobileNetV2 is a lightweight, high-speed model
# ResNet50 is a heavier but more accurate model
MobileNet = tf.keras.applications.MobileNetV2(weights='imagenet')
ResNet50 = tf.keras.applications.ResNet50(weights='imagenet')

# Function to preprocess image so it can be fed into MobileNetV2 model
# It takes a PIL image, resizes it, and normalizes its pixel values
def preprocess_image(image):
    try:
        # Resize the image to 224x224 pixels 
        # which is required input size for MobileNetV2
        image = tf.image.resize(image, (224, 224))
    except Exception as e:
        # Catch any image processing exceptions
        return render_template('index.html', cant=True)
    # Normalize pixel values to the range [-1, 1], 
    # also a MobileNetV2 requirement
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image

# Function to read an image file and convert it to base64 encoding
# This is useful for displaying images on a web page without saving them
def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

# Function to classify the image using the MobileNetV2 model 
# Preprocesses the image, adds a batch dimension, and makes predictions
def classify_image_using_MobileNet(image):
    # Preprocess the image
    image = preprocess_image(image)
    # Add a batch dimension since the model expects batched input
    image = tf.expand_dims(image, axis=0)
    # Use the model to make predictions
    predictions = MobileNet.predict(image)
    # Decode the prediction to human-readable labels
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions)
    # Get the label of the class with the highest confidence
    label = decoded_predictions[0][0][1]
    return label

# Similar to above but for the ResNet50 model
# Function to classify an image using the ResNet50 model
def classify_image_using_ResNet50(image_path):
    # Load the image and preprocess it
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = tf.keras.applications.resnet50.preprocess_input(image_array)
    # Make predictions
    prediction = ResNet50.predict(image_array)
    # Decode predictions to human-readable labels
    decoded_prediction = tf.keras.applications.resnet50.decode_predictions(prediction, top=1)[0]
    # Extract label of the most likely class
    label = decoded_prediction[0][1]
    return label

# Define the root route
@app.route('/', methods=['GET', 'POST'])
def home():
    # Render the index.html template and set the default model to 'mobile'
    return render_template('index.html', model="mobile")

# Route for image classification using MobileNetV2
@app.route('/classify-using-MobileNetV2', methods=['GET', 'POST'])
def classifyUsingMobileNetV2():
    # Declare global variables to hold image, label and base64 data
    global image, label, image_data_base64
    image = label = image_data_base64 = None
    # Check if POST request
    if request.method == 'POST':
        # Validate if image exists in request
        if 'image' not in request.files:
            return render_template('index.html', nopart=True, model="mobile")
        # Get the image from request files
        image = request.files['image']
        # If no file selected, return
        if image.filename == '':
            return render_template('index.html', nofile=True, model="mobile")
        # Try reading and converting the image
        try:
            # Read the image in binary
            image_data = image.read()
            # Encode image to base64
            image_data_base64 = base64.b64encode(image_data).decode('utf-8')
            # Convert to PIL format for further operations
            img = Image.open(BytesIO(image_data))
        except Exception as e:
            return render_template('index.html', cant=True, model="mobile")
        # Try classification
        try:
            # Classify the image
            label = classify_image_using_MobileNet(img)
        except Exception as e:
            return render_template('index.html', cant=True, model="mobile")
        # Return the prediction results
        return render_template('index.html', result=label, data=image_data_base64, model="mobile")

# Similar to above but for ResNet50
@app.route('/classify-using-ResNet50', methods=['GET', 'POST'])
def classifyUsingResNet50():
    global image, label, image_data_base64
    image = label = image_data_base64 = None
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', nopart=True, model="resnet")
        image = request.files['image']
        if image.filename == '':
            return render_template('index.html', nofile=True, model="resnet")
        try:
            # Save the image in a predefined path
            image_path = "static/uploaded_image.jpg"
            image.save(image_path)
        except Exception as e:
            return render_template('index.html', cant=True, model="resnet")
        try:
            # Classify using ResNet50
            label = classify_image_using_ResNet50(image_path)
            # Encode image to base64 for rendering in HTML
            image_data_base64 = get_base64_encoded_image(image_path)
        except Exception as e:
            return render_template('index.html', cant=True, model="resnet")
        return render_template('index.html', result=label, data=image_data_base64, model="resnet")

# Feedback route for incorrect classification
@app.route('/feedback/incorrect', methods=['GET', 'POST'])
def thanks_for_feedback_incorrect():
    if request.method == 'POST':
        # User marked the classification as incorrect. 
        return render_template('index.html', result=label, data=image_data_base64, valid=True, incorrect=True)

# Feedback route for correct classification
@app.route('/feedback/correct', methods=['GET', 'POST'])
def thanks_for_feedback_correct():
    if request.method == 'POST':
        # User marked the classification as correct. 
        return render_template('index.html', result=label, data=image_data_base64, valid=True, correct=True)

# Main function to run the Flask application
if __name__ == "__main__":
    # Run the Flask web server, debug mode is enabled
    app.run(debug=True)

# Function to create a new Flask app instance
# Useful if you're running this as a WSGI application
def create_app():
    return app

'''

ResNet50 = tf.keras.applications.ResNet50(weights='imagenet')
MobileNet = tf.keras.applications.MobileNetV2(weights='imagenet')

# Function to prepare medical images for analysis
def prepare_medical_image(image):
    try:
        # Your journey has involved adapting to new environments, much like this image processing step
        image = tf.image.resize(image, (224, 224))
    except Exception as e:
        # Your ability to face challenges head-on will translate to problem-solving here
        return render_template('index.html', error=True)
    # Normalize pixel values, akin to your quest for precision in medical diagnoses
    image = tf.keras.applications.resnet_v2.preprocess_input(image)
    return image

# Function to encode medical images for display
def encode_medical_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

@app.route('/')
def index():
    # Display an image reflecting your transformation
    image_path = 'path_to_your_image.jpg'
    encoded_image = encode_medical_image(image_path)
    
    return render_template('index.html', encoded_image=encoded_image)

# Route for image classification
@app.route('/classify', methods=['POST'])
def classify_image():
    try:
        if 'image' not in request.files:
            return render_template('index.html', error=True)
        
        image = request.files['image']
        if image.filename == '':
            return render_template('index.html', error=True)
        
        image_data = image.read()
        image_data_base64 = base64.b64encode(image_data).decode('utf-8')
        img = Image.open(BytesIO(image_data))
        
        model_name = request.form.get('model')
        if model_name == 'mobile':
            label = classify_image_using_MobileNet(img)
        elif model_name == 'resnet':
            image_path = "static/uploaded_image.jpg"
            image.save(image_path)
            label = classify_image_using_ResNet50(image_path)
            image_data_base64 = get_base64_encoded_image(image_path)
        else:
            return render_template('index.html', error=True)
        
        return render_template('index.html', result=label, data=image_data_base64, model=model_name)
    except Exception as e:
        return render_template('index.html', error=True)
'''