from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load the saved model
model = tf.keras.models.load_model("mnist_model.h5")

def preprocess_image(image_data):
    # Convert image data to PIL Image
    img = Image.open(io.BytesIO(image_data))
    
    # Convert to grayscale
    img = img.convert('L')
    
    # Resize to 20x20 (standard MNIST digits are about 20x20 centered in 28x28 images)
    img = img.resize((20, 20), Image.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Invert if necessary (MNIST expects white digits on black background)
    if np.mean(img_array) > 128:
        img_array = 255 - img_array
        
    # Create a 28x28 empty image with a black background
    output = np.zeros((28, 28), dtype=np.uint8)
    
    # Center the 20x20 digit in the 28x28 image
    # This adds 4 pixels of padding around the digit, exactly like MNIST
    start_x = (28 - 20) // 2
    start_y = (28 - 20) // 2
    output[start_y:start_y+20, start_x:start_x+20] = img_array
    
    # Apply Gaussian blur for smoothing
    output = cv2.GaussianBlur(output, (3, 3), 0.5)
    
    # Apply binary threshold to make the digit black and white
    _, output = cv2.threshold(output, 40, 255, cv2.THRESH_BINARY)
    
    # Normalize to 0-1 range exactly as MNIST data is normalized
    output = output.astype(np.float32) / 255.0
    
    # Reshape to model input format (batch_size, height, width, channels)
    output = output.reshape(1, 28, 28, 1)
    
    return output

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image data from request
        image_data = request.files['image'].read()
        
        # Preprocess the image
        processed_image = preprocess_image(image_data)
        
        # Make prediction
        prediction = model.predict(processed_image)
        predicted_digit = int(np.argmax(prediction[0]))
        confidence = float(prediction[0][predicted_digit])
        
        return jsonify({
            'success': True,
            'digit': predicted_digit,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

# Create templates directory if it doesn't exist
if not os.path.exists('templates'):
    os.makedirs('templates')

if __name__ == '__main__':
    app.run(debug=True) 