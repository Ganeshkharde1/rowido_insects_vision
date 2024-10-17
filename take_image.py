#this server will take image from other server to detect insects

from flask import Flask, request, jsonify
import cv2
from ultralytics import YOLO
import json
from collections import Counter
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the YOLO model
location = 'best.pt'  # Path to your YOLO model
model = YOLO(location)


# Function to process the image and count insects
def process_image(image):
    # Perform prediction
    results = model.predict(source=image, conf=0.25)

    # List to store detected insect names
    detected_insects = []

    # Iterate over the predictions and draw bounding boxes with class labels
    for result in results:
        for box in result.boxes:
            # Extract the object class (name)
            class_id = int(box.cls[0])  # Get the class ID
            class_name = model.names[class_id]  # Get the class name from the model
            detected_insects.append(class_name)

    # Count the occurrence of each insect type
    insect_counts = dict(Counter(detected_insects))

    return insect_counts


# Define the API endpoint to accept image uploads
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read the image file into a format suitable for OpenCV
        image = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # Process the image and get insect counts
        insect_counts = process_image(image)

        # Return the result as JSON
        return jsonify(insect_counts), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Run the app
if __name__ == '__main__':
    # app.run(debug=True, host='0.0.0.0', port=5000)
    app.run(debug=True)
