import os
from flask import Flask, request, jsonify, render_template
import cv2 as opencv
import torch
import numpy as np
import base64

app = Flask(__name__)

confidence = 0.5
num_bottles = 0
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./best.pt')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # get image file from POST request
    file = request.files['image']

    # read image using OpenCV
    img = opencv.imdecode(np.fromstring(file.read(), np.uint8), opencv.IMREAD_UNCHANGED)

    # resize image
    img1= opencv.resize(img, (420, 420))

    # run object detection model
    results = model(img1)

    # extract bounding boxes, labels, and scores from results
    boxes = results.xyxy[0].numpy()
    labels = results.names[0]
    scores = results.xyxyn[0][:, 4].numpy()

    # loop through each detection and count number of bottles
    num_bottles = 0
    for box, score in zip(boxes, scores):
        if score >= confidence:
            x1, y1, x2, y2, _, _ = box
            color = (0, 255, 0)
            thickness = 2
            opencv.rectangle(img1, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
            num_bottles = num_bottles + 1

    # encode image to base64 format
    _, buffer = opencv.imencode('.jpg', img1)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    # create response object
    response = {
        'num_bottles': num_bottles,
        'image': img_base64
    }

    # return JSON response
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
