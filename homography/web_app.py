# web_app.py
from flask import Flask, render_template, request, redirect, url_for, send_file
import numpy as np
import cv2
import os
import json

app = Flask(__name__)

# Ensure the uploads folder exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Get the original image dimensions
    image = cv2.imread(filepath)
    height, width = image.shape[:2]

    return render_template('transform_drag.html', filename=file.filename, width=width, height=height)

@app.route('/transform', methods=['POST'])
def transform_image():
    filename = request.form['filename']
    coords = request.form.getlist('coords[]')
    coords = np.array(json.loads(coords[0]))  # Load the JSON string into a numpy array

    # Ensure the coordinates are in the correct order
    if len(coords) != 4:
        return "Error: Exactly four coordinates are required."

    image_path = os.path.join(UPLOAD_FOLDER, filename)
    image = cv2.imread(image_path)

    # Apply the four-point transformation
    warped_image = four_point_transform(image, coords)

    output_path = os.path.join(UPLOAD_FOLDER, 'warped_' + filename)
    cv2.imwrite(output_path, warped_image)

    return send_file(output_path, as_attachment=True)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename))

if __name__ == '__main__':
    app.run(debug=True)