import tensorflow as tf
import os
import sys
import numpy as np
import random
import math
import time
import cv2
from skimage.io import imread, imshow
from skimage.color import rgba2rgb
from tensorflow.python.keras.models import load_model

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_PATH = 'test_case1.png'
MODEL_NAME = 'model_case1.h5'
POSE_GT = [0.14944827  0.11535898  3.11691206  0.69939148 -2.09283805  1.31253995]

# Load the test image
img = imread(IMG_PATH)
img = rgba2rgb(img)
X = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation = cv2.INTER_AREA)
X = np.expand_dims(X, axis=-1)
X = np.expand_dims(X, axis=0)
print(X.shape)

# Load the model
model = load_model(MODEL_NAME)

# Run the inference
start_time = time.time()
pose_pred = model.predict(X)
print("--- %s milliseconds ---" % (time.time()*1000 - start_time*1000))
print(pose_pred[0])

pose_err = ut.comparePoses(POSE_GT, pose_pred[0])
print('Pose Error: '+str(pose_err))