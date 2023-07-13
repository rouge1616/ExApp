 
from __future__ import print_function, absolute_import, division, unicode_literals
import tensorflow as tf
import os
import sys
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import pickle
#import pandas as pd

import utils as ut

from tqdm import tqdm
from itertools import chain

from random import seed
from random import randint

import statistics

 
from skimage.io import imread, imshow
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import elasticdeform

from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Input, BatchNormalization
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Lambda, Conv2DTranspose, concatenate
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.optimizers import Adam

#print(tf.__version__)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

seed(45)

#tf.enable_eager_execution()

CASE_ID = 'case1'
 
IMG_WIDTH = 256
IMG_HEIGHT = 256
DOFS = 6
NBDATA = -1
MODE = 0 # if 1 train on all the set
TOTAL_STYLE = 15 # how many style in total
SHOW_SAMPLE = False
DEF_AUG = False

TRAIN_STYLE = ['style02', 'style03', 'style04', 'style05', 'style06', 'style07', 'style08', 'style09', 'style10', 'style11', 'style12', 'style13', 'style14', 'style15']
VALID_STYLE = ['style01']
K_FOLD = VALID_STYLE[0]



PROJ_PATH = './data/'+str(CASE_ID)+'/proj/'
POSE_PATH = './data/'+str(CASE_ID)+'/poses.pkl'

X_train, X_valid = ut.loadAndSplitRawData(PROJ_PATH, IMG_HEIGHT, IMG_WIDTH, NBDATA, TOTAL_STYLE, TRAIN_STYLE, VALID_STYLE)
y_train = ut.loadPoseDataDict(POSE_PATH, DOFS, NBDATA, len(TRAIN_STYLE))
y_valid = ut.loadPoseDataDict(POSE_PATH, DOFS, NBDATA, len(VALID_STYLE))

print('Data Loaded')
print(X_train.shape)
print(y_train.shape) 
print(X_valid.shape)
print(y_valid.shape) 


if DEF_AUG:
    # deformation augmentation
    alpha_factor = 2.0
    for i in range(0, len(X_train)):
        # apply deformation
        X_train = np.append(X_train, np.expand_dims(ut.elastic_transform(X_train[i], IMG_HEIGHT * alpha_factor, IMG_HEIGHT * 0.08, IMG_HEIGHT * 0.08), axis=0), axis=0)
        y_train = np.append(y_train, np.expand_dims(y_train[i], axis=0), axis=0)
        if i%(len(X_train)/10) == 0:
            print('Deformation augmentation on Train set %: '+str((i/len(X_train))*100))

    # deformation augmentation
    for i in range(0, len(X_valid)):
        # apply deformation with a random 3 x 3 grid
        X_valid = np.append(X_valid, np.expand_dims(ut.elastic_transform(X_valid[i], IMG_HEIGHT * alpha_factor, IMG_HEIGHT * 0.08, IMG_HEIGHT * 0.08), axis=0), axis=0)
        y_valid = np.append(y_valid, np.expand_dims(y_valid[i], axis=0), axis=0)
        if i%(len(X_valid)/10) == 0:
            print('Deformation augmentation on Valid set %: '+str((i/len(X_valid))*100))

    print('Data Augmented')
    print(X_train.shape)
    print(y_train.shape) 
    print(X_valid.shape)
    print(y_valid.shape)         


X_train = X_train/255.0
X_train = np.expand_dims(X_train, axis=-1)
y_train = np.asarray(y_train)

X_valid = X_valid/255.0
X_valid = np.expand_dims(X_valid, axis=-1)
y_valid = np.asarray(y_valid)


if SHOW_SAMPLE:
    fig = plt.figure(figsize=(10,4))    

    ax1 = fig.add_subplot(241)
    ax1.imshow(np.squeeze(X_train[0]))
    ax1.set_title(str(y_train[0]), fontsize=8)

    ax2 = fig.add_subplot(242)
    ax2.imshow(np.squeeze(X_train[1]))
    ax2.set_title(str(y_train[1]), fontsize=8)

    ax3 = fig.add_subplot(243)
    ax3.imshow(np.squeeze(X_train[2]))
    ax3.set_title(str(y_train[2]), fontsize=8)

    ax4 = fig.add_subplot(244)
    ax4.imshow(np.squeeze(X_train[3]))
    ax4.set_title(str(y_train[3]), fontsize=8)

    ax5 = fig.add_subplot(245)
    ax5.imshow(np.squeeze(X_train[4]))
    ax5.set_title(str(y_train[4]), fontsize=8)

    ax6 = fig.add_subplot(246)
    ax6.imshow(np.squeeze(X_train[5]))
    ax6.set_title(str(y_train[5]), fontsize=8)

    ax7 = fig.add_subplot(247)
    ax7.imshow(np.squeeze(X_train[6]))
    ax7.set_title(str(y_train[6]), fontsize=8)

    ax8 = fig.add_subplot(248)
    ax8.imshow(np.squeeze(X_train[7]))
    ax8.set_title(str(y_train[7]), fontsize=8)
    plt.show()





inputs = Input((IMG_HEIGHT, IMG_WIDTH, 3))

c0 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (inputs)
#c0 = Dropout(0.1) (c0)
#c0 = BatchNormalization()(c0)
c0 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c0)
#c0 = BatchNormalization()(c0)

p0 = MaxPooling2D((2, 2)) (c0)

c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p0)
#c1 = Dropout(0.1) (c1)
#c1 = BatchNormalization()(c1)
c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c1)
#c1 = BatchNormalization()(c1)

p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p1)
#c2 = Dropout(0.2) (c2)
#c2 = BatchNormalization()(c2)
c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c2)
#c2 = BatchNormalization()(c2)

p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p2)
#c3 = Dropout(0.2) (c3)
#c3 = BatchNormalization()(c3)
c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c3)
#c3 = BatchNormalization()(c3)

c4 = Flatten()(c3)
c5 = Dense(128, activation='relu')(c4)
c6 = Dense(64, activation='relu')(c5)
c7 = Dense(32, activation='relu')(c6)
outputs = Dense(DOFS, activation='linear')(c7)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

model_filename = "model_"+str(CASE_ID)+".h5"

if MODE == 1:
    results = model.fit(X_train, y_train, batch_size=8, epochs=100)
    model.save(model_filename)
else:    
    callbacks = [
        EarlyStopping(patience=15, verbose=1, min_delta=0.0001),
        ReduceLROnPlateau(factor=0.1, patience=10, min_lr=0.00001, verbose=1),
        ModelCheckpoint(model_filename, verbose=1, save_best_only=True)
    ]
    results = model.fit(X_train, y_train, batch_size=4, epochs=200, callbacks=callbacks, validation_data=(X_valid, y_valid))
            

# Evaluate on validation set (this must be equals to the best log_loss)
print(model.evaluate(X_valid, y_valid, verbose=1))


# Get actual number of epochs model was trained for
N = len(results.history['loss'])

#Plot the model evaluation history
plt.style.use("ggplot")
fig = plt.figure(figsize=(40,8))

fig.add_subplot(1,1,1)
plt.title("Training Loss")
plt.plot(np.arange(0, N), results.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), results.history["val_loss"], label="val_loss")
plt.plot(np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r",label="best model")

plt.show()

np.set_printoptions(suppress=True)

# run inference on validation set

ix = randint(0, X_valid.shape[0])
p_train = model.predict(np.expand_dims(X_valid[ix], axis=0), verbose=1)

print(p_train)
print(y_valid[ix])

ix = randint(0, X_valid.shape[0])
p_train = model.predict(np.expand_dims(X_valid[ix], axis=0), verbose=1)

print(p_train)
print(y_valid[ix])

ix = randint(0, X_valid.shape[0])
p_train = model.predict(np.expand_dims(X_valid[ix], axis=0), verbose=1)

print(p_train)
print(y_valid[ix])
