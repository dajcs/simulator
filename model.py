#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#from keras.applications.vgg16 import VGG16
#from keras.applications.inception_v3 import InceptionV3
#from keras.applications.vgg16 import preprocess_input
#from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.noise import GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Dropout, Flatten, ELU
#from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from sklearn.utils import shuffle

import cv2
import json
import numpy as np
import pandas as pd
from scipy import misc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#csv_file = 'tr1/driving_log.csv'
#plt.hist(drlog['speed'],50)

def load_cam(drlog, cam, shift):
    X_train = []
    Xf_train = []
    for f in drlog[cam]:
        img = misc.imread(f.lstrip())[12:140,:,:] # 128x320  # read only a strip of the road view
        img = cv2.GaussianBlur(img,(3,3),0)
        img = misc.imresize(img, 0.5)             #  64x160  # make it smaller to fit into memory (8GB RAM)
        imgf= cv2.flip(img,1)          # flipping img around y axe
        img = img.astype(np.float32)
        imgf= imgf.astype(np.float32)
        img = img/255. - 0.5                      # range [-0.5, 0.5]
        imgf = img/255. - 0.5                      # range [-0.5, 0.5]
        X_train.append(img)
        Xf_train.append(imgf)
    X_train = np.array(X_train)
    Xf_train = np.array(Xf_train)
    y_train = np.array([drlog['steering_angle'] + shift, drlog['speed']])
    y_train = np.swapaxes(y_train,0,1)
    yf_train = np.array([-(drlog['steering_angle'] + shift), drlog['speed']])  # flip has -steering_angle
    yf_train = np.swapaxes(yf_train,0,1)
    X_train = np.concatenate((X_train, Xf_train), axis=0)
    y_train = np.concatenate((y_train, yf_train), axis=0)
    return X_train, y_train


def train_test(csv_file, left_shift, right_shift):
    header = ['center_cam', 'left_cam', 'right_cam', 'steering_angle', 'throttle', 'break', 'speed']
    drlog = pd.read_csv(csv_file, names=header)
    drlog.loc[drlog['speed']<10, 'speed'] = 10  # make speed at least 10 -> range [10..30]
    drlog['speed'] -= 20  # range [-10, 10]
    drlog['speed'] /= 50  # range [-0.25, 0.25]

    X, y = load_cam(drlog, 'center_cam', 0)
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
    print('center_cam loaded')

    X_left, y_left = load_cam(drlog, 'left_cam', left_shift)
#    Xl_train, Xl_test, yl_train, yl_test = train_test_split(X_left, y_left, test_size=0.05, random_state=42)
    print('left_cam loaded')

    X_right, y_right = load_cam(drlog, 'right_cam', right_shift)
#    Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_right, y_right, test_size=0.05, random_state=42)
    print('right_cam loaded')

    X = np.concatenate((X, X_left, X_right), axis=0)
    y = np.concatenate((y, y_left, y_right), axis=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

    return X_train, y_train, X_test, y_test

# track1 #######
print('Loading data for track1')
X1, y1, X1_test, y1_test = train_test('tr1/driving_log.csv', 0.14, -0.14)
#X1,y1 = shuffle(X1,y1)
print('track1 training and testing data has been loaded')

# track2 ########
#print('Loading data for track2')
#X2, y2, X2_test, y2_test = train_test('tr2/driving_log.csv', 0.15, -0.16)
##X2, y2 = shuffle(X2, y2)
#print('track2 training and testing data has been loaded')

#X_train = np.concatenate((X1, X2), axis=0)
#y_train = np.concatenate((y1, y2), axis=0)
#X_test = np.concatenate((X1_test, X2_test), axis=0)
#y_test = np.concatenate((y1_test, y2_test), axis=0)
#print('Training & testing data concatenate ok')
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
#X_train, y_train = shuffle(X_train, y_train)

#print('X_train.shape:',X_train.shape)
#print('y_train.shape:',y_train.shape)
#print('X_test.shape:',X_test.shape)
#print('y_test.shape:',y_test.shape)



np.random.seed(1337)

batch_size = 32
nb_epoch = 2


nb_filters = 16
pool_size = (2,2)
input_shape = (64, 160, 3)
print('input_shape=', input_shape)

model = Sequential()

#model.add(GaussianNoise(0.05, input_shape=input_shape))

model.add(Convolution2D(3*nb_filters, 5, 5, border_mode='same', input_shape=input_shape)) #  3 * 48x160
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=pool_size))                   # 48 * 32x80

model.add(Convolution2D(4*nb_filters, 5, 5, border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=pool_size))            # 64 * 16x40

model.add(Convolution2D(5*nb_filters, 5, 5, border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=pool_size))            # 80 * 8x20

model.add(Convolution2D(6*nb_filters, 3, 3,  border_mode='same')) # subsample=(1,2),
model.add(BatchNormalization())
model.add(Activation('elu'))                            # 96 * 8x20
model.add(MaxPooling2D(pool_size=pool_size))            # 96 * 4x10

model.add(Flatten())                                    # 3840

model.add(Dense(2048))
model.add(Dropout(0.33))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))

model.add(Dense(512))
model.add(Dropout(0.33))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))

model.add(Dense(128))
model.add(Dropout(0.22))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))
model.add(Dense(2))

model.summary()

model.compile(loss='mse', optimizer='adam')
#checkpointer = ModelCheckpoint(filepath='model.h5', verbose=1, save_best_only=True)

#for i in range(5):
#    history = model.fit(X1, y1,
#                        batch_size=batch_size, nb_epoch=1,
#                        verbose=1, validation_data=(X1_test, y1_test))#, callbacks=[checkpointer])
##    history = model.fit(X2, y2,
##                        batch_size=batch_size, nb_epoch=1,
##                        verbose=1, validation_data=(X2_test, y2_test))#, callbacks=[checkpointer])
#    mjs = 'model'+str(i+1)+'.json'
#    mh5 = 'model'+str(i+1)+'.h5'
#    model_json = model.to_json()
#    with open(mjs, 'w') as f:
#        json.dump(model_json, f)
#    model.save_weights(mh5)

history = model.fit(X1, y1,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X1_test, y1_test))#, callbacks=[checkpointer])

model_json = model.to_json()
with open('model.json', 'w') as f:
    json.dump(model_json, f)
model.save_weights('model.h5')


## Loading model
#with open('model.json', 'r') as jfile:
#    model = model_from_json(json.load(jfile))
#
## Compiling model
#model.compile(optimizer="adam", loss="mse", verbose=1)
#model.load_weights('model.h5')
#
#
#header = ['center_cam', 'left_cam', 'right_cam', 'steering_angle', 'throttle', 'break', 'speed']
#drlog = pd.read_csv('./tr0/driving_log.csv', names=header)
#
#X_test, y_test = load(drlog, 'left_cam')
##plt.imshow(X_test[1111]+0.5)
#y_pred_left = model.predict(X_test)
#
#
#X_test, y_test = load(drlog, 'right_cam')
#plt.imshow(X_test[1111]+0.5)
#y_pred_right = model.predict(X_test)
