#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 21:02:55 2018

@author: deepayanbhadra
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

trainImages = np.squeeze(np.load('trainImages.npy'))
testImages = np.squeeze(np.load('testImages.npy'))
trainLabels = np.squeeze(np.load('trainLabels.npy'))
testLabels = np.squeeze(np.load('testLabels.npy'))

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Preprocess the data : Scale these values to a range of 0 to 1
trainImages = trainImages / 255.0
testImages = testImages / 255.0

# Build the model

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])


# Compile the model

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model

history = model.fit(trainImages, trainLabels,  
                     nb_epoch=10, batch_size=32,  
                     validation_data=(testImages, testLabels)) 

# summarize history for accuracy  
   
 plt.subplot(211)  
 plt.plot(history.history['acc'])  
 plt.plot(history.history['val_acc'])  
 plt.title('model accuracy')  
 plt.ylabel('accuracy')  
 plt.xlabel('epoch')  
 plt.legend(['train', 'test'], loc='upper left')  
   
 # summarize history for loss  
   
 plt.subplot(212)  
 plt.plot(history.history['loss'])  
 plt.plot(history.history['val_loss'])  
 plt.title('model loss')  
 plt.ylabel('loss')  
 plt.xlabel('epoch')  
 plt.legend(['train', 'test'], loc='upper left')  
 plt.show()  

# Evaluate accuracy
test_loss, test_acc = model.evaluate(testImages, testLabels)
print('Test accuracy:', test_acc)

# Make predictions

predictions = model.predict(testImages)
print(np.argmax(predictions[0]))
print(testLabels[0])

# 2nd choice of model for Fashion MNIST

from keras.layers import Conv2D, MaxPool2D, Flatten
from keras.layers import Dense, Dropout

model = keras.models.Sequential([
    keras.layers.Conv2D(32, (5, 5), padding="same", input_shape=[28, 28, 1]),
    keras.layers.MaxPool2D((2,2)),
    keras.layers.Conv2D(64, (5, 5), padding="same"),
    keras.layers.MaxPool2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(1024, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])

X_train = X_train.reshape([-1, 28, 28, 1])
X_test = X_test.reshape([-1, 28, 28, 1])
X_train = X_train/255
X_test = X_test/255

model.compile(keras.optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train,  
                     nb_epoch=10, batch_size=32,  
                     validation_data=(X_test, y_test)) 









