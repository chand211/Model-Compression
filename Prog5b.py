import numpy as np
import sklearn as sk
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses
from tensorflow.keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt
from scipy import linalg
import csv
import random

#rank
r = 10

with open('firstwts.csv', newline='') as csvfile:
    layerone = np.array(list(csv.reader(csvfile)))
with open('secondwts.csv', newline='') as csvfile:
    layertwo = np.array(list(csv.reader(csvfile)))
with open('thirdwts.csv', newline='') as csvfile:
    layerthree = np.array(list(csv.reader(csvfile)))

Ulone,Slone,VHlone = linalg.svd(layerone, full_matrices=False)
Ultwo,Sltwo,VHltwo = linalg.svd(layertwo, full_matrices=False)
Ulthree,Slthree,VHlthree = linalg.svd(layerthree, full_matrices=False)

Ulone = Ulone[:,:r]
Slone = np.diag(Slone[:r])
VHlone = VHlone[:r,:]

Ultwo = Ultwo[:,:r]
Sltwo = np.diag(Sltwo[:r])
VHltwo = VHltwo[:r,:]

Ulthree = Ulthree[:,:r]
Slthree = np.diag(Slthree[:r])
VHlthree = VHlthree[:r,:]

L1 = np.dot(Slone,VHlone)
print(Ulone.shape[0],Ulone.shape[1])
print(L1.shape[0],L1.shape[1])

L2 = np.dot(Sltwo,VHltwo)
print(Ultwo.shape[0],Ultwo.shape[1])
print(L2.shape[0],L2.shape[1])

L3 = np.dot(Slthree,VHlthree)
print(Ulthree.shape[0],Ulthree.shape[1])
print(L3.shape[0],L3.shape[1])

#Reading in data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Saving data
y_test1 = y_test

#Flattening each image
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])

#Regularizing images
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

#Using one-hot encoding
y_train = np_utils.to_categorical(y_train,10)
y_test = np_utils.to_categorical(y_test,10)

#Constructing the model
model = models.Sequential()
model.add(layers.Dense(r,input_dim=784,activation='relu'))
model.add(layers.Dense(100,activation='relu'))
model.add(layers.Dense(r,activation='relu'))
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dense(r,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

model.build()

def biases(x):
    array = np.array([random.random() for i in range(x)])
    array = array.reshape(x,)
    return array

model.layers[0].set_weights([ Ulone,biases(r) ])
model.layers[1].set_weights([L1,biases(100)])
model.layers[2].set_weights([ Ultwo,biases(r) ])
model.layers[3].set_weights([L2,biases(50)])
model.layers[4].set_weights([ Ulthree,biases(r) ])
model.layers[5].set_weights([L3,biases(10)])

#compiling the model
model.compile(optimizer='adam', loss=losses.CategoricalCrossentropy(), metrics=['accuracy'])

history = model.fit(x_train, y_train,
                epochs=10,
                shuffle=True,
                validation_data=(x_test, y_test))

model.summary()

yp_test = model.predict(x_test)
yp_test = np.argmax(yp_test, axis=1)
cm_test = confusion_matrix(y_test1, yp_test)
t3 = ConfusionMatrixDisplay(cm_test)
t3.plot()
plt.show()
plt.clf()

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy and loss')
plt.ylim([min(history.history['val_loss']), max(history.history['val_loss'])])
plt.legend(loc='lower right')
plt.show()


