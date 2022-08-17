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

#constructing the model
model = models.Sequential()
model.add(layers.Dense(100, input_dim=784,activation='relu'))
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

#compiling the model
model.compile(optimizer='adam', loss=losses.CategoricalCrossentropy(), metrics=['accuracy'])

history = model.fit(x_train, y_train,
                epochs=100,
                shuffle=True,
                validation_data=(x_test, y_test))

model.summary()

first_layer_weights = model.layers[0].get_weights()[0]
np.savetxt('firstwts.csv', first_layer_weights, delimiter=',')

second_layer_weights = model.layers[1].get_weights()[0]
np.savetxt('secondwts.csv', second_layer_weights, delimiter=',')

third_layer_weights = model.layers[2].get_weights()[0]
np.savetxt('thirdwts.csv', third_layer_weights, delimiter=',')

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
plt.ylim([0.75, 1])
plt.legend(loc='lower right')
plt.show()

plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy and loss')
plt.ylim([min(history.history['val_loss']), max(history.history['val_loss'])])
plt.legend(loc='lower right')
plt.show()
