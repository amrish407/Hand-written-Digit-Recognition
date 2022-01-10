import cv2 as cv
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

from sklearn.datasets import fetch_openml
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
import sklearn.metrics as metrics
import tensorflow as tf
import keras
from tensorflow.keras.callbacks import TensorBoard
import time

mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()

x_train

print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

x_train = x_train.astype('float32')
x_train = x_train / 255
x_test = x_test.astype('float32')
x_test = x_test / 255

x_train[10]

x_train.shape

for i in range(250,259):
       plt.subplot(330+(i+1))
       plt.imshow(x_train[i],cmap=plt.get_cmap('gray'))
       
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))
model.compile(optimizer = "adam",loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=3)
prediction = model.predict([x_test])

print("probability")
prediction[0]

print(np.argmax(prediction[25]))

plt.imshow(x_test[38],cmap = plt.cm.binary)





   
   