# -*- coding: utf-8 -*-
"""
Created on Sun May 31 15:57:12 2020

@author: Abi
"""
# import csv
# data = reduce_data.ProcessData(segment.da, nbprocs)
# # read flash.dat to a list of lists
# datContent = [i.strip().split() for i in open("./segment.dat").readlines()]

# # write it as a new CSV file
# with open("./segment.csv", "wb") as f:
#     writer = csv.writer(f)
    # writer.writerows(datContent)

import pandas as pd
df=pd.read_csv('segment.dat', sep=' ')
print(df)

x_data=df.iloc[:,1:20].values
y_data=df.iloc[:,-1].values
print(x_data.shape)
print(y_data.shape)

from keras.utils import np_utils

y_data=np_utils.to_categorical(y_data)
print(y_data)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
print(x_train.shape)
print(x_test.shape)

import keras
from keras.models import Sequential
from keras.layers import Dense

model=Sequential()

model.add(Dense(256, activation='relu',input_dim=19))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='softmax'))
#Adagrad - RMSprop
model.compile(optimizer='Adagrad',loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
t=model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=19, epochs=4)

scores=model.evaluate(x_test,y_test)
print('accuracy :', scores)

from matplotlib import pyplot as plt


plt.plot(t.history['accuracy'],color='b', label="Training accuracy")
plt.plot(t.history['val_accuracy'],color='y', label="Test accuracy")
plt.plot(t.history['loss'],color='g', label="Training loss")
plt.plot(t.history['val_loss'],color='r', label="Test accuracy")
plt.title('Model')
plt.xlabel('Epoch')
plt.legend(['Train Accuracy','Test Accuracy','Train Loss','Test Loss'],loc='bottom left')
print(plt.show())

from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import numpy as np

rounded_pred = model.predict_classes(x_test, batch_size=128, verbose=0)
rounded_labels=np.argmax(y_test, axis=1)
print(confusion_matrix(rounded_labels,rounded_pred))
print(plot_confusion_matrix(conf_mat=confusion_matrix(rounded_labels,rounded_pred)))

scores = model.evaluate(x_test, y_test, verbose = 0)
print('Test score:', scores[0]*100)
print('Test accuracy:', scores[1]*100)