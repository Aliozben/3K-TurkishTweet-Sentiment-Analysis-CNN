
# -*- coding: utf-8 -*-
"""
        Created on Sun May 17 15:29:17 2020

    @author: Abi
"""

texts_=[]
labels_ = []

folders=[ '1','2', '3']
filename='data/3000tweet/raw_texts/'
for x in folders:
    tweet_count=0 
    path=os.path.join(filename, x)        
    for t in os.listdir(path):
        tweet_count+=1   
        p2=os.path.join(path,t)          
        f=open(p2, "r")          
        texts_.append(f.read())
        labels_.append(x)  
        f.close()
    print(x,'. Klasör tweet sayısı :',tweet_count)


from keras.preprocessing.text import Tokenizer    
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from keras.utils import to_categorical

token = Tokenizer()
token.fit_on_texts(texts_)
texts_ = token.texts_to_sequences(texts_)
texts_ = pad_sequences(texts_)

texts_ =StandardScaler().fit_transform(texts_)

labels_ = preprocessing.LabelEncoder().fit_transform(labels_)
labels_ = to_categorical(labels_)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =\
     train_test_split(texts_, labels_, test_size = 0.5)

max_futures=1500
maxlen=28 
batch_size=40   
embedding_dims=400
epochs=3

from keras.models import Sequential
from keras.layers import Dense , Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D

print('Model build..')
model=Sequential()
model.add(Embedding(max_futures, embedding_dims,input_length=maxlen))
    

model.add(Conv1D(embedding_dims, 3, padding='valid', activation='relu', strides=1))
model.add(GlobalMaxPooling1D())

model.add(Dense(3))
model.add(Activation('sigmoid'))
    
model.compile(loss='binary_crossentropy', optimizer='Adadelta', metrics=['accuracy'])
model.summary()
t=model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))


la_ratio=model.evaluate(x_test,y_test)
print('Loss/Accuracy :', la_ratio)

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
