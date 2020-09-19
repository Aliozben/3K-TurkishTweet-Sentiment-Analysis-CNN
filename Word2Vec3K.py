import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Conv1D, MaxPool1D, GlobalMaxPool1D, Activation, LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
import re
import nltk
from nltk.corpus import stopwords
from sklearn import preprocessing
from gensim.parsing.preprocessing import remove_stopwords
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from os import listdir
from os.path import isfile, join
import pandas as pd  # For data handling
from time import time  # To time our operations
from collections import defaultdict  # For word frequency

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


#Gereksiz kelime, boşluk, işaretleri silmek için
nltk.download('stopwords')
stop_word_list = stopwords.words('turkish')
stop_word_list

def preprocess_text(sen):
    sentence = re.sub('[\d\s]', ' ', str(sen))
    sentence = re.sub('[^\w\s]', ' ', str(sentence))
    sentence = re.sub(r"\b[\w\s]\b", ' ',str(sentence))
    sentence = re.sub(r'\s+', ' ', sentence)

    WPT = nltk.WordPunctTokenizer()
    tokens = WPT.tokenize(sentence)
    filtered_tokens = [token for token in tokens if token not in stop_word_list]
    single_doc = ' '.join(filtered_tokens)    
    return single_doc.lower()


x=[]
# print(texts_)
for i in texts_:
    x.append(preprocess_text(i))
pd.DataFrame(data=x)
y=labels_

words = []
for i in x:
    words.append(i.split())
word2vec_model = Word2Vec(words, size = 200, window = 5, min_count = 1, workers = 16, sample=0.01,  min_alpha=0.0001, negative=0)
token = Tokenizer()
token.fit_on_texts(x)
x = token.texts_to_sequences(x)
x = pad_sequences(x)

scaler = StandardScaler()
x = scaler.fit_transform(x)

encode = preprocessing.LabelEncoder()
y = encode.fit_transform(y)
y = to_categorical(y)

#Train ve Test işlemleri için
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5, random_state = 0)

print('x Train : ' + str(x_train.shape))
print('x Test : ' + str(x_test.shape))
print('y Train : ' + str(y_train.shape))
print('y Test : ' + str(y_test.shape))


model = Sequential()
model.add(word2vec_model.wv.get_keras_embedding(True))
model.add(LSTM(units=128))
model.add(Dense(3, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
t = model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))

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

scores = model.evaluate(x_test, y_test, verbose = 0)
print('Test score:', scores[0]*100)
print('Test accuracy:', scores[1]*100)