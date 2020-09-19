import pandas as pd
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

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

print(x_train.shape)
print(x_test.shape)
k_range = range(1, 9)

scores = []

for k in k_range:
    classifier = KNeighborsClassifier(n_neighbors=k, algorithm='auto' ,metric='manhattan', p = 2, weights='uniform', leaf_size=30, n_jobs=-1)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))

print(scores)
print('Accuracy:'+ str((sum(scores)/float(len(scores)))))

