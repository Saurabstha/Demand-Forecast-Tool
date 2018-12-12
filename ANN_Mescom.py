import numpy as np
import pandas as pd
import keras

from keras.models import Sequential
from keras.layers import Dense

from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score

dataset = pd.read_csv(r"F:\TERI\MESCOM.csv",sep=",")
print(dataset.shape)
dataset.head(3)

dataset = dataset.drop(dataset.query('demand=="NaN"').index)
dataset = dataset.drop(dataset.query('hour=="NaN"').index)
dataset = dataset.drop(dataset.query('hr=="NaN"').index)
dataset = dataset.drop(dataset.query('min=="NaN"').index)
print(dataset.shape)

X = dataset.iloc[:,2:9].values
y = dataset.iloc[:,9].values

y = y.reshape(-1,1)

labelEncoder_X = LabelEncoder()
X[:,3] = labelEncoder_X.fit_transform(X[:,3])

sc_x = StandardScaler()
sc_y = StandardScaler()

X = sc_x.fit_transform(X)
y = sc_y.fit_transform(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

model = Sequential()
model.add(Dense(50,input_dim=7,activation="relu"))
model.add(Dense(100,activation="relu"))
model.add(Dense(50,activation="relu"))
model.add(Dense(1))

model.compile(loss="mean_squared_error",optimizer="adam")
model.fit(X_train,y_train,nb_epoch=200,verbose=2)

y_predict = model.predict(X_test)
score = explained_variance_score(y_test,y_predict)
