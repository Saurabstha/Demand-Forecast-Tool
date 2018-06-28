import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.metrics import explained_variance_score
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"F:\TERI\MESCOM.csv",sep=",")
dataset.head()

dataset = dataset.drop(dataset.query('demand=="NaN"').index)
dataset = dataset.drop(dataset.query('hour=="NaN"').index)
dataset = dataset.drop(dataset.query('hr=="NaN"').index)
dataset = dataset.drop(dataset.query('min=="NaN"').index)

X = dataset.iloc[:,2:9].values
y = dataset.iloc[:,9].values

labelEncoder_X = LabelEncoder()
X[:,3] = labelEncoder_X.fit_transform(X[:,3])


reg = RandomForestRegressor()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
reg.fit(X_train,y_train)

y_pred = reg.predict(X_test)

score = explained_variance_score(y_test,y_pred)
print(score)
