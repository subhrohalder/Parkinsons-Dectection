#import library
import numpy as np
import pandas as pd

#dataset link: https://archive.ics.uci.edu/ml/datasets/Parkinsons

#import dataset
dataset = pd.read_csv('parkinsons.csv')

#data selection
y=dataset.loc[:,'status'].values
X=dataset.drop(columns=['name','status']).values

#Split Train and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.25)

#Standardization(Feature Scaling)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

sc.fit(X_train)
X_train=sc.transform(X_train)
X_test=sc.transform(X_test)

#Model Fitting
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,y_train)

#Prediction
y_pred= model.predict(X_test)

#accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)

