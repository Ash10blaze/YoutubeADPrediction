import pandas as pd 
import numpy as np
data=pd.read_csv('adprediction.csv')
#print(data)
#data cleaning
#checking dtype
#print(data.info())
#checking null values
#print(data.isnull().sum())
#removing unwanted columns
data.drop(['Ad Topic Line','City','Country','Timestamp',],axis=1,inplace=True)
#print(data)

#Modeling(using LogisticRegression)
x=data.iloc[:,:-1].values
y=data.iloc[:,5].values
#splitting training and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
#importing model
from sklearn.linear_model import LogisticRegression
Regressor=LogisticRegression()
Regressor.fit(x_train,y_train)
#prediction
y_pred=Regressor.predict(x_test)
print(y_pred)
#checking accuracy
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))
