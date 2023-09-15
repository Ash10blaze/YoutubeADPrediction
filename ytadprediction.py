import pandas as pd 
import numpy as np
import gradio as gr
data=pd.read_csv('adprediction.csv')
#print(data)
#data cleaning
#checking dtype
#print(data.info())
#checking null values
#print(data.isnull().sum())
#removing unwanted columns
data.drop(['Ad Topic Line','City','Country','Timestamp',],axis=1,inplace=True)
#print(data.columns)
x=data.drop(['Clicked on Ad'],axis=1)
#print(x)
y=data['Clicked on Ad']
#print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
from sklearn.metrics import classification_report
#print(classification_report(y_test,y_pred))
def classify(num):
    if(num>0.5):
        return "Clicked On AD"
    else:
        return "Didnt Click On AD"
def predict_adclick(time_spent_on_site, age, area_income, daily_internet_usage, is_male):
    input_array=np.array([[time_spent_on_site, age, area_income, daily_internet_usage, is_male]])
    pred=model.predict(input_array)
    output=classify(pred[0])
    return output
iface=gr.Interface(fn=predict_adclick,
    inputs = [
    gr.inputs.Number(label="Enter your daily Time(in hrs)Spent on Site"),
    gr.inputs.Number(label="Enter Your Age"),
    gr.inputs.Number(label="Enter Area Income(In Rs)"),
    gr.inputs.Number(label="Enter Your Daily Internet Usage(KBs,GBs,MBs etc)"),
    gr.inputs.Number(label="Gender(1 for Male 0 for Female)")
],outputs="text",live=True)
iface.launch(share=True,debug=True)

