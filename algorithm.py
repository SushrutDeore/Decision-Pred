from flask import Flask, render_template, request, redirect, session, flash, jsonify
from mysqlconnection import MySQLConnector


# from __future__ import print_function
import re
import os
import json


EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9.+_-]+@[a-zA-Z0-9._-]+\.[a-zA-Z]+$')
NAME_REGEX = re.compile(r'[0-9]')
PASS_REGEX = re.compile(r'.*[A-Z].*[0-9]')

app = Flask(__name__)
mysql = MySQLConnector(app, 'twitter_data')


@app.route('/')
def index():
    return render_template('prediction.html')



import pandas as pd

mybmidata=pd.read_csv(r"C:\Users\Lenovo\Desktop\datasets\DataSets-master\bmi_male_female.csv")

x_bmi=mybmidata.iloc[:,0:3]
#x_bmi[x_bmi["Gender"]=="Male"]=0
#x_bmi[x_bmi["Gender"]=="Female"]=1
x_bmi["Gender"]=x_bmi["Gender"].map({"Male":0,"Female":1})

y_target=mybmidata.iloc[:,3]

Index_name=pd.Series(["Extremely Weak","Weak","Normal","Overweight","Obesity","High Obesity"])

X_input=x_bmi.values
Y_target=y_target.values

X_train=X_input[:400]
X_test=X_input[400:]

Y_train=Y_target[:400]
Y_test=Y_target[400:]

from sklearn.neighbors import KNeighborsClassifier

trainer=KNeighborsClassifier(n_neighbors=5)

learner=trainer.fit(X_train,Y_train)

i=learner.predict([[0,100,25]])
k=Index_name[i]
print(Index_name[i])

