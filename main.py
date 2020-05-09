from flask import Flask, render_template, request, redirect, session, flash, jsonify
from mysqlconnection import MySQLConnector
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import seaborn as sb
from sklearn.metrics import confusion_matrix


# from __future__ import print_function
import re
import os
import json
import MySQLdb
import MySQLdb.cursors as cursors

EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9.+_-]+@[a-zA-Z0-9._-]+\.[a-zA-Z]+$')
NAME_REGEX = re.compile(r'[0-9]')
PASS_REGEX = re.compile(r'.*[A-Z].*[0-9]')

app = Flask(__name__)
mysql = MySQLConnector(app, 'twitter_data')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/UserLogin')
def user():
    return render_template('UserLogin.html')


@app.route('/Register')
def register():
    return render_template('Register.html')


@app.route('/logout')
def logout():
    return render_template('index.html')


@app.route('/Prediction2', methods=['POST'])
def Prediction2():
    gender = request.form['gender_input']
    height = request.form['height_input']
    weight = request.form['weight_input']
    if gender == "female":
        gender = 1
    if gender == "male":
        gender = 0
    mybmidata = pd.read_csv(r"C:\python37\Disease_Prediction(4 datasets)\bmi_male_female.csv")

    x_bmi = mybmidata.iloc[:, 0:3]
    # x_bmi[x_bmi["Gender"]=="Male"]=0
    # x_bmi[x_bmi["Gender"]=="Female"]=1
    x_bmi["Gender"] = x_bmi["Gender"].map({"Male": 0, "Female": 1})

    y_target = mybmidata.iloc[:, 3]

    Index_name = pd.Series(["Extremely Weak", "Weak", "Normal", "Overweight", "Obesity", "High Obesity"])

    X_input = x_bmi.values
    Y_target = y_target.values

   # X_train = X_input[:]
    #X_test = X_input[400:]

    #Y_train = Y_target[:]
    #Y_test = Y_target[400:]

    from sklearn.model_selection import train_test_split

    X_train, X_test, Y_train, Y_test = train_test_split(X_input, Y_target, test_size=0.2, random_state=42)

    from sklearn.neighbors import KNeighborsClassifier

    trainer = KNeighborsClassifier(n_neighbors=5)

    learner = trainer.fit(X_train, Y_train)

    YA = Y_test
    Yp = learner.predict(X_test)

    from sklearn.metrics import accuracy_score

    acc = accuracy_score(YA, Yp) * 100


    i = learner.predict([[gender, height, weight]])
    result = (Index_name[i].values)[0]

    print(result)
    print("Accuracy  is ", acc)

    return render_template('KNN.html', result=result)

@app.route('/Prediction3', methods=['POST'])
def Prediction3():

    headache = request.form['headache_input']
    seizures = request.form['seizures_input']
    nausea = request.form['nausea_input']
    VP = request.form['VP_input']
    RS = request.form['RS_input']

    mydata = pd.read_csv(r"C:\python37\Disease_Prediction(4 datasets)\SymptomsDataset.csv")


    x = mydata.iloc[:, 0:5]
    y = mydata.iloc[:, 5]

    x_input = x.values
    y_output = y.values

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x_input, y_output, test_size=.2, random_state=110)
    from sklearn.tree import DecisionTreeClassifier
    trainer = DecisionTreeClassifier()
    learner = trainer.fit(x_train, y_train)
    yp = learner.predict([[headache, seizures, nausea, VP, RS]])
    print(yp)

    YA = y_test
    Yp = learner.predict(x_test)

    from sklearn.metrics import accuracy_score

    acc = accuracy_score(YA, Yp) * 100

    if(yp[0]==1) :
        yp='Brain Tumour'
        k = 'Doctor recommended is : Dr Sanjay Batra'
    if(yp[0]==0):
        if (headache == '0' and seizures == '0' and nausea == '0' and VP == '0' and RS == '0'):
            yp = "No disease"
            k = ''
        else:
            yp='Migraine'
            k = 'Doctor recommended is : Dr Rajesh Kota'
    if (yp[0] == 2):
        yp = 'Meningioma'
        k=' Doctor recommended is : Dr Vedika Bhope'


    print("Accuracy  is ", acc)

    return render_template('DT2.html', yp=yp,k=k)

@app.route('/Prediction4', methods=['POST'])
def Prediction4():

    from sklearn.naive_bayes import GaussianNB

    pregnancies = int(request.form['pregnancies_input'])
    glucose = int(request.form['glucose_input'])
    BP = int(request.form['BP_input'])
    ST = int(request.form['ST_input'])
    insulin = int(request.form['insulin_input'])
    bmi = float(request.form['bmi_input'])
    DPF = float(request.form['DPF_input'])
    age= int(request.form['age_input'])

    mydata = pd.read_csv(r'C:\python37\Disease_Prediction(4 datasets)\pima-diabetes.csv')

    mydata["Pregnancies"] = mydata["Pregnancies"].replace(0, np.nan)
    mydata["Glucose"] = mydata["Glucose"].replace(0, np.nan)
    mydata['BloodPressure'] = mydata['BloodPressure'].replace(0, np.nan)
    mydata['SkinThickness'] = mydata['SkinThickness'].replace(0, np.nan)
    mydata['Insulin'] = mydata['Insulin'].replace(0, np.nan)
    mydata['BMI'] = mydata["BMI"].replace(0, np.nan)
    mydata['DiabeticPedigreeFunction'] = mydata['DiabeticPedigreeFunction'].replace(0, np.nan)
    mydata['Age'] = mydata['Age'].replace(0, np.nan)

    mydata["Pregnancies"].fillna(mydata["Pregnancies"].mean(), inplace=True)
    mydata["Glucose"].fillna(mydata["Glucose"].mean(), inplace=True)
    mydata["BloodPressure"].fillna(mydata["BloodPressure"].mean(), inplace=True)
    mydata["SkinThickness"].fillna(mydata["SkinThickness"].mean(), inplace=True)
    mydata["Insulin"].fillna(mydata["Insulin"].mean(), inplace=True)
    mydata["BMI"].fillna(mydata["BMI"].mean(), inplace=True)
    mydata["DiabeticPedigreeFunction"].fillna(mydata["DiabeticPedigreeFunction"].mean(), inplace=True)
    mydata["Age"].fillna(mydata["Age"].mean(), inplace=True)

    mydata.head()

    X = mydata.iloc[:, 0:8]
    Y = mydata.iloc[:, 8]

    from sklearn.model_selection import train_test_split

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


    model = GaussianNB().fit(X_train, Y_train)


    yp = model.predict([[pregnancies, glucose, BP, ST, insulin, bmi, DPF, age]])

    YA = Y_test
    Yp = model.predict(X_test)

    from sklearn.metrics import accuracy_score

    acc = accuracy_score(YA, Yp) * 100

    print(yp)

    if(yp[0]==1) :
        yp='yes '
        k = ' Doctor recommended is : Dr Shrikant Patil'
    if(yp[0]==0):
        yp='no'
        k=''

    print("Accuracy  is ", acc)

    return render_template('NB.html', yp=yp,k=k)


@app.route('/Prediction', methods=['POST'])
def Prediction():

    age = request.form['age_input']
    gender = request.form['gender_input']
    TB = request.form['TB_input']
    DB = request.form['DB_input']
    AP = request.form['AP_input']
    AA = request.form['AA_input']
    AsA = request.form['AsA_input']
    TP = request.form['TP_input']
    A = request.form['A_input']
    AGR = request.form['AGR_input']

    if gender == "female":
        gender = 1
    if gender == "male":
        gender = 0

    mydata = pd.read_csv(r"C:\python37\Disease_Prediction(4 datasets)\indian_liver_patient.csv")

    mydata.fillna(mydata["Albumin_and_Globulin_Ratio"].mean(), inplace=True)

    x = mydata.iloc[:, 0:10]
    y = mydata.iloc[:, 10]

    x.Gender[x.Gender == "Male"] = 0
    x.Gender[x.Gender == "Female"] = 1

    x_input = x.values
    y_output = y.values

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x_input, y_output, test_size=.2, random_state=110)
    from sklearn.tree import DecisionTreeClassifier
    trainer = DecisionTreeClassifier()
    learner = trainer.fit(x_train, y_train)
    yp = learner.predict([[age, gender, TB, DB, AP, AA, AsA, TP, A, AGR]])
    print(yp)

    YA = y_test
    Yp = learner.predict(x_test)

    from sklearn.metrics import accuracy_score

    acc = accuracy_score(YA, Yp) * 100

    if(yp[0]==1) :
        yp='no'
        j=''
    if(yp[0]==2):
        yp='yes'
        j='Doctor recommended is : Dr Prasad Bhate'

    print("Accuracy  is ", acc)

    return render_template('DT.html', yp=yp, j=j)


@app.route('/changeUserPass')
def changeUserPass():
    return render_template('changeUserPass.html')


@app.route('/UserHome')
def user_home():
    return render_template('UserHome.html')

@app.route('/Tab1')
def tab1():
    return render_template('Tab1.html')

@app.route('/Tab2')
def tab2():
    return render_template('Tab2.html')

@app.route('/Tab3')
def tab3():
    return render_template('Tab3.html')

@app.route('/Tab4')
def tab4():
    return render_template('Tab4.html')


@app.route('/reg', methods=['POST'])
def register_user():
    query = "INSERT INTO user (username, password, email, mob) VALUES (:name, :pass, :email_id, :mob)"
    data = {
        'name': request.form['username'],
        'email_id': request.form['email'],
        'mob': request.form['mob'],
        'pass': request.form['password']
    }
    '''
    query2 = "SELECT email FROM user"
    s=mysql.query_db(query2, email)
    print(s)

    if data.==s:
        return render_template('index.html')
    else:'''
    mysql.query_db(query, data)

    return render_template('UserLogin.html')


@app.route('/ulogin', methods=['POST'])
def ulogin():
    username = request.form['username']
    input_password = request.form['password']
    email_query = "SELECT * FROM user WHERE username = :uname and password = :pass"
    query_data = {'uname': str(username), 'pass': str(input_password)}
    stored_email = mysql.query_db(email_query, query_data)
    if not stored_email:
        return redirect('/')

    else:
        if request.form['password'] == stored_email[0]['password']:
            return render_template('UserHome.html',username=username)

        else:
            return redirect('/')


@app.route('/viewbehv')
def viewb():
    return render_template('view_behv.html')


if __name__ == "__main__":
    app.run(debug=True)
