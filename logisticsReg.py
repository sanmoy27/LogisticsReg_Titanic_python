# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
###Logistics Regression
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn

from pandas import Series, DataFrame
from pylab import rcParams
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics 
from sklearn.metrics import classification_report

url = 'https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/titanic-train.csv'
titanic = pd.read_csv(url)
titanic.head()
titanic.info()
titanic_data = titanic.drop(['PassengerId','Name','Ticket','Cabin'], 1)
titanic_data.head()

titanic_data.isnull().sum()

np.mean(titanic_data.loc[titanic_data['Pclass']==2, 'Age'])

def age_approx(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return np.mean(titanic_data.loc[titanic_data['Pclass']==1, 'Age'])
        elif Pclass == 2:
            return np.mean(titanic_data.loc[titanic_data['Pclass']==2, 'Age'])
        else:
            return np.mean(titanic_data.loc[titanic_data['Pclass']==3, 'Age'])
    else:
        return Age
    
titanic_data['Age'] = titanic_data[['Age', 'Pclass']].apply(age_approx, axis=1)
titanic_data.isnull().sum()
titanic_data.dropna(inplace=True)
gender = pd.get_dummies(titanic_data['Sex'], drop_first=True)
gender.head()
embark_loc = pd.get_dummies(titanic_data['Embarked'], drop_first=True)
embark_loc.head()
titanic_data.drop(['Sex', 'Embarked', 'Pclass', 'Fare'], axis=1, inplace=True)
titanic_data.head()
titanic_dmy=pd.concat([titanic_data, gender, embark_loc], axis=1)
titanic_dmy.head()
titanic_dmy.info()
X=titanic_dmy.iloc[:,1:7].values
y=titanic_dmy.iloc[:,0].values

#Splitting into train and test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Preprocessing
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# Fitting Logistic Model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Predict on test data
y_pred = classifier.predict(X_test)

# ConfusionMatrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))



