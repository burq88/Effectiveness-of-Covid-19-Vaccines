# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 11:39:36 2021

@author: Kim
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

import warnings
warnings.filterwarnings('ignore')

# Loading data
os.chdir("C:\\Users\\Kim\\Desktop\\PSU\\DANN888\\data\\selected")
df = pd.read_csv('COVID-19_Case_Surveillance_Public_Use_Data_with_Geography_cleaned_oct21.csv')
df.head()
df.columns

df['icuF']=pd.factorize(df['icu_yn'])[0]
(unique, counts) = np.unique(df.icuF, return_counts=True)
(unique2, counts2) = np.unique(df.icu_yn, return_counts=True)
print( unique,unique2)
print( counts,counts2)

df['hospF']=pd.factorize(df['hosp_yn'])[0]
(unique, counts) = np.unique(df.hospF, return_counts=True)
(unique2, counts2) = np.unique(df.hosp_yn, return_counts=True)
print( unique,unique2)
print( counts,counts2)

df['deathF']=pd.factorize(df['death_yn'])[0]
(unique, counts) = np.unique(df.deathF, return_counts=True)
(unique2, counts2) = np.unique(df.death_yn, return_counts=True)
print( unique,unique2)
print( counts,counts2)

# transform category variables to dummy varaibles
dm1 = pd.get_dummies(df["age_group"])
dm2 = pd.get_dummies(df["sex"])
dm3 = pd.get_dummies(df["race"])
dm4 = pd.get_dummies(df["ethnicity"])
dm5 = pd.get_dummies(df["current_status"])
dm6 = pd.get_dummies(df["symptom_status"])

frames = [dm1, dm2,dm3,dm4,dm5,dm6]
x1 = pd.concat(frames,axis=1)
x2=df[['icuF','hospF' ]]
X= pd.concat( [x1, x2],axis=1)

# set the target variable death variable 
y = df.deathF

# input variables
X.columns
X.head(2)

#  split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

# Random Forest model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

(unique3, counts3) = np.unique(predictions, return_counts=True)
(unique4, counts4) = np.unique(y_test, return_counts=True)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions)
TN, FP, FN, TP = confusion_matrix(y_test, predictions).ravel()

print('True Positive(TP)  = ', TP)
print('False Positive(FP) = ', FP)
print('True Negative(TN)  = ', TN)
print('False Negative(FN) = ', FN)
accuracy0 =  (TP+TN) /(TP+FP+TN+FN)
print('Accuracy of the binary classification = {:0.3f}'.format(accuracy0))

# important features 
import pandas as pd
feature_imp = pd.Series(model.feature_importances_ , index=['0 - 17 years', '18 to 49 years', '50 to 64 years', '65+ years',
       'Female', 'Male', 'American Indian/Alaska Native', 'Asian', 'Black',
       'Multiple/Other', 'Native Hawaiian/Other Pacific Islander', 'White',
       'Hispanic/Latino', 'Non-Hispanic/Latino', 'Laboratory-confirmed case',
       'Probable Case', 'Asymptomatic', 'Symptomatic', 'icuF', 'hospF']).sort_values(ascending=False)

feature_imp

# Cross check validation
# Accuracy vs n_estimator
from sklearn.model_selection import cross_val_score
from sklearn import metrics

n_estimator = range(50, 300, 20)
accuracy = []
accuracy_test=[]
for i in n_estimator:
    RF = RandomForestClassifier(n_estimators= i, random_state = 0)
    scores = cross_val_score(RF, X_train, y_train, cv=10)    
    accuracy.append(scores.mean())
    RF.fit(X_train, y_train)
    RF_pred = RF.predict(X_test)
    accuracy_test.append(metrics.accuracy_score(y_test, RF_pred))
    
plt.rcParams['figure.figsize'] = [6, 6] 
plt.figure()    
plt.plot(n_estimator, accuracy)
plt.title('Ensemble Accuracy')
plt.ylabel('Accuracy train data')
plt.xlabel('Number of base estimators in ensemble')
plt.ylim([0.7, 1])
print('best n_estiamtor = ', n_estimator[accuracy.index(max(accuracy))], ', Accuracy = ', max(accuracy)) 


