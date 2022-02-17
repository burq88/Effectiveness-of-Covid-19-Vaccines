# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 08:14:58 2021

@author: Kim
"""

# Classificatin methods
import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
import os
#from sklearn.model_selection import train_test_split 
#from sklearn import metrics
#import re

pd.set_option('display.max_columns', None) 

# data loading
os.chdir("C:\\Users\\Kim\\Desktop\\PSU\\DANN888\\data\\selected")
df1 = pd.read_csv('COVID-19_Case_Surveillance_Public_Use_Data_with_Geography.csv')

df1.head()

# check null data 
df1.isnull().sum(axis=0)
df1.isnull().values.sum() 

# we drop most null coulums
df2=df1.drop(['case_positive_specimen_interval', 'case_onset_interval', 'underlying_conditions_yn' , 'process'], axis=1)

df2.head()

# check null data 
df2.isnull().sum(axis=0)

df3=df2.dropna()

# now we remove missing unkown attibutes

options = ['yes', 'no','YES', 'NO','Yes', 'No']
# selecting rows based on condition
df3h1 = df3.loc[df3['hosp_yn'].isin(options)]
df3h2 = df3h1.loc[df3h1['icu_yn'].isin(options)]
df3h3 = df3h2.loc[df3h2['death_yn'].isin(options)]
df3h4 = df3h3.loc[df3h3['exposure_yn'].isin(options)]

df3h4.head(10)

df3h5=df3h4[df3h4.ethnicity != 'Missing']
df3h5=df3h5[df3h5.ethnicity != 'Unknown']

df3h5=df3h5[df3h5.race != 'Missing']
df3h5=df3h5[df3h5.race != 'Unknown']

df3h5=df3h5[df3h5.symptom_status != 'Missing']
df3h5=df3h5[df3h5.symptom_status != 'Unknown']

df3h5.describe

df3h5.to_csv('COVID-19_Case_Surveillance_Public_Use_Data_with_Geography_cleaned.csv',index=False)

