# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 13:06:16 2021

@author: Burq Amjad
"""

import numpy as np
import pandas as pd
import os

Dataset_1 = pd.read_csv(r'C:\Users\burq_\Downloads\DAAN 888-Design and Implementation of Analytics system\DAAN 888 Datasets\Final datasets\Dataset-1-owid-covid-data.csv')

Dataset_1.drop(Dataset_1.columns[[0]], axis=1, inplace=True)

list(Dataset_1.columns)

Dataset_1.shape
Dataset_1.isnull().sum()
Dataset_1.isnull().sum().sum() # returns the total number of missing values in the entire dataset

# checking which values are null in the continent column
# and finding it's equivalent country in the location column
# for that we make only choose the location and continent columns and save them in a separate variable
# then we'll make the location our index and return the index values for the null values in the continent column
# this way we can get the names of the countries for which the continent column has a null value
# then we can replace the null values with the correct continent name given the country name

dt_continent = Dataset_1[['continent', 'location']]
dt_continent.head()

dt_continent = dt_continent.set_index(dt_continent['location'])
null_values= dt_continent[dt_continent['continent'].isnull()].index.tolist()
set(null_values)

# we see that the rows that has missing values for the continent variable
# have continent names in the location variable
# except for three locations that are named as 'world', 'international' and 'European Union'
# We are going to perform a two step process here
# as we know that there's no such country and continent as 'international', 'world' and 'European Union'
# first we'll replace all the values as 'international' in the location column by 'world' and 'European Union' by 'Europe'
# and then we'll fill all the null values with 'world' in the continent column that has 'world' in the location col

Dataset_1['location'].replace({'International': 'World', 'European Union' : 'Europe'}, inplace=True)
Dataset_1.continent.unique()
Dataset_1['continent'].fillna(Dataset_1['location'], inplace=True)

Dataset_1.dtypes
Dataset_1.tests_units.unique()

# the only object types here are continent, location and tests_units
# replace null values in 'tests_units' with 'units unclear'
# 'units unclear' is one of the values in tests_units col

Dataset_1['tests_units'].fillna('units unclear', inplace=True)


#sorting the dataset according to the date column

Dataset_1['date'] = pd.to_datetime(Dataset_1['date'])
Dataset_1 = Dataset_1.set_index(Dataset_1['date'])
Dataset_1 = Dataset_1.sort_index()

#creating a new dataframe with the vaccination columns only to see where
#the first non-null value for vaccinations is
new_df = Dataset_1.iloc[:, 33:44]
new_df.head()
list(new_df.columns)
new_df.first_valid_index()

#splitting the dataset according to the date we got for the first non null value in the 
#vaccination columns

before_vax = Dataset_1['1/1/2020':'11/30/2020']
after_vax  = Dataset_1['12/1/2020':]

before_vax = pd.DataFrame(before_vax)
after_vax = pd.DataFrame(after_vax)
before_vax.head()
after_vax.head()

before_vax.count()
after_vax.count()

#removing vaccination columns from before_vax dataset

before_vax.drop(before_vax.iloc[:, 33:44], axis = 1, inplace = True)
before_vax.count()

before_vax.isnull().sum()


#removing all the null values where there's a null for total cases 
# and replacing the rest with 0s

before_vax.dropna(axis=0, subset=['total_cases'], inplace=True)
before_vax.isnull().sum()

before_vax.dtypes

after_vax.isnull().sum()

before_vax.fillna(0, inplace=True)
before_vax.isnull().sum()
before_vax.head()

after_vax.dropna(axis=0, subset=['total_cases'], inplace=True)
after_vax.fillna(0, inplace=True)
after_vax.isnull().sum()

before_vax.shape
after_vax.shape

before_vax.to_csv(r'C:\Users\burq_\Downloads\DAAN 888-Design and Implementation of Analytics system\DAAN 888 Datasets\Final datasets\before_vaccination.csv', index = False)

after_vax.to_csv(r'C:\Users\burq_\Downloads\DAAN 888-Design and Implementation of Analytics system\DAAN 888 Datasets\Final datasets\after_vaccination.csv', index = False)

##################################################################################################################
##################################################################################################################
################### DUMMY VARIABLES FOR DATASET-1 ################################################################
##################################################################################################################
##################################################################################################################

##########CREATING DUMMIES FOR CONTINENTS###################

# before vaccination 

cont_dummy_before_vax = pd.get_dummies(data= before_vax, columns=['continent', 'tests_units'])
list(cont_dummy_before_vax.columns)

cont_dummy_before_vax.drop(cont_dummy_before_vax.columns[[0]], axis =1, inplace = True)

cont_dummy_before_vax.shape

# after vaccination
cont_dummy_after_vax = pd.get_dummies(data= after_vax, columns=['continent', 'tests_units'])
list(cont_dummy_after_vax.columns)

cont_dummy_after_vax.drop(cont_dummy_after_vax.columns[[0]], axis =1, inplace = True)
cont_dummy_after_vax.shape

# saving the dataset with countries removed and continents, tests_unit converted to dummies

cont_dummy_before_vax.to_csv(r'C:\Users\burq_\Downloads\DAAN 888-Design and Implementation of Analytics system\DAAN 888 Datasets\Final datasets\before_vaccination_w_continents_dummy.csv', index = False)
cont_dummy_after_vax.to_csv(r'C:\Users\burq_\Downloads\DAAN 888-Design and Implementation of Analytics system\DAAN 888 Datasets\Final datasets\after_vaccination_w_continents_dummy.csv', index = False)

#################### CREATING DUMMIES FOR COUNTRIES##########################

# before vaccination 

country_dummy_before_vax = pd.get_dummies(data= before_vax, columns=['location', 'tests_units'])
list(country_dummy_before_vax.columns)

country_dummy_before_vax.drop(country_dummy_before_vax.columns[[0]], axis =1, inplace = True)

# after vaccination
country_dummy_after_vax = pd.get_dummies(data= after_vax, columns=['location', 'tests_units'])
list(country_dummy_after_vax.columns)

country_dummy_after_vax.drop(country_dummy_after_vax.columns[[0]], axis =1, inplace = True)

# saving the dataset with countries removed and continents, tests_unit converted to dummies

country_dummy_before_vax.to_csv(r'C:\Users\burq_\Downloads\DAAN 888-Design and Implementation of Analytics system\DAAN 888 Datasets\Final datasets\before_vaccination_w_countries_dummy.csv', index = False)
country_dummy_after_vax.to_csv(r'C:\Users\burq_\Downloads\DAAN 888-Design and Implementation of Analytics system\DAAN 888 Datasets\Final datasets\after_vaccination_w_countries_dummy.csv', index = False)

####################################################################################################################
######################### DATASET 3 ################################################################################
####################################################################################################################

# Cleaning Dataset-3

Dataset_3 = pd.read_csv(r'C:\Users\burq_\Downloads\DAAN 888-Design and Implementation of Analytics system\DAAN 888 Datasets\Final datasets\Dataset-3 COVID-19_Vaccinations_in_the_United_States_County.csv')

list(Dataset_3.columns)

Dataset_3.drop(Dataset_3.columns[[1,3,6,7,9,11,14,16,18,20,22,23,24,25,26,27,28,29,30,31]], axis=1, inplace=True)

list(Dataset_3.columns)
Dataset_3.dtypes
Dataset_3.shape

Dataset_3.count()
Dataset_3.isnull().sum()

# we only have null values in the columns for 12+ vaccination 
# That's probably because the vaccination became available later for kids age 12 and up
# we will impute it with 0

Dataset_3.fillna(0, inplace = True)

Dataset_3.to_csv(r'C:\Users\burq_\Downloads\DAAN 888-Design and Implementation of Analytics system\DAAN 888 Datasets\Final datasets\Dataset-3-cleaned.csv', index = False)

############################################################################################################################
###################################### DATASET 5 ##########################################################################
###########################################################################################################################

#Cleaning dataset 5

Dataset_5 = pd.read_csv(r'C:\Users\burq_\Downloads\DAAN 888-Design and Implementation of Analytics system\DAAN 888 Datasets\Final datasets\Dataset-5 COVID-19_Case_Surveillance_Public_Use_Data_with_Geography.csv')

list(Dataset_5.columns)

################## REPLACING INCONSISTENT VALUES IN EVERY COL ###########################################

Dataset_5.replace(['Unknown', 'Missing'], 'not specified', inplace=True)

Dataset_5a = Dataset_5.drop(Dataset_5.columns[[2,3,4,9,10,11]], axis=1)

list(Dataset_5a.columns)
Dataset_5a.shape
Dataset_5a.isnull().sum()
Dataset_5a.dtypes

# gives us the count of unique values in each column

Dataset_5a.apply(pd.value_counts)

# Checking counts of individual columns for more details
Dataset_5a['res_state'].value_counts()
Dataset_5a['age_group'].value_counts()
Dataset_5a['race'].value_counts()

Dataset_5a.age_group.unique()


# As all the remaining columns are categorical, we are going to fill the null values with 'not specified' instead of 0
Dataset_5a.fillna('not specified', inplace = True)
Dataset_5a.isnull().sum()
list(Dataset_5a.columns)

# dropping underlying condition from dataset_5a

Dataset_5b = Dataset_5a.drop(Dataset_5a.columns[[-1]], axis=1)
list(Dataset_5b.columns)

# dropping cols from dataset_5 and then dropping the null values

Dataset_5c = Dataset_5.drop(Dataset_5.columns[[2,3,4,9,10,11,18]], axis =1)
list(Dataset_5c.columns)

Dataset_5c.isnull().sum()

Dataset_5c.dropna(inplace = True)

Dataset_5c.isnull().sum().sum()

############checking values in each column###################

Dataset_5c['res_state'].value_counts()
Dataset_5c.res_state.nunique()
Dataset_5c['age_group'].value_counts()
Dataset_5c['sex'].value_counts()
Dataset_5c['race'].value_counts()
Dataset_5c['ethnicity'].value_counts()
Dataset_5c['exposure_yn'].value_counts()
Dataset_5c['current_status'].value_counts()
Dataset_5c['symptom_status'].value_counts()
Dataset_5c['hosp_yn'].value_counts()
Dataset_5c['icu_yn'].value_counts()
Dataset_5c['death_yn'].value_counts()
Dataset_5['underlying_conditions_yn'].value_counts()

#########################################################################

Dataset_5a.shape
Dataset_5b.shape
Dataset_5c.shape

Dataset_5a.to_csv(r'C:\Users\burq_\Downloads\DAAN 888-Design and Implementation of Analytics system\DAAN 888 Datasets\Final datasets\Dataset-5a-cleaned-w-underlying-conditions-nulls-imputed.csv', index = False)

Dataset_5b.to_csv(r'C:\Users\burq_\Downloads\DAAN 888-Design and Implementation of Analytics system\DAAN 888 Datasets\Final datasets\Dataset-5b-cleaned-wo-underlying-conditions-nulls-imputed.csv', index = False)

Dataset_5c.to_csv(r'C:\Users\burq_\Downloads\DAAN 888-Design and Implementation of Analytics system\DAAN 888 Datasets\Final datasets\Dataset-5c-cleaned-wo-underlying-conditions-nulls-removed.csv', index = False)

