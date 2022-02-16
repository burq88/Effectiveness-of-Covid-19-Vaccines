# Effectiveness-of-Covid-19-Vaccines

1.	All the coding for our project has been carried out in Python.
2.	You’ll need Anaconda on your system to run the project files.
3.	Some visualizations are created in Tableau. You’ll need that installed on your systems as well to reproduce those visualizations.
4.	We have a mix of Jupyter notebooks (IPYNB File) and Python files (PY File).
5.	All the coding files are inside the code folder.
6.	Our Dataset folder contains both the original datasets in the ‘Original dataset’ folder and the cleaned and transformed datasets inside ‘all cleaned-transformed datasets’ folder.
7.	All the datasets have been cleaned using coding that’s saved in the jupyter notebook named ‘DAAN 888 Project-Dataset cleaning’. We also have a python file for the same purpose under the same name.
8.	After cleaning and some transformation we created the following datasets from Dataset-1
a.	before_vaccination
b.	before_vaccination_w_continents_dummy 
c.	before_vaccination_w_countries_dummy
d.	after_vaccination
e.	after_vaccination_w_continents_dummy
f.	after_vaccination_w_countries_dummy
9.	Following datasets were created from Dataset-5:
a.	Dataset-5a-cleaned-w-underlying-conditions-nulls-imputed
b.	Dataset-5b-cleaned-wo-underlying-conditions-nulls-imputed
c.	Dataset-5c-cleaned-wo-underlying-conditions-nulls-removed
10.	We originally had 5 datasets but later we figured out that one of the datasets (Dataset-4) was of no use to us. So, it was removed, and dataset-5 was renamed as Dataset-4 later in our project.
11.	But our cleaned datasets are still named as Dataset-5a, 5b, 5c, so don’t get confused there. The cleaning process was carried out before the decision to remove Dataset-4 was made.
12.	We’ve only used ‘before_vaccination_w_continents_dummy’ and ‘after_vaccination_w_continents_dummy’ for analysis of Dataset-1.
13.	The other extra datasets were created in case we decided to carry out analysis on country level. But we decided to carry out the analysis on continent level only.
14.	Further transformation and analysis were carried out on Dataset-1 (‘before_vaccination_w_continents_dummy’ and ‘after_vaccination_w_continents_dummy’) by using Jupyter notebook named ‘DAAN888_Team2_FeatureTransformations_MLRModeling’. The further transformed datasets were saved as ‘before_vaccination_w_continents_dummy_T’ and ‘after_vaccination_w_continents_dummy_T’.
15.	For Dataset-2, only a visualization was created by using Tableau. It is named as ‘Dataset 2 Visualizations’ in the code folder.
16.	For Dataset-3, Time series analysis were carried out by using Jupyter notebook named ‘DAAN888-Dataset-3-time-series’.
17.	For dataset-4, we used two modeling methods. Random Forest and XGBoost.
18.	For Random Forest the coding in python file named ‘Random_Forest_Dataset_4’ was used and the dataset named ‘Jeong-Dataset-4-COVID-19_Case_Surveillance_Public_Use_Data_with_Geography_cleaned_oct21’ was used for the analysis. That dataset was cleaned using the the python file named ‘clean_Surveilllance’.
19.	For the XGBoost model, the dataset ‘Dataset-5a-cleaned-w-underlying-conditions-nulls-imputed’ was used and the jupyter notebook ‘DAAN888-Dataset-5-XGBoost’ was used.
