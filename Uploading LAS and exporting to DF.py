# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 19:37:53 2020

@author: bruno
"""

#Import libraries. Lasio is a library to handle petrophysical data.
import lasio

# Define the root and relative paths clearly
DATA_DIR = "data/" 
INPUT_FILE = DATA_DIR + "well_logs.csv"
OUTPUT_MODEL_DIR = "models/final/"

#Upload .LAS files (well log files exported from Petrophysics software)
las_well1 = lasio.read("INPUT_FILE")

#Convert .LAS to DATAFRAME
df_well1 = las_well1.df()

#Verify if there are null values
df_well1.isnull().sum()

#Keep only relevant columns and drop null values
df_well1.columns
df_well1_clear =df_well1.dropna(subset=['POR_LAB','PERM_LAB'])
well1_data = df_well1_clear.dropna()

#Export as .CSV
well1_data.to_csv(OUTPUT_MODEL_DIR, index = False)

############## Ends here



