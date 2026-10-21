# %%
# Imports
import csv
import numpy as np
import matplotlib.pyplot as plt
from helpers import *
from Data_preprocessing import *

# Load the data for selected features
data_path = 'data'
x_train, x_test, y_train, train_ids, test_ids, selected_names = load_and_select_data(data_path)

# Overview coding for missing or refused values od selected features
missing_refused_codes = {
    0: [7, 9],
    2: [9],
    3: [9],
    4: [9],
    5: [9],
    8: [9],
    9: [77,99],
    10: [9],
    11: [9],
    12: [9]
}

# Replace missing values by NaN
x_train, x_test = replace_missing_codes(
    x_train, x_test, missing_refused_codes
    )

# All our data at the moment is categorical, adjust here if needed
column_types = ["categorical"] * x_train.shape[1]

# Learn preprocessing settings from the training data, then use the same settings for both sets
preprocessing_values = fit_preprocessor(x_train, column_types)

# Preprocessing
x_train_preproc = transform_data(
    x_train, column_types, preprocessing_values
)
x_test_preproc = transform_data(
    x_test, column_types, preprocessing_values
)
#%%