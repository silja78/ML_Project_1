# Exploratory data analysis
# %%
# Imports
import numpy as np
import matplotlib.pyplot as plt
from helpers import *
# %%
# Load the data
#data_path = '/home/silja/ML_EPFL/ML_Project_1/data'
data_path = '../data'
x_train, x_test, y_train, train_ids, test_ids = load_csv_data(data_path)
# %%
print("Shape x_train: ", x_train.shape)
print("Shape x_test: ", x_test.shape)
print("Shape y_train: ", y_train.shape)
print("Train_ids: ", train_ids)
print("Test_ids: ", test_ids)
# %%
