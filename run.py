#%%
# Imports
import csv
import numpy as np
import matplotlib.pyplot as plt
from ml_utils import *
from helpers import *
from Data_preprocessing import *
from implementations import *

#%%
# Load the data for selected features
data_path = 'data'
x_train, x_test, y_train, train_ids, test_ids, selected_names = load_and_select_data(data_path)

#%%
# Overview coding for missing or refused values of selected features
missing_refused_codes = find_missing_codes(selected_names, data_path)

# Replace missing values by NaN
x_train, x_test = replace_missing_codes(
    x_train, x_test, missing_refused_codes
    )

#%%
# Labels: {-1, 1} -> {0, 1}
y_train_01 = (y_train == 1).astype(float)

#%%
# Split Training into Training and Validation parts
train_idx, val_idx = stratified_split(y_train_01, val_ratio=0.2, seed=1)
x_train_part, y_train_part_01 = x_train[train_idx], y_train_01[train_idx]
x_val_part, y_val_part_01 = x_train[val_idx], y_train_01[val_idx]

print("Positive rate train:", y_train_part_01.mean(), "val:", y_val_part_01.mean())

#%%
# All our data at the moment is categorical, adjust here if needed
column_types = ["categorical"] * x_train.shape[1]

# Learn preprocessing settings from the training data, then use the same settings for training and validation
preprocessing_values = fit_preprocessor(x_train_part, column_types)

# Preprocessing
x_train_part_preproc = transform_data(
    x_train_part, column_types, preprocessing_values
)
x_val_part_preproc = transform_data(
    x_val_part, column_types, preprocessing_values
)

# Add a bias column of ones
tx_train_part = np.c_[np.ones(x_train_part_preproc.shape[0]), x_train_part_preproc]
tx_val_part = np.c_[np.ones(x_val_part_preproc.shape[0]), x_val_part_preproc]

#%%
#Training and choice of the parameters for logistic regression
gammas = np.logspace(-5, 0, 9)  # 0.0001, 0.001, 0.01, 0.1, 1.0
iters_values = [100, 300, 500, 1000, 2000]
thresholds = np.arange(0.05, 0.96, 0.01)
gamma_best, max_iters_best, thr_best, results = find_best_logistic_params(y_train_part_01, tx_train_part, y_val_part_01, tx_val_part, gammas, iters_values, thresholds)

#%%
metric_names = [
    "Threshold",
    "Train loss",
    "Validation loss",
    "F1",
    "Accuracy",
    "Precision",
    "Recall",
]
for k, name in enumerate(metric_names):
    print(f"\n{name}")
    print("gamma values:", gammas)
    print("iterations:  ", iters_values)
    print(np.round(results[:, :, k], 4))
#%%
#Final model

# Learn preprocessing settings from the training data, then use the same settings for training and testing
preprocessing_values = fit_preprocessor(x_train, column_types)

# Preprocessing
x_train_preproc = transform_data(
    x_train, column_types, preprocessing_values
)
x_test_preproc = transform_data(
    x_test, column_types, preprocessing_values
)

# Add a bias column of ones
tx_train = np.c_[np.ones(x_train_preproc.shape[0]), x_train_preproc]
tx_test = np.c_[np.ones(x_test_preproc.shape[0]), x_test_preproc]

# Retrain on all the training data
w_final, _ = logistic_regression(
    y_train_01, tx_train, np.zeros(tx_train.shape[1]), max_iters_best, gamma_best
)

probs_test = sigmoid(tx_test @ w_final)
y_pred = np.where(probs_test >= thr_best, 1, -1)
print("Fraction predicted positive:", np.mean(y_pred == 1))

#%%
create_csv_submission(test_ids, y_pred, "submission.csv")