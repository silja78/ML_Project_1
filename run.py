#%%
# Imports
import csv
import numpy as np
import matplotlib.pyplot as plt
from ml_utils import stratified_split, f1_score, best_threshold
from helpers import *
from Data_preprocessing import *
from implementations import *

#%%
# Load the data for selected features
data_path = 'data'
x_train, x_test, y_train, train_ids, test_ids, selected_names = load_and_select_data(data_path)

#%%
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

#%%
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
# Labels: {-1, 1} -> {0, 1}
y01 = (y_train == 1).astype(float)

# Add a bias column of ones
tx_all = np.c_[np.ones(x_train_preproc.shape[0]), x_train_preproc]
tx_test = np.c_[np.ones(x_test_preproc.shape[0]), x_test_preproc]

#%%
#Validation dataset
train_idx, val_idx = stratified_split(y01, val_ratio=0.2, seed=1)
tx_tr, y_tr = tx_all[train_idx], y01[train_idx]
tx_val, y_val = tx_all[val_idx], y01[val_idx]

print("Positive rate train:", y_tr.mean(), "val:", y_val.mean())

#%%
#Training and choice of the parameters
'''max_iters = 2000
for gamma in [0.1, 0.5, 1.0]:
    w, loss = logistic_regression(
        y_tr, tx_tr, np.zeros(tx_tr.shape[1]), max_iters, gamma
    )
    probs_val = sigmoid(tx_val @ w)
    thr, f1 = best_threshold(y_val, probs_val, np.arange(0.05, 0.55, 0.01))
    print(f"gamma={gamma}  loss={loss:.4f}  best thr={thr:.2f}  F1={f1:.4f}")
    '''
for max_iters in [500, 2000, 5000]:
    w, loss = logistic_regression(
        y_tr, tx_tr, np.zeros(tx_tr.shape[1]), max_iters, 1.0
    )
    probs_val = sigmoid(tx_val @ w)
    thr, f1 = best_threshold(y_val, probs_val, np.arange(0.05, 0.55, 0.01))
    print(f"max_iters={max_iters}  loss={loss:.4f}  thr={thr:.2f}  F1={f1:.4f}")
#%%

#%%
#Final model
gamma_best = 1   # set to the best value found above
thr_best = 0.18     # set to the best threshold found above

# Retrain on all the training data
w_final, _ = logistic_regression(
    y01, tx_all, np.zeros(tx_all.shape[1]), max_iters, gamma_best
)

probs_test = sigmoid(tx_test @ w_final)
y_pred = np.where(probs_test >= thr_best, 1, -1)
print("Fraction predicted positive:", np.mean(y_pred == 1))

#%%
create_csv_submission(test_ids, y_pred, "submission.csv")