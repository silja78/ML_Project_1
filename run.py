#%%
# Imports
import numpy as np
import matplotlib.pyplot as plt
from ml_utils import *
from helpers import *
from Data_preprocessing import *
from implementations import *


#Number of the lines to modify depending on the model we want : 17 (feature_selection_file_name), 70 to 72 (if we want to train the model of another grid), 95 (name_file), 104-106 (adapting the parameter to run the final model), 134 (creating the csv file for the submission with the result of the prediction for each patient)
#%%
# Load the data for selected features
data_path = 'data'
feature_path = ''
#comment or uncomment the line depending on the model you want
feature_selection_file_name = "feature_selection_complex_model.csv"
#feature_selection_file_name = "feature_selection_simple_model.csv"

x_train, x_test, y_train, train_ids, test_ids, selected_names = load_and_select_data(data_path, feature_path,feature_selection_file_name)
# Change here the name of the file depending of the model you want to use



#%%
# Overview coding for missing or refused values of selected features
missing_refused_codes = find_missing_codes(selected_names, feature_path,feature_selection_file_name)

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
#Preprocessing steps
# Coding for numerical and categorical in feature_selection.csv
column_types = get_column_types(selected_names, feature_path,feature_selection_file_name)
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
#to find the best stable gamma
n = tx_train_part.shape[0]
L = 0.25 * np.linalg.eigvalsh(tx_train_part.T @ tx_train_part / n)[-1]
print("colonnes :", tx_train_part.shape[1], " L =", L, " 1/L =", 1 / L)
#%%
#Training and choice of the parameters for logistic regression
gammas = np.logspace(-1, 0, 10)  
iters_values = [2000, 5000, 10000]
thresholds = np.arange(0.05, 0.96, 0.01)
gamma_best, max_iters_best, thr_best, results = find_best_logistic_params(y_train_part_01, tx_train_part, y_val_part_01, tx_val_part, gammas, iters_values, thresholds)

#%%
#Create the csv file with all the parameters used in this model
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
    
    
#Change the name of the file to avoid erasing the previous one
#comment or uncomment the line depending on the model you want
name_file = 'result_complex_model.csv'
#name_file = 'result_simple_model.csv'
save_results_csv(results, gammas, iters_values, name_file)
#%%
#Final model parameters ? (if we validate it)
print('gamma_best :',gamma_best, ' \n max_iters_best :',max_iters_best, ' \n thr_best :', thr_best)
#%%
#Preprocessing and setting of the manual parameters if needed
#change the value here if the model take values with bad loss
gamma_best=0.27
max_iters_best=10000
thr_best=0.21
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
w_final, loss = logistic_regression(
    y_train_01, tx_train, np.zeros(tx_train.shape[1]), max_iters_best, gamma_best
)

probs_test = sigmoid(tx_test @ w_final)
y_pred = np.where(probs_test >= thr_best, 1, -1)
print("Fraction predicted positive:", np.mean(y_pred == 1))
print("Final training loss (all training data):", loss)

#%%
#Create csv file with the different classes (prediction of : the patient has coronary disease or not)
create_csv_submission(test_ids, y_pred, "submission_complex_model.csv")

#%%
# Useless baseline: predict "no heart attack" (-1) for everyone

# One prediction per test sample, all equal to -1
y_pred_baseline = predict_all_negative(len(test_ids))

# Use a different file name so the real submission.csv is not overwritten
create_csv_submission(test_ids, y_pred_baseline, "submission_baseline.csv")
# %%
