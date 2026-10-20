# %%
# Imports
import csv
import numpy as np
import matplotlib.pyplot as plt
from helpers import *
# %%
# Load the data
data_path = 'data'
x_train, x_test, y_train, train_ids, test_ids = load_csv_data(data_path)
# %%
# First look at the data
print("Shape x_train: ", x_train.shape)
print("Shape x_test: ", x_test.shape)
print("Shape y_train: ", y_train.shape)
print("Train_ids: ", train_ids)
print("Test_ids: ", test_ids)
print("Predictions:", np.unique(y_train, return_counts=True))
print("NaN in x_train:", np.isnan(x_train).sum())
print("NaN in x_test:", np.isnan(x_test).sum())
# %%
# Feature selection
keep_mask = np.genfromtxt(
    os.path.join(data_path, "feature_selection.csv"),
    delimiter=";",
    dtype=int,
    usecols=1,
)
# %%
# Delete ID column
keep_mask = keep_mask[1:]
# %%
print("Feature mask shape: ", keep_mask.shape)
x_train = x_train[:, keep_mask == 1]
x_test = x_test[:, keep_mask == 1]
print("Shape x_train: ", x_train.shape)
print("Shape x_test: ", x_test.shape)
# %%
# Variable names (exclude ID)
with open(os.path.join(data_path, "x_train.csv"),
          newline="", encoding="utf-8") as file:
    header = next(csv.reader(file))
feature_names = header[1:]
selected_names = [
    name for name, keep in zip(feature_names, keep_mask)
    if keep == 1
]
# %%
# Quality data
for i, name in enumerate(selected_names):
    col = x_train[:, i]
    missing_count = np.isnan(col).sum()
    missing_percent = np.isnan(col).mean() * 100
    valid = col[~np.isnan(col)]
    unique_count = np.unique(valid).size

    print(
        f"{name:30s} "
        f"missing={missing_count} ({missing_percent:.1f}%) "
        f"unique_values={unique_count}"
    )
# %%
# Set missing, refused, etc. to NaN
missing__refused_codes = {
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
for col_index, codes in missing__refused_codes.items():
    x_train[np.isin(x_train[:, col_index], codes), col_index] = np.nan
    x_test[np.isin(x_test[:, col_index], codes), col_index] = np.nan

for i, name in enumerate(selected_names):
    col = x_train[:, i]
    missing_count = np.isnan(col).sum()
    missing_percent = np.isnan(col).mean() * 100
    valid = col[~np.isnan(col)]
    unique_count = np.unique(valid).size

    print(
        f"{name:30s} "
        f"missing={missing_count} ({missing_percent:.1f}%) "
        f"unique_values={unique_count}"
    )
# %%
# Preprocess categorical and numerical data
def fit_preprocessor(x, column_types):
    """Learn categories and numeric preprocessing values from training data."""
    categories = []
    medians = []
    minima = []
    maxima = []

    for j, column_type in enumerate(column_types):
        column = x[:, j]
        observed = column[~np.isnan(column)]

        if column_type == "categorical":
            categories.append(np.unique(observed))
            medians.append(None)
            minima.append(None)
            maxima.append(None)

        elif column_type == "numeric":
            categories.append(None)
            medians.append(np.median(observed))
            minima.append(np.min(observed))
            maxima.append(np.max(observed))

    return categories, medians, minima, maxima


def transform_data(x, column_types, preprocessing_values):
    """Encode categorical columns and impute/scale numeric columns."""
    categories, medians, minima, maxima = preprocessing_values
    output_columns = []

    for j, column_type in enumerate(column_types):
        column = x[:, j]

        if column_type == "categorical":
            known_categories = categories[j]
            encoded = np.zeros(
                (x.shape[0], len(known_categories) + 1)
            )

            for k, category in enumerate(known_categories):
                encoded[:, k] = (column == category)

            # Last column indicates a missing value.
            encoded[:, -1] = np.isnan(column)
            output_columns.append(encoded)

        else:  # numeric
            column[np.isnan(column)] = medians[j]

            minimum = minima[j]
            maximum = maxima[j]

            if maximum == minimum:
                scaled = np.zeros(column.size)
            else:
                scaled = (column - minimum) / (maximum - minimum)

            output_columns.append(scaled.reshape(-1, 1))

    return np.concatenate(output_columns, axis=1)


# All our data at the moment is categorical, adjust here if needed
column_types = ["categorical"] * x_train.shape[1]

# Learn settings from the training data, then use the same settings for both sets
preprocessing_values = fit_preprocessor(x_train, column_types)

x_train_preproc = transform_data(
    x_train, column_types, preprocessing_values
)
x_test_preproc = transform_data(
    x_test, column_types, preprocessing_values
)
# %%
