# Imports
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from helpers import *

# Load the data for the selected features
def load_and_select_data(data_path):
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(data_path)
    
    keep_mask = np.genfromtxt(
        os.path.join(data_path, "feature_selection.csv"),
        delimiter=";",
        dtype=int,
        usecols=1,
    )

    # Delete ID column
    keep_mask = keep_mask[1:]

    x_train = x_train[:, keep_mask == 1]
    x_test = x_test[:, keep_mask == 1]

    # Variable names (exclude ID)
    with open(os.path.join(data_path, "x_train.csv"),
            newline="", encoding="utf-8") as file:
        header = next(csv.reader(file))
    feature_names = header[1:]
    selected_names = [
        name for name, keep in zip(feature_names, keep_mask)
        if keep == 1
    ]

    return x_train, x_test, y_train, train_ids, test_ids, selected_names

def replace_missing_codes(x_train, x_test, missing_codes):
    for col_index, codes in missing_codes.items():
        x_train[np.isin(x_train[:, col_index], codes), col_index] = np.nan
        x_test[np.isin(x_test[:, col_index], codes), col_index] = np.nan

    return x_train, x_test

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