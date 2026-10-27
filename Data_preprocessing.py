
# Imports
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from helpers import *



# Load the data for the selected features
def load_and_select_data(data_path, feature_path, feature_selection_file_name):
    """load the data and keep only the features selected in the selection file.

    Args:
        data_path: str, path to the folder containing x_train.csv, x_test.csv
            and y_train.csv
        feature_path: str, path to the folder containing the feature selection file
        feature_selection_file_name: file that contains the features we have selected to train our model
    Returns:
        x_train: shape=(N_train, D), the selected training features
        x_test:  shape=(N_test, D), the selected test features
        y_train: shape=(N_train,), the training labels
        train_ids: shape=(N_train,), the training sample ids
        test_ids:  shape=(N_test,), the test sample ids
        selected_names: list of D strings, the names of the selected features
    """
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(data_path)

    keep_mask = np.genfromtxt(
        os.path.join(feature_path, feature_selection_file_name),
        delimiter=";",
        dtype=int,
        usecols=1,
    )

    # Delete ID column
    keep_mask = keep_mask[1:]

    x_train = x_train[:, keep_mask == 1]
    x_test = x_test[:, keep_mask == 1]

    # Variable names (exclude ID)
    with open(
        os.path.join(data_path, "x_train.csv"),
        newline="",
        encoding="utf-8",
    ) as file:
        header = next(csv.reader(file))

    feature_names = header[1:]
    selected_names = [
        name for name, keep in zip(feature_names, keep_mask)
        if keep == 1
    ]

    return x_train, x_test, y_train, train_ids, test_ids, selected_names


def find_missing_codes(selected_names, feature_path,feature_selection_file_name):
    """find the answer codes meaning "missing" or "refused" for each selected feature.

    Args:
        selected_names: list of D strings, the names of the selected features
        feature_path: str, path to the folder containing the feature selection file

    Returns:
        a dictionary {column index: list of int codes}, only for the features
        that have at least one such code
    """
    missing_refused_codes = {}

    selected_indices = {
        name: index for index, name in enumerate(selected_names)
    }

    with open(
        os.path.join(feature_path, feature_selection_file_name),
        newline="",
        encoding="utf-8",
    ) as file:
        variable_rows = csv.reader(file, delimiter=";")

        for variable in variable_rows:
            feature_name = variable[0].strip()

            if feature_name not in selected_indices:
                continue

            cell = variable[3].strip()
            if not cell:
                continue

            codes = [
                int(code.strip())
                for code in cell.split(",")
                if code.strip()
            ]

            index = selected_indices[feature_name]
            missing_refused_codes[index] = codes

    return missing_refused_codes


def replace_missing_codes(x_train, x_test, missing_codes):
    """replace the "missing" or "refused" codes by NaN, in place.

    Args:
        x_train: shape=(N_train, D), the training features
        x_test: shape=(N_test, D), the test features
        missing_codes: dictionary {column index: list of int codes}

    Returns:
        x_train: shape=(N_train, D), with NaN instead of the codes
        x_test: shape=(N_test, D), with NaN instead of the codes
    """
    for col_index, codes in missing_codes.items():
        x_train[
            np.isin(x_train[:, col_index], codes), col_index
        ] = np.nan

        x_test[
            np.isin(x_test[:, col_index], codes), col_index
        ] = np.nan

    return x_train, x_test


def get_column_types(selected_names, feature_path,feature_selection_file_name):
    """get the type of each selected feature from the feature selection file.

    Args:
        selected_names: list of D strings, the names of the selected features
        feature_path: str, path to the folder containing the feature selection file

    Returns:
        a list of D strings, each one either "categorical" or "numeric",
        in the same order as selected_names
    """
    type_by_name = {}

    with open(
        os.path.join(feature_path, feature_selection_file_name),
        newline="",
        encoding="utf-8",
    ) as file:
        variable_rows = csv.reader(file, delimiter=";")

        for variable in variable_rows:
            feature_name = variable[0].strip()

            if feature_name in selected_names:
                code = variable[4].strip().lower()

                if code == "c":
                    type_by_name[feature_name] = "categorical"
                elif code == "n":
                    type_by_name[feature_name] = "numeric"

    return [type_by_name[name] for name in selected_names]


# Preprocess categorical and numerical data
def fit_preprocessor(x, column_types):
    """learn the categories and the numeric preprocessing values from the training data.

    Args:
        x: shape=(N, D), the raw training features, with NaN for missing values
        column_types: list of D strings, each one either "categorical" or "numeric"

    Returns:
        categories: list of D entries, the sorted unique observed values for
            each categorical column, None for numeric columns
        medians: list of D entries, the median of each numeric column
            (used to replace NaN), None for categorical columns
        minima: list of D entries, the minimum of each numeric column,
            None for categorical columns
        maxima: list of D entries, the maximum of each numeric column,
            None for categorical columns
    """
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
    """encode the categorical columns and impute and scale the numeric columns.

    Categorical columns are one-hot encoded, with one extra column indicating a
    missing value. Numeric columns have their NaN replaced by the median and are
    scaled to [0, 1] with min-max scaling.

    Args:
        x: shape=(N, D), the raw features, with NaN for missing values
        column_types: list of D strings, each one either "categorical" or "numeric"
        preprocessing_values: tuple (categories, medians, minima, maxima),
            as returned by fit_preprocessor

    Returns:
        a matrix of shape (N, D'), the preprocessed features, where D' is the
        number of columns after the one-hot encoding
    """
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
