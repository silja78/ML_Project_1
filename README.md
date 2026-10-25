# Machine Learning Project 1 (CS-433 EPFL)

## Description
In this repository you can find our code for the Machine Learning Project 1 of the CS-433 course at EPFL in 2026.

### Part 1: Machine Learning Methods Implementations
In implementations.py we implemented the machine learning methods of the course:
- Linear regression using gradient descent <br>
**mean squared error gd(y, tx, initial w, max iters, gamma)**
- Linear regression using stochastic gradient descent <br>
**mean squared error sgd(y, tx, initial w, max iters, gamma)**
- Least squares regression using normal equations <br>
**least squares(y, tx)**
- Ridge regression using normal equations <br>
**ridge regression(y, tx, lambda )**
- Logistic regression using gradient descent (y ∈ {0, 1}) <br>
**logistic regression(y, tx, initial w, max iters, gamma)**
- Regularized logistic regression using gradient descent (y ∈ {0, 1}, with regularization term λ∥w∥2) <br>
**reg logistic regression(y, tx, lambda , initial w, max iters, gamma)**
 
 All functions return the learned weights and the loss.

### Part 2: Generating Predictions for Medical Dataset
For estimating the likelihood of developing coronary heart disease (MICHD) given the dataset of the 2015 BRFSS, we used the logistic regression method that we implemented in implementations.py, because this is a binary classification task and only (regularized) logisitc regression of our method implementations is meant for this use case.

Information about the dataset can be found here (https://www.cdc.gov/brfss/annual_data/annual_2015.html).

All the steps of our pipeline can be found in this repository.

**Data preprocessing**
- We first selected only the features that are relevant for the predicition and loaded the data.
- We then cleaned the data by removing missing and refused answers, as well as the answer "I don't know" using the Codebook (https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf).
- We converted our prediction values -1,1 to 0,1 to use the implementations of the course.
- Then, we splitted the provided train datasets into a train (80%) and validation part (20%).
- After that, we calculated median, minimum and maximum values for numerical features and evaluated the categories of categorical features of the train part.
- With the derived values from the train part in the step before, we transformed our train and validation part. We transformed our numerical data by using the median of the feature for NaN and did min-max scaling. We transformed our categorical data by using one-hot vector encoding.
- We added a bias column to both train and validation part.

**Best paramter search**
- For different combinations of learning rates and maximum iteration numbers, we evaluated the best threshold for the logistic regression, because the data is unbalanced, there are more people without a coronary heart disease than with it.
- We looked at the training loss, validation loss, F1-score, accuracy, precision and recall of the model for all the tested combinations of learning rates and maximum iteration numbers and manually chose the best parameter that reduce the losses, but also show high F1-score and accuracy.

**Final model**
- With these best parameters, we then trained our final model. For this, we did the preprocessing steps again (this time for the whole train dataset).
- Finally, we saved the estimations of our model as a csv file.

All data preprocessing fucntions can be found in Data_preprocessing.py, all training functions can be found in implementations.py and ml_utils.py. Loading the whole dataset and writing the submission csv file can be done with functions from helpers.py. To run the whole pipeline execute run.py.

In feature_selection.csv we marked with 1, the columns of the dataset that we want to include and with 0 the ones that we want to exclude. We also wrote down the missing/refused/I Don't know Codebook codes to later delete those values and marked whether the column is categorical or numerical for the columns we wanted to include. All this information in feature_selection.csv gets imported in the code to perform the data preprocessing.

## Installations
To add

## Usage
To reproduce our results:
- Make a folder in the folder of the project and call it "data".
- Download the data into this folder.
- Execute run.py to get the estimates of our final model.
