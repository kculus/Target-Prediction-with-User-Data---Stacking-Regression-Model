# Target-Prediction-with-User-Data---Stacking-Regression-Model

This project uses a stacking regression model to predict a target variable based on user features. The model combines Linear Regression and CatBoost, saving prediction results to a `submission.csv` file.

## Contents
1. [Project Overview](#project-overview)
2. [Data Preprocessing](#data-preprocessing)
3. [Feature Engineering](#feature-engineering)
4. [Model Training and Evaluation](#model-training-and-evaluation)
5. [Usage](#usage)
6. [Requirements](#requirements)

## Project Overview

The purpose of this project is to predict a target variable from user data. The project preprocesses the data, fills missing values, applies feature engineering, scales numerical data, and then uses stacking regression to make predictions.

## Data Preprocessing

This project uses the following data files:

- `targets_train.csv`: Target values in the training data.
- `user_features_train.csv`, `user_features_test.csv`: User features for training and testing.
- `users_train.csv`, `users_test.csv`: Additional user data for training and testing.

Data preprocessing steps:
1. Training and test data are merged by the `ID` key.
2. Missing values in numerical columns are filled with the mean, and missing values in categorical columns are filled with "Unknown."
3. Unused columns (`first_open_date`, `ID`, `country`, `platform`, `device_category`, `device_brand`, `device_model`, `ad_network`) are removed.

## Feature Engineering

Additional features created include:

- `Total_Level_Duration`: Total time spent across levels.
- `Avg_Level_Duration`: Average time spent across levels.

These features help to better represent user behavior at each level.

## Model Training and Evaluation

Two models are combined in a stacked regression configuration:

1. **Linear Regression**: A simple linear regression model.
2. **CatBoost Regressor**: A gradient boosting model suitable for categorical data.

These models are combined using `StackingRegressor`, with a Linear Regression model as the final estimator. The model's performance is evaluated using the Mean Squared Error (MSE) metric on a validation set.

When the code is run, the validation error is printed to the console:

```plaintext
Validation MSE: <calculated_mse>
