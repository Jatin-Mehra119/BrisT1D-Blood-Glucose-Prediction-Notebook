# Blood Glucose Prediction Model Using LightGBM

## Table of Contents

1.  [Introduction](#introduction)
2.  [Dataset Description](#dataset-description)
3.  [Code Overview](#code-overview)
4.  [Results](#results)
5.  [Acknowledgements](#acknowledgements)

----------

## Introduction

This project builds a machine learning model to predict blood glucose (BG) levels one hour into the future using time-series data from continuous glucose monitors (CGM), insulin pumps, and smartwatches. The dataset includes multiple time-series features such as past BG readings, insulin doses, carbohydrate intake, and activity levels.

The model employs **LightGBM**, a gradient boosting framework, leveraging GPU acceleration for efficient training.

----------

## Dataset Description

The dataset comes from a study on type 1 diabetes. It includes data for:

-   **Training**: Collected over three months from nine participants.
-   **Testing**: Includes data from unseen participants over a different time period.

### Data Files

-   **`train.csv`**: Training data (chronological and overlapping).
-   **`test.csv`**: Testing data (randomized and non-overlapping).
-   **`sample_submission.csv`**: Example submission format.
-   **`activities.txt`**: Description of activities.

### Columns Overview

1.  **Participant Metadata**: `id`, `p_num`, `time`.
2.  **Features**:
    -   `bg-X:XX`: Blood glucose readings from the past 6 hours.
    -   `insulin-X:XX`: Insulin doses.
    -   `carbs-X:XX`: Carbohydrates consumed.
    -   `hr-X:XX`: Heart rate.
    -   `steps-X:XX`: Steps taken.
    -   `cals-X:XX`: Calories burned.
    -   `activity-X:XX`: Activities performed.
3.  **Target**: `bg+1:00` (future blood glucose level to predict).

----------

## Code Overview

### Key Steps

1.  **Data Preprocessing**:
    
    -   Handle missing values using interpolation and imputation.
    -   Add time-based features (`sin_hour`, `cos_hour`) to capture cyclical patterns.
    -   Normalize features using `StandardScaler`.
2.  **Feature Engineering**:
    
    -   Construct 12 intervals of features for BG, insulin, carbs, HR, steps, and calories.
    -   Aggregate features into groups for efficient processing.
3.  **Model Training**:
    
    -   Split data into training and validation sets.
    -   Train a LightGBM model with hyperparameter tuning using **GridSearchCV**.
    -   Employ early stopping to prevent overfitting.
4.  **Evaluation**:
    
    -   Compute metrics: **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)**.
    -   Fine-tune the model based on cross-validation results.
5.  **Prediction**:
    
    -   Generate predictions for the test set.
    -   Save results to `Submission_LGB_Fine-tuning.csv`.

----------


----------

## Results

### Evaluation Metrics

-   **Training RMSE**: 1.3431
-   **Validation RMSE**: 1.7371
-   **Validation MAE**: 1.2802

### Model Performance

The LightGBM model performs well in predicting future BG levels, leveraging GPU acceleration for efficient training on high-dimensional time-series data.

----------

## Acknowledgements

The dataset and challenge were provided by a medical research study on type 1 diabetes. Special thanks to the researchers and participants for making this data available for machine learning applications.
