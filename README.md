# Ames Housing Price Prediction

A machine learning project that explores the Ames housing dataset and builds a reproducible workflow for predicting house sale prices.

## Overview

The goal of this project is to predict `SalePrice` using information about residential properties in Ames, Iowa.

Rather than jumping straight into modeling, the project follows a structured data science workflow: first understanding and validating the data, then exploring patterns, engineering useful features, and finally training and evaluating multiple models.

A key focus throughout the project is **avoiding data leakage**. Statistics and transformations are learned from the training data only and then applied to the test data.

## Dataset

The project uses the Ames housing dataset.

The target variable is:

* `SalePrice` — the final sale price of the property.

The workflow examines features such as:

* `OverallQual` — overall material and finish quality
* `Gr Liv Area` — above-ground living area
* `Garage Area` — garage size
* `Total Bsmt SF` — basement area
* `Year Built` — original construction year
* `Neighborhood`
* `Kitchen Qual`
* `Garage Type` and `Garage Finish`

The dataset source is not specified in the workflow.

## Workflow

The project is organized into four main notebooks.

### 1. Data Understanding & Wrangling

`01_data_understanding_and_wrangling.ipynb`

The first stage focuses on making sure the data is reliable before analyzing it.

This includes:

* Understanding the dataset structure and data types
* Checking for duplicate IDs and invalid values
* Handling obvious structural missing values
* Standardizing column formats
* Splitting the data into training and test sets

The train/test split happens before statistics are calculated, helping prevent information from the test set from influencing later steps.

### 2. Exploratory Data Analysis

`02_eda.ipynb`

The training data is explored without modifying it.

The analysis includes distributions, categorical variables, correlations, scatter plots, pairplots, outlier investigation, and skewness analysis.

Several relationships with `SalePrice` are investigated, particularly variables such as `OverallQual`, `Gr Liv Area`, `Garage Area`, `Total Bsmt SF`, and `Year Built`.

Statistical tests are also used to support observations from the visual analysis. For example, when the equal-variance assumption was not satisfied across neighborhoods, the workflow used the Kruskal–Wallis test instead of ANOVA.

The purpose of this stage is not to make final claims, but to identify patterns that can guide feature engineering and modeling.

### 3. Feature Engineering

`03_feature_engineering.ipynb`

The next stage prepares the data for machine learning.

It includes:

* Statistical missing-value imputation
* Outlier treatment
* Skewness transformations
* Log transformation of `SalePrice`
* Creation of derived features
* Encoding categorical variables
* Multicollinearity treatment
* Feature scaling where appropriate
* Feature selection
* Optional PCA experimentation

A `ColumnTransformer` is used to combine preprocessing steps into a reusable pipeline. The pipeline is fitted on the training data and saved for use during modeling.

### 4. Modeling & Evaluation

`04_modeling_and_evaluation.ipynb`

Several models are trained and compared, starting with a linear regression baseline.

The workflow considers:

* Linear Regression
* Ridge
* Lasso
* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost *(optional)*

Hyperparameters can be tuned using `GridSearchCV` or `RandomizedSearchCV`.

Models are compared using metrics such as **RMSE** and **R²**, with additional evaluation using MAE on the final model.

The analysis also goes beyond a single score by examining residuals, prediction errors, and performance across groups such as neighborhoods, price bands, and property ages.

For the best tree-based model, explainability techniques such as SHAP, Partial Dependence Plots, and Permutation Importance are used to better understand the model's behavior.

## Key Observations

The exploratory workflow indicates that several property characteristics have meaningful relationships with `SalePrice`.

In particular, `OverallQual` shows a strong positive relationship with sale price, while variables related to living area, basement space, garage space, and property age also receive attention during the analysis.

The relationship between features and price is not always perfectly linear. Some variables show nonlinear patterns, variation within groups, and potential outliers.

These are **observations from the exploratory stage**, rather than claims of causation. Their role is to guide the later feature engineering and modeling stages.

## Technologies & Libraries

The workflow uses the Python data science ecosystem, including:

* **Python** — primary programming language
* **Pandas / NumPy** — data manipulation and numerical operations
* **Matplotlib / Seaborn** — visualization and exploratory analysis
* **Scikit-learn** — preprocessing, statistical utilities, modeling, pipelines, and hyperparameter tuning
* **XGBoost** — optional gradient-boosting model
* **SHAP** — model explainability
* **Joblib** — persistence of the preprocessing pipeline
* **Jupyter Notebook** — development and analysis environment

## Project Structure

```text
.
├── 01_data_understanding_and_wrangling.ipynb
├── 02_eda.ipynb
├── 03_feature_engineering.ipynb
├── 04_modeling_and_evaluation.ipynb
├── train.csv
├── test.csv
└── README.md
```

`train.csv` and `test.csv` are generated after the initial data-splitting stage and reused by the following notebooks.

## How to Run

The workflow is designed to be executed in order:

```text
01 → 02 → 03 → 04
```

1. Run `01_data_understanding_and_wrangling.ipynb` to validate and split the data.
2. Run `02_eda.ipynb` to explore the training data.
3. Run `03_feature_engineering.ipynb` to build the preprocessing pipeline.
4. Run `04_modeling_and_evaluation.ipynb` to train, tune, compare, and evaluate the models.

The exact dataset download/setup instructions are not specified in the current workflow, so the required dataset path would need to be added before running the project from a fresh environment.

## What I Learned

One of the main lessons from this workflow is that a machine learning project is not just about choosing a model.

Understanding the data, checking assumptions, investigating relationships, handling skewness and missing values, and preventing leakage all affect how trustworthy the final results are.

The project also emphasizes keeping preprocessing reproducible: once the preprocessing pipeline is fitted, the same pipeline should be used when preparing data for the models.

## Future Improvements

Possible next steps include:

* Complete the final model comparison and evaluation
* Experiment with the optional XGBoost and PCA tracks
* Refine feature engineering based on the EDA findings
* Compare explainability results across the strongest models
* Add the final model metrics and conclusions to this README
* Add an executive-summary notebook for a more presentation-friendly version of the project
