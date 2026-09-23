# 🏠 Ames Housing Price Prediction

An end-to-end machine learning project for predicting residential house sale prices using the Ames Housing dataset.

## 🚀 Live Demo

**[Try the Streamlit App](https://house-price-prediction-hmuqws3kjgfyqkjdwsfy3s.streamlit.app/)**

The interactive application allows users to enter housing characteristics and generate a predicted sale price using the trained machine-learning pipeline.

---

## 📌 Project Overview

This project develops a reproducible machine-learning workflow for predicting residential sale prices from the Ames Housing dataset.

The workflow covers:

* Data cleaning and validation
* Exploratory Data Analysis (EDA)
* Feature engineering
* Data preprocessing
* Model training and evaluation
* Model comparison
* Export of reusable prediction artifacts
* Interactive deployment with Streamlit

The goal is to transform raw housing records into a reliable modeling dataset and deploy the resulting model for interactive predictions.

---

## 🔎 Problems Solved

| Problem                           | Solution                                                          | Outcome                         |
| --------------------------------- | ----------------------------------------------------------------- | ------------------------------- |
| Inconsistent raw data             | Missing-value handling, category checks, and data-type validation | Cleaned modeling dataset        |
| High-dimensional housing features | Numeric/categorical preprocessing and feature engineering         | Model-ready feature set         |
| Skewed sale-price target          | Applied `log1p(SalePrice)` transformation                         | More stable target for modeling |
| Reproducible predictions          | Saved preprocessing and trained model artifacts                   | Reusable inference pipeline     |
| Model accessibility               | Built an interactive Streamlit application                        | Public prediction demo          |

---

## 🧠 Machine Learning Workflow

```text
Raw Ames Housing Data
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Train / Test Split
        ↓
Preprocessing
        ↓
Model Training & Comparison
        ↓
Final Model
        ↓
Saved Prediction Pipeline
        ↓
Streamlit Application
```

---

## 📊 Modeling

### Experimental Setup

* **Validation:** 80/20 train-validation split
* **Cross-validation:** 5-fold CV
* **Random state:** `42`
* **Primary metric:** RMSE on `log1p(SalePrice)`
* **Features:** 81 before preprocessing
* **Features after preprocessing:** 219

### Results

* **Baseline RMSE:** `0.4016`
* **Best recorded RMSE:** `0.1141`

The final modeling workflow evaluates different approaches using the same validation strategy to make comparisons more consistent.

---

## 💡 Key Learnings

* Data-cleaning decisions can have a substantial effect on model performance.
* A log-transformed target can reduce the influence of unusually expensive properties.
* Feature engineering should be evaluated using a consistent validation strategy.
* Saving the preprocessing pipeline alongside the model is important for reliable inference.
* A model is more useful when its preprocessing and prediction steps can be reproduced outside the training notebook.

---

## 📁 Project Outputs

Important reusable artifacts include:

* `final_model.joblib` — trained prediction model
* `fitted_preprocessing.joblib` — fitted preprocessing pipeline
* `AmesHousing_cleaned.csv` — cleaned dataset
* `X_train_final.csv` — final training features
* `X_test_final.csv` — final test features
* `y_train_log.csv` — log-transformed training target
* `y_test_log.csv` — log-transformed test target

---

## 🖥️ Streamlit Application

The project includes a deployed Streamlit application that provides an interactive interface for making house-price predictions.

### Live Application

**[Open the Ames Housing Price Predictor →](https://house-price-prediction-hmuqws3kjgfyqkjdwsfy3s.streamlit.app/)**

---

## 🛠️ Tech Stack

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Seaborn**
* **Joblib**
* **Streamlit**
* **Jupyter Notebook**

---

## ▶️ Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/bode987-s/<repository-name>.git
cd <repository-name>
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

**Windows:**

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit application

```bash
streamlit run app.py
```

---

## 📓 Reproducing the Analysis

The original workflow is organized into notebooks covering:

1. **Data Wrangling** — cleaning and validating the source data
2. **EDA** — distributions, relationships, and data-quality analysis
3. **Feature Engineering** — creating and selecting model features
4. **Modeling** — training, comparison, and evaluation

The saved model and preprocessing artifacts can then be used for inferenc
