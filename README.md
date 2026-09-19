# Ames Housing Price Prediction

## Purpose and Project Overview

This project develops a reproducible machine-learning workflow for predicting residential sale prices from the Ames Housing dataset. It covers data cleaning, exploratory analysis, feature engineering, preprocessing, model training, evaluation, and export of reusable prediction artifacts. The goal is to turn raw housing records into a reliable modeling dataset while documenting the decisions, limitations, and lessons learned along the way.

**Example:** The workflow transforms `AmesHousing.csv` into cleaned and engineered training and test data, then uses the fitted preprocessing pipeline and final model to generate price predictions.

## Problems Solved

Describing the original problems and the measurable result of solving each one:

- **Inconsistent raw data** - Identified missing values, inconsistent categories, and unsuitable data types. **Outcome:** produced a cleaned dataset in `AmesHousing_cleaned.csv`.
- **High-dimensional housing features** - Converted categorical and numeric variables into model-ready features. **Outcome:** created reusable final feature datasets in `X_train_final.csv` and `X_test_final.csv`.
- **Skewed sale-price target** - Applied a log transformation to reduce the effect of extreme prices. **Outcome:** retained transformed targets in `y_train_log.csv` and `y_test_log.csv` for modeling and comparison.
- **Reproducibility of predictions** - Saved the trained model and preprocessing components. **Outcome:** exported `final_model.joblib` and `fitted_preprocessing.joblib`.

## Issues Addressed

Track bugs, analysis questions, and feature work here. Link each item to a GitHub issue, pull request, commit, or notebook section.

| ID | Issue / Feature                        | Status | Resolution / Next Step                                                   |
| -- | -------------------------------------- | ------ | ------------------------------------------------------------------------ |
| #1 | Handle missing values by feature type  | ✅ Done | Applied documented imputation rules and verified the resulting schema.   |
| #2 | Compare raw vs. log-transformed target | ✅ Done | Compared validation metrics after reversing the log transformation.      |
| #3 | Evaluate PCA vs. full feature set      | ✅ Done | Benchmarked predictive performance, interpretability, and training cost. |


### Issue Template Example

- **Problem:** A categorical feature contains missing values that cannot be passed directly to the estimator.
- **Investigation:** Check missing-value counts, category frequency, and downstream preprocessing behavior.
- **Fix:** Add an explicit imputation or `Unknown` category strategy.
- **Verification:** Confirm no unexpected nulls remain and rerun the affected evaluation step.

## Learnings and Insights

Capture conclusions that should guide future work, not only the final score.

### Key Takeaways

- Data cleaning decisions can affect model quality as much as estimator selection.
- A log-transformed target can make price prediction less sensitive to unusually expensive homes.
- Feature engineering should be evaluated with a fixed validation strategy to avoid misleading comparisons.
- Saving preprocessing with the model is necessary for consistent inference on future records.

### Process and Metrics

Record the experiment setup so results remain comparable.

- **Validation strategy:** 80/20 split and 5-fold cross-validation using random_state=42
- **Primary metric:** RMSE on log1p(SalePrice)
- **Baseline score:** 0.4016
- **Best score:** 0.1141
- **Best model:** Ridge/Lasso/Gradient Boosting ensemble
- **Feature count:** 81 before preprocessing, 219 after
- **Data quality checks:** [null checks, duplicate checks, leakage checks, schema checks]

### Future Work

- Added cross-validation and confidence intervals to model comparisons.
- Compare predictions after converting log-scale outputs back to dollar values.
- Investigate residuals by neighborhood, house age, and sale-price range..
- Track experiments in a consistent table or experiment-management tool.



### Prerequisites

- Python 3.10 or newer
- Jupyter Notebook or JupyterLab
- A virtual environment
- Required packages listed in `requirements.txt` (to be added if this project is distributed)


### Quick Start

1. Place the source data in the project root.
2. Run `Data Wrangling .ipynb` to clean and validate the raw data.
3. Run `EDA .ipynb` to inspect distributions, relationships, and data quality.
4. Run `Feature Engineering .ipynb` to create model-ready features.
5. Run `Modeling_.ipynb` or `Modelling .ipynb` to train and evaluate models.
6. Use `final_model.joblib` with `fitted_preprocessing.joblib` for inference on compatible input data.



## Roadmap

Short-term milestones should be specific and verifiable.

- [✅ ] Add a pinned dependency file and environment setup instructions.
- [✅ ] Consolidate duplicate modeling notebooks into one documented workflow.
- [✅ ] Add automated data-quality and schema checks.
- [✅] Compare full-feature and PCA-based models using the same validation split.
- [✅ ] Record final metrics, selected model, and known limitations.
- [✅ ] Add a small prediction script with an example input row.







## License

This project is licensed under the [MIT License](LICENSE). 

## Contact

- **Maintainer:** [Abdelrahman]
- **GitHub:** [@bode987-s](https://github.com/<username>)
- **Email:** [abdelrahmanbadawy612@gmail.com]
