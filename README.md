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

| ID | Issue or feature | Status | Resolution or next step |
|---|---|---|---|---|
| #1 | Handle missing values by feature type | Done | Applied documented imputation rules and verified the resulting schema. |
| #2 | Compare raw and log-transformed target values | Done | Compare validation metrics after reversing the log transformation. |
| #3 | Evaluate PCA against the full feature set | Done | Benchmark accuracy, interpretability, and training cost. |

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

- **Validation strategy:** [for example, train/test split with a fixed random seed]
- **Primary metric:** [for example, RMSE or RMSLE]
- **Baseline score:** `[metric] = [value]`
- **Best score:** `[metric] = [value]`
- **Best model:** `[model name]`
- **Feature count:** `[count before]` before preprocessing, `[count after]` after preprocessing
- **Data quality checks:** [null checks, duplicate checks, leakage checks, schema checks]

### Future Work

- Add cross-validation and confidence intervals to model comparisons.
- Compare predictions after converting log-scale outputs back to dollar values.
- Investigate residuals by neighborhood, house age, and sale-price range.
- Add an inference script or lightweight API for new property records.
- Track experiments in a consistent table or experiment-management tool.

## How to Use or Run

### Prerequisites

- Python 3.10 or newer
- Jupyter Notebook or JupyterLab
- A virtual environment
- Required packages listed in `requirements.txt` (to be added if this project is distributed)

### Setup

```bash
git clone https://github.com/<owner>/<repository>.git
cd <repository>
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Quick Start

1. Place the source data in the project root.
2. Run `Data Wrangling .ipynb` to clean and validate the raw data.
3. Run `EDA .ipynb` to inspect distributions, relationships, and data quality.
4. Run `Feature Engineering .ipynb` to create model-ready features.
5. Run `Modeling_.ipynb` or `Modelling .ipynb` to train and evaluate models.
6. Use `final_model.joblib` with `fitted_preprocessing.joblib` for inference on compatible input data.

**Example command:**

```bash
jupyter lab
```

> Keep generated datasets and serialized models versioned only when their size, provenance, and reproducibility are understood. For larger artifacts, document the storage location and retrieval step here.

## Roles and Contributions

Record ownership by person or team. Update this table as work is completed.

| Contributor | Area | Responsibilities |
|---|---|---|
| [Name] | Data wrangling | Cleaning rules, missing values, validation checks |
| [Name] | Exploratory analysis | Distributions, correlations, outlier analysis |
| [Name] | Feature engineering | Encoding, transformations, feature selection or PCA |
| [Name] | Modeling | Baselines, tuning, evaluation, model export |
| [Name] | Documentation | README, decisions, reproducibility, release notes |

**Example:** `[Name]` owns feature engineering and maintains the preprocessing artifact used by the final model.

## Roadmap

Short-term milestones should be specific and verifiable.

- [ ] Add a pinned dependency file and environment setup instructions.
- [ ] Consolidate duplicate modeling notebooks into one documented workflow.
- [ ] Add automated data-quality and schema checks.
- [ ] Compare full-feature and PCA-based models using the same validation split.
- [ ] Record final metrics, selected model, and known limitations.
- [ ] Add a small prediction script with an example input row.

## Documentation Links

Keep important technical context close to the code.

- [Data description](data_description.txt) - Dataset fields and definitions.
- [Data wrangling notebook](Data%20Wrangling%20.ipynb) - Cleaning and validation.
- [EDA notebook](EDA%20.ipynb) - Exploratory analysis.
- [Feature engineering notebook](Feature%20Engineering%20.ipynb) - Transformations and feature construction.
- [Modeling notebook](Modeling_.ipynb) - Training and evaluation.
- Architecture diagram: `[add link to docs/architecture.md or an image]`
- Decision records: `[add link to docs/decisions/]`
- Experiment log: `[add link to an experiment table or tracking tool]`

### Example Architecture

```text
Raw CSV
  -> Data wrangling
  -> Exploratory analysis
  -> Feature engineering and preprocessing
  -> Model training and evaluation
  -> Saved model and prediction workflow
```

## Contribution Guidelines

### Style

- Keep notebook steps focused and name outputs clearly.
- Prefer reproducible code with fixed seeds where randomness is involved.
- Explain non-obvious data transformations near the code that applies them.
- Do not commit secrets, local environment files, or unexplained generated artifacts.

### Pull Request Process

1. Create a focused branch for one issue or feature.
2. Link the pull request to the related issue.
3. Describe the data, code, and documentation changes.
4. Include validation results and note any changed metrics or artifacts.
5. Request review from the owner of the affected area.
6. Merge only after the relevant notebooks or checks run successfully.

**Pull request example:**

> Adds log-target comparison and updates the modeling notebook. Validation used the fixed test split; RMSLE changed from `[old value]` to `[new value]`. Related to #2.

## License

This project is licensed under the [MIT License](LICENSE). Add the license file before publishing if one does not already exist.

## Contact

- **Maintainer:** [Name]
- **GitHub:** [@username](https://github.com/<username>)
- **Project issues:** [Open an issue](../../issues)
- **Email:** [name@example.com]
