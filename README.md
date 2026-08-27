# Ames Housing Price Prediction

Predicting home sale prices in Ames Dataset, using a structured, leakage-aware machine learning pipeline.

## Overview

This project tackles a classic regression problem: given a rich set of characteristics about a house — its size, quality, age, location, and dozens of other attributes — predict what it sold for. The target variable is `SalePrice`, and the model is evaluated on RMSE calculated on the log-transformed price rather than the raw dollar value, since sale prices are right-skewed and errors on expensive homes shouldn't dominate the evaluation the way they would on a linear scale.

Beyond the prediction task itself, the project was built to practice something that's easy to get wrong in a notebook environment: keeping the workflow honest. It's built around a deliberate separation of stages — understanding the data, exploring it, engineering features, and modeling — so that decisions made early (like how missing values are treated, or how the data is split) don't quietly leak information into later stages and inflate performance.

## Dataset

The dataset is the well-known Ames Housing dataset, `[Add dataset source, e.g. Kaggle link or original De Cock paper]`. It contains detailed property records for homes sold in Ames, Iowa, with a wide mix of numerical and categorical features — living area, overall quality and condition ratings, garage and basement characteristics, neighborhood, year built, and more.

**Target variable:** `SalePrice`

Some of the features investigated most closely during the project include:

- `Gr Liv Area` (above-ground living area)
- `Overall Qual` (overall material and finish quality)
- `Garage Area`
- `Total Bsmt SF`
- `Year Built`
- `Neighborhood`
- `Kitchen Qual`

The dataset is known to contain a handful of documented outliers and some structural quirks (for example, missing values that actually mean "this house doesn't have this feature," like no garage or no pool, rather than missing data in the usual sense).

## Workflow / Methodology

The project is organized into four notebooks, each with a single responsibility. Keeping these separate was a deliberate choice — it makes it much easier to enforce the rule that nothing downstream is allowed to peek at information it shouldn't have.

### 1. Data Understanding & Wrangling

Before touching a single visualization, the first notebook is about getting the data into a trustworthy state. This means writing down the business problem, checking the schema and dtypes, catching duplicate IDs, and validating logical constraints (a house shouldn't be sold before it was built, for instance). Only *structural* nulls are resolved here — cases where "missing" has an obvious real-world meaning, like a "no garage" entry in a garage-related column. Anything that requires a statistical decision, like imputing a missing lot frontage, is deliberately left for later.

The train/test split happens at the very end of this notebook, before any full-dataset statistics are computed. That ordering matters: it's the foundation for keeping the rest of the pipeline leakage-free, since every later step fits only on the training data and simply applies those fitted transformations to the test set.

### 2. Exploratory Data Analysis

With clean, split data in hand, the second notebook explores the training set only. This covers univariate and categorical distributions, correlation analysis (including flagging — not yet treating — multicollinearity candidates), and bivariate relationships between the strongest predictors and `SalePrice` using scatter plots and a pairplot.

This stage also includes formal statistical testing rather than relying on visual impressions alone: assumption checks (Shapiro–Wilk for normality, Levene's for equal variances), correlation tests (Pearson for continuous predictors like `Gr Liv Area`, Spearman for ordinal ones like `Overall Qual`), and group comparisons across categories like `Neighborhood` and `Kitchen Qual`. One concrete example of the testing driving a decision: Levene's test rejected the assumption of equal variances across neighborhoods (p < 0.001), so Kruskal–Wallis was used instead of ANOVA for that comparison — the workflow was written to make that substitution the planned fallback rather than an ad hoc fix.

No data is modified in this notebook. Its job is to confirm patterns and hand off a set of documented insights to the next stage.

### 3. Feature Engineering

This is where the data actually gets transformed — and everything here is fit on the training set and only ever applied (not refit) to the test set. Statistical imputation happens now, outlier thresholds are computed from the training distribution, and skewed predictors (along with `SalePrice` itself) are log- or Box-Cox-transformed.

New features are created based on domain reasoning — total square footage, house age at time of sale, time since remodel, and simple flags like has-pool or has-garage. Categorical features are encoded (ordinal mappings for quality/condition scales, one-hot encoding for nominal categories, with rare categories consolidated first), numeric features are scaled where the downstream model needs it, and multicollinearity is addressed using VIF. Feature selection combines filter methods (correlation, mutual information) with embedded methods (Lasso coefficients), and PCA is explored as an optional, separate modeling track rather than a required step.

All of these steps are bundled into a single `ColumnTransformer` pipeline, fit once on the training data and persisted (via `joblib`) so that Notebook 4 uses the exact same fitted transformations rather than re-deriving them.

### 4. Modeling & Evaluation

The final notebook loads the saved pipeline and starts from a simple Linear Regression baseline before training and comparing several models — Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, and optionally XGBoost — each wrapped in a `Pipeline` alongside the preprocessing step. Hyperparameters are tuned with grid or randomized search, and models are compared on RMSE, R², and training time.

Once a final model is selected, evaluation goes beyond a single metric: residual plots, error broken down by neighborhood, price band, and property age, and a look at the worst-predicted individual rows. Explainability is handled with SHAP (using a tree explainer on the best-performing tree-based model) and partial dependence plots for the top features, with permutation importance used as a cross-check against both SHAP and the filter-based importance scores from Notebook 3.

## Key Findings / Observations

- Levene's test found unequal variance in `SalePrice` across neighborhoods (p < 0.001), which is why Kruskal–Wallis replaced ANOVA for that specific group comparison rather than assuming ANOVA's assumptions held.
- `SalePrice` itself is skewed enough to warrant a log transform, which is also why the modeling target and evaluation metric are both defined in log space.
- A short list of numerical predictors — `Gr Liv Area`, `Overall Qual`, `Garage Area`, `Total Bsmt SF`, and `Year Built` — were treated as the strongest candidates worth investigating closely in both the bivariate analysis and the pairplot.

`[Add specific correlation values, model RMSE/R² results, and top SHAP features once the modeling notebook has been run.]`

## Technologies & Libraries

- **Python** — core language for the project
- **pandas / numpy** — data wrangling and numerical operations
- **scikit-learn** — preprocessing (`ColumnTransformer`, `Pipeline`), model training, and hyperparameter search (`GridSearchCV` / `RandomizedSearchCV`)
- **matplotlib / seaborn** — visualization for EDA (implied by boxplots, scatter plots, and pairplots)
- **scipy** — statistical testing (Shapiro–Wilk, Levene's, Pearson, Spearman, ANOVA, Kruskal–Wallis, chi-square)
- **XGBoost** — optional gradient boosting model
- **SHAP** — model explainability
- **joblib** — persisting the fitted preprocessing pipeline

`[Confirm exact library versions once the environment is finalized.]`

## Project Structure

```
├── 01_data_understanding_and_wrangling.ipynb
├── 02_eda.ipynb
├── 03_feature_engineering.ipynb
├── 04_modeling_and_evaluation.ipynb
└── README.md
```


## What I Learned

Structuring the project around four single-purpose notebooks made the leakage-prevention rule much easier to actually follow, rather than just state as a principle. Deciding what counts as a "structural" null versus one that needs statistical imputation turned out to be a more nuanced judgment call than expected, and worth separating into its own step rather than handling everything in one pass. Running formal statistical tests instead of relying on visual intuition alone — like catching the unequal-variance issue with Levene's test — also made it clear how much a single test result can change a downstream methodological choice.

