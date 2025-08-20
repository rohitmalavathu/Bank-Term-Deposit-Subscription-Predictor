# Bank Term Deposit Subscription Predictor

## Overview

This project predicts whether a banking client will subscribe to a term deposit using tabular features from the **Kaggle Playground Series — Season 5, Episode 8** competition. It includes an end-to-end notebook covering EDA, preprocessing, modeling, and generation of a Kaggle-ready submission.

## Datasets

The following datasets are used in this project (downloaded from the competition’s **Data** tab and stored locally, e.g., `./data/`):

* **train.csv**: Training features and the competition’s target column.
* **test.csv**: Test features without the target.
* **sample\_submission.csv**: Example file showing required submission format.
* *(Output)* **submission.csv**: Created by the notebook for upload to Kaggle.
  These files are provided by the Kaggle competition.

## Data Processing Steps

1. **Import Libraries**: Load `numpy`, `pandas`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, and plotting tools.
2. **Load Data**: Read `train.csv`, `test.csv`, and `sample_submission.csv` from the local `data/` directory.
3. **Exploratory Data Analysis (EDA)**: Inspect distributions, missingness, and target balance.
4. **Data Cleaning**: Handle missing values and inconsistent categories.
5. **Feature Engineering**: Encode categorical variables (e.g., one-hot/target/ordinal), derive simple interaction or calendar features as appropriate.
6. **Modeling & Validation**: Establish baselines (e.g., Logistic Regression), train boosted trees (XGBoost/LightGBM/CatBoost), and evaluate via cross-validation (ROC-AUC/PR).
7. **Class Imbalance Handling**: Consider class weights or resampling (e.g., SMOTE) if needed.
8. **Submission Generation**: Predict on `test.csv` and write `submission.csv` in the required format for Kaggle upload.

## Key Features

* **Reproducible Workflow**: Single notebook (`code.ipynb`) from EDA → training → submission.
* **Robust Encoding**: Clean handling of categorical variables to avoid leakage.
* **Reliable Validation**: Cross-validated metrics for comparable model selection.
* **Imbalance-Aware**: Options for class weighting or resampling.
* **Leaderboard-Ready**: Produces `submission.csv` compatible with the competition format.

## Dependencies

Install the core Python packages:

```bash
pip install numpy pandas scikit-learn xgboost lightgbm catboost imbalanced-learn matplotlib seaborn jupyter
```

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/rohitmalavathu/Bank-Term-Deposit-Subscription-Predictor.git
   cd Bank-Term-Deposit-Subscription-Predictor
   ```
2. Download the Kaggle competition data and place all CSVs in `./data/`.
3. Launch Jupyter and open the notebook:

   ```bash
   jupyter notebook
   ```
4. Run `code.ipynb` end-to-end to train models and create `submission.csv`.

## Results

* The notebook reports validation metrics (e.g., ROC-AUC) and saves a Kaggle-ready `submission.csv`.
* Public/Private leaderboard scores can be added here after you submit on the competition page.

## Future Improvements

* Add a configurable `sklearn` `Pipeline` with `ColumnTransformer`.
* Perform systematic hyperparameter tuning (e.g., Optuna).
* Try model ensembling/stacking for potential leaderboard gains.
* Add SHAP-based feature importance and error analysis.
* Package the workflow into a small CLI for headless runs.

## Author

Rohit Malavathu
[rohitmalavathu@vt.edu](mailto:rohitmalavathu@vt.edu)
