# Credit Risk Prediction

A machine learning project that predicts whether a borrower may experience serious delinquency within the next two years, using the **Give Me Some Credit** dataset.

## Overview

The project explores an end-to-end classification workflow:

- Exploratory data analysis (EDA)
- Data preprocessing
- Handling class imbalance
- Training and comparing machine learning models
- Hyperparameter tuning with Optuna
- A Streamlit app for interactive predictions

## Dataset

- **Dataset:** Give Me Some Credit
- **Target:** `SeriousDlqin2yrs`
- **Task:** Binary classification

The target indicates whether a person experienced serious delinquency within the next two years.

Download the dataset from Kaggle: [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit/data)

Place the dataset files in the location expected by the notebooks, or update the file paths in the notebook before running it.

## Models

The project compares three classifiers:

- Logistic Regression
- Decision Tree
- Random Forest

To address class imbalance, the training workflow compares:

- Baseline training
- Random oversampling
- SMOTE

The best sampling configuration is selected for each model using validation performance. Optuna is then used to tune model hyperparameters, with **PR-AUC** as the optimization metric.

## Input Features

The prediction app uses these features:

- `RevolvingUtilizationOfUnsecuredLines`
- `age`
- `NumberOfTime30-59DaysPastDueNotWorse`
- `DebtRatio`
- `MonthlyIncome`
- `NumberOfOpenCreditLinesAndLoans`
- `NumberOfTimes90DaysLate`
- `NumberRealEstateLoansOrLines`
- `NumberOfTime60-89DaysPastDueNotWorse`
- `NumberOfDependents`

## Project Structure

A suggested structure is:

```text
credit-risk-prediction/
├── data/
│   └── ...
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_preprocessing_and_model_training.ipynb
├── models/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   └── random_forest.joblib
├── app.py
├── requirements.txt
└── README.md
```

Your actual folder and notebook names may differ. Keep the saved model files in a `models` directory that the app can locate.

## Installation

Python 3.10 or newer is recommended.

1. Clone or download this repository.
2. Open a terminal in the project directory.
3. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

   On Windows, activate it with:

   ```powershell
   .venv\Scripts\activate
   ```

4. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

If the notebooks use additional libraries, install those as well.

## Run the Notebooks

Run the EDA notebook first, then the preprocessing and model-training notebook. Execute the cells in order to train, evaluate, tune, and save the models.

Make sure the dataset paths are correct and that the model-saving cells complete successfully.

## Run the Streamlit App

From the directory containing `app.py`, run:

```bash
streamlit run app.py
```

Choose a loaded model, enter the requested feature values, and submit the form to view the predicted delinquency probability.

If a model does not appear in the app, check that its `.joblib` file exists in the expected `models` directory and rerun the model-saving cell if necessary.

## Evaluation

Because the target classes are imbalanced, accuracy alone may not provide a useful comparison. The training workflow uses **PR-AUC** for tuning and should also be evaluated with metrics such as precision, recall, F1-score, and the confusion matrix.

## Limitations

- Predictions depend on the quality and representativeness of the training data.
- Results may change with different train/validation splits, preprocessing choices, or sampling methods.
- A predicted probability is a model estimate, not a guarantee of a borrower's future behavior.
- This project is for educational and analytical use; it should not be used as the sole basis for real lending decisions.
