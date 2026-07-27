# F1 Podium-Finish Prediction using Race Result Data

Binary classification project that predicts whether a Formula 1 driver will finish on the podium (1st, 2nd, or 3rd) using historical race result data from the 2019–2024 seasons.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methods](#methods)
- [Results](#results)
- [Libraries](#libraries)
- [Setup](#setup)
- [Project Structure](#project-structure)
- [Future Work](#future-work)

## Introduction

Formula 1 race outcomes depend on many factors—driver skill, team performance, track characteristics, and starting grid position, among others. Rather than predicting the exact finishing order, this project frames the problem as **binary classification**: will a driver achieve a podium finish or not?

Three different classification models are trained and compared on the same features and evaluation setup:

- **KNN** (baseline)
- **Logistic Regression**
- **XGBoost** (with GridSearchCV hyperparameter tuning)

Models are trained on 2019–2023 data and evaluated on a held-out **2024 season** for external validation, in addition to an 80/20 train/test split on the historical data.

## Dataset

- **File:** `16_F1_Race_Results_2019_2024.csv`
- **Target:** `podium_finish` — 1 if finishing position is 1st–3rd, 0 otherwise
- **Features used:**
  - `Starting Grid`
  - `Driver` (one-hot encoded)
  - `Team` (one-hot encoded)
  - `Track` (one-hot encoded)

Rows with missing values in the selected features are dropped. The `season` column is used only for splitting train vs. validation data and is excluded from model inputs to avoid leakage.

## Methods

1. **Exploratory analysis** — missing values, top drivers/teams by points, podium distribution
2. **Preprocessing** — binary target creation, one-hot encoding, stratified 80/20 split (2019–2023)
3. **Feature scaling** — `StandardScaler` applied for KNN and Logistic Regression
4. **Model training** — KNN (`k=5`), Logistic Regression, baseline XGBoost
5. **Hyperparameter tuning** — `GridSearchCV` on XGBoost (`n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`)
6. **Evaluation** — classification reports, confusion matrices, ROC-AUC on the 2024 holdout

## Results

On the **2024 external validation** set (ROC-AUC):

| Model | ROC-AUC |
|---|---|
| XGBoost (Tuned) | 0.921 |
| Logistic Regression | 0.897 |
| KNN | 0.713 |
| Random guess | 0.500 |

XGBoost (after tuning) was the strongest performer, followed by Logistic Regression. KNN served as a useful baseline but lagged on the holdout season.

For this problem, **podium-class recall and F1** are especially important metrics—a model that rarely predicts podiums correctly would miss the outcomes we care about most.

See `f1_analysis_2019_2024.ipynb` for full classification reports, confusion matrices, and the color-coded ROC curve plot.

## Libraries

- [pandas](https://pandas.pydata.org/) — data loading and manipulation
- [NumPy](https://numpy.org/) — numerical operations (via pandas/sklearn)
- [scikit-learn](https://scikit-learn.org/) — preprocessing, models, metrics, GridSearchCV
- [XGBoost](https://xgboost.readthedocs.io/) — gradient boosted classifier
- [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/) — visualizations

### Prerequisites

- Python 3.11+
- Jupyter Notebook or JupyterLab

### Setup

Once the project is downloaded to your local directory...
```bash
cd f1-podium-predictor/
```
If necessary, upgrade pip...
```bash
pip install --upgrade pip
```
Instantiate a virtual environment 
```bash
python -m venv .venv
# Activate on windows
.venv\Scripts\activate.bat
# ...on mac/linux
source .venv/bin/activate
```
Install the required libraries from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Run the notebook

1. Place `16_F1_Race_Results_2019_2024.csv` in the project root (same folder as the notebook).
2. Open and run `f1_analysis_2019_2024.ipynb` from top to bottom.
3. For the modeling section, run cells in order—especially **Data Preparation** before the individual model cells, and **Grid Search** before **Model Comparison** and **ROC Curve**.

## Project Structure

```
.
├── README.md
├── requirements.txt
├── f1_analysis_2019_2024.ipynb   # Main analysis notebook
└── 16_F1_Race_Results_2019_2024.csv
```

## Future Work

Potential improvements mentioned in the analysis:

- Add race-day features such as tire compound, pit strategy, weather, penalties, or mechanical retirements
- Address class imbalance (podium finishes are relatively rare)
- Try additional models or ensemble methods
- Expand temporal validation across multiple holdout seasons
