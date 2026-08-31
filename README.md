# Wine Quality Prediction (Binary Classification)

A machine learning project that predicts whether a wine is "good" quality
(score ≥ 7) or not, using its chemical properties. Built and extended from
a learning exercise based on the UCI Wine Quality dataset.

## Dataset

- Source: [UCI Machine Learning Repository - Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality)
- Two datasets used: `winequality-red.csv` and `winequality-white.csv`
- 11 chemical input features (acidity, sugar, sulphates, alcohol, etc.)
- Original `quality` score (0–10) is converted into a binary label:
  `1` if quality ≥ 7 ("good"), else `0`

## What this project does

1. Loads and explores the red and white wine datasets
2. Preprocesses data (binarizes the target, scales features for Logistic Regression)
3. Trains two models on each dataset: Logistic Regression and Random Forest
4. Evaluates both with accuracy, classification report, and confusion matrix
5. Compares feature importance to see which chemical properties matter most
6. Saves trained models to disk with `joblib` for reuse without retraining

## Setup

```bash
git clone https://github.com/afrinps19/winery/tree/main
cd winery
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Open `wine_quality.ipynb` in VS Code, select the `venv` interpreter as your
kernel, and run all cells in order.

## Requirements

See `requirements.txt` — pandas, numpy, matplotlib, seaborn, scikit-learn,
jupyter, ipykernel.

## Results

## Results

| Model               | Red Wine | White Wine |
|----------------------|----------|------------|
| Logistic Regression  | 89.37%   | 80.00%     |
| Random Forest         | 94.37%   | 89.08%     |

Random Forest outperformed Logistic Regression on both datasets. Red wine
classification was consistently more accurate than white wine, likely due
to differences in class balance and feature separability between the two
datasets. Alcohol and sulphates were among the most important predictors
in both cases.

## Future improvements

- Hyperparameter tuning with GridSearchCV
- A standalone prediction script for new wine samples