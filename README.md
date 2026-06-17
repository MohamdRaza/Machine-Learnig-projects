# Titanic ML Project

A hands-on practice notebook covering classification, regression, clustering, and preprocessing techniques using the Titanic dataset (plus a few standalone examples).

## What's in this notebook

1. Classification models (Titanic survival prediction)**
The same Titanic data (Pclass, Sex, Age, SibSp, Parch, Fare, Embarked) is preprocessed and fed into several classifiers, with accuracy, confusion matrix, and classification report for each:
- K-Nearest Neighbors (KNN)
- Naive Bayes (GaussianNB)
- Logistic Regression
- Support Vector Machine (SVC)
- Decision Tree (with tree plot)
- Gradient Boosting
- AdaBoost
- XGBoost with GridSearchCV hyperparameter tuning

2. Regression examples
- Simple linear regression on a small sample dataset
- Predicting Fare from other Titanic features using Linear Regression
- A separate salary prediction example: connects to a MySQL database, pulls an `employees` table, cleans it, trains a Linear Regression model, and saves it with `joblib`

3. Feature scaling & encoding
- Comparing MinMaxScaler, StandardScaler, MaxAbsScaler, and RobustScaler on a sample dataset, then applying scaling to the Titanic data
- Label Encoding and One-Hot Encoding examples on Titanic categorical columns

4. Clustering
- K-Means (with the elbow method) on a sample income/spending dataset
- Hierarchical/Agglomerative Clustering with dendrograms
- DBSCAN on a sample dataset with irregularly shaped clusters
- K-Means, Agglomerative, and DBSCAN applied together to Titanic features (Age, Fare, Pclass)

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
joblib
mysql-connector-python
scipy
```

Install with:
```
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib mysql-connector-python scipy
```

## How to run

1. Place `train.csv` (the Titanic dataset) in the same folder as the notebook.
2. Run cells from top to bottom — most sections reload and re-clean the data independently, so they can also be run somewhat out of order.
3. The MySQL section requires a running MySQL server with a `company_db` database and `employees` table; update the host/user/password/database in that cell to match your setup, or skip that section if you don't need it.

## Note

This notebook is meant as a practice/reference collection of ML techniques rather than a single end-to-end pipeline — each model section is mostly self-contained, so feel free to jump to the algorithm you're interested in.
