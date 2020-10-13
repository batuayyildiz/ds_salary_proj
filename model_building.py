# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("eda_data.csv")

# Choose relevant attributes
print(df.columns)
attrs = ["avg_salary", "Rating", "Size", "Type of ownership", "Industry", "Sector", "Revenue",
         "hourly", "employer_provided", "job_state", "is_same_location", "python_yn",
         "spark_yn", "aws_yn", "excel_yn", "job_simp", "seniority", "desc_len", "num_of_comps"]
df_new = df[attrs]
# Get Dummies
df_dummy = pd.get_dummies(data = df_new)
# Train Test Split
from sklearn.model_selection import train_test_split
X = df_dummy.drop("avg_salary", axis = 1)
y = df_dummy["avg_salary"].values

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.2,
                                                    random_state = 42)
# Multiple Linear Regression
import statsmodels.api as sm
X_sm = sm.add_constant(X)
model = sm.OLS(y, X_sm)
print(model.fit().summary())
# Linear Regression
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score
lr = LinearRegression()
lr_score = np.mean(cross_val_score(lr, X_train, y_train, scoring = "neg_mean_absolute_error", cv = 5))
print(f"Linear Regression Cross Validation Score: {lr_score}")

# Lasso Regression
lasso = Lasso()
lasso_score = np.mean(cross_val_score(lasso, X_train, y_train, scoring = "neg_mean_absolute_error", cv = 5))
print(f"Lasso Regression Cross Validation Score: {lasso_score}")

# Tuning Lasso with different alpha values
alphas = []
errors = []

for i in range(1, 100):
    alphas.append(i / 100)
    lasso = Lasso(alpha = i / 100)
    errors.append(np.mean(cross_val_score(lasso, X_train, y_train, scoring = "neg_mean_absolute_error", cv = 5)))

plt.plot(alphas, errors)

err = dict(zip(alphas, errors))
# Creating a dataframe with this alpha-error pair
df_err = pd.DataFrame(err, columns = ["alpha", "error"])
# Random Forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
rf_score = np.mean(cross_val_score(rf,
                                   X_train,
                                   y_train,
                                   scoring = "neg_mean_absolute_error",
                                   cv = 5))
print(rf_score)
# Hyperparameter Tuning with GridSearchCV
from sklearn.model_selection import GridSearchCV
grid = {"n_estimators" : np.arange(10, 300, 10),
        "max_depth" : [None, 1, 2, 3],
        "criterion" : ["mse", "mae"],
        "max_features" : ["auto", "sqrt", "log2"]}

gs_rf = GridSearchCV(estimator = rf,
                     param_grid = grid,
                     scoring = "neg_mean_absolute_error",
                     cv = 5,
                     verbose = 2)
gs_rf.fit(X_train, y_train) # Takes too long time.

# Test ensembles
preds = rf.predict(X_test)

# Check some evaluation metrics
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, preds)
print(f"MAE: {mae}")



 