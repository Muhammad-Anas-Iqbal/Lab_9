import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# Load dataset
yachts = pd.read_csv("yacht_hydrodynamics.data", sep=r"\s+", header=None)

# Prepare data
X = yachts[[5]]
y = yachts[6]

# Linear Regression
linear_model = LinearRegression()

# Quadratic Regression
quadratic_model = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("linear", LinearRegression())
])

# Five-fold cross-validation
folds = KFold(n_splits=5, shuffle=True, random_state=42)

scoring = "neg_root_mean_squared_error"

# Calculate RMSE for linear model
linear_rmse = -cross_val_score(
    linear_model, X, y, cv=folds, scoring=scoring
)

# Calculate RMSE for quadratic model
quadratic_rmse = -cross_val_score(
    quadratic_model, X, y, cv=folds, scoring=scoring
)

# Print results
print("Linear Model Fold RMSE:", linear_rmse.round(2))
print("Linear Model Mean RMSE:", linear_rmse.mean().round(2))

print("\nQuadratic Model Fold RMSE:", quadratic_rmse.round(2))
print("Quadratic Model Mean RMSE:", quadratic_rmse.mean().round(2))

# Compare models
if linear_rmse.mean() < quadratic_rmse.mean():
    print("\nBetter Model: Linear Regression")
else:
    print("\nBetter Model: Quadratic Regression")