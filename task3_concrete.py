import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Load dataset and remove duplicate rows
concrete = pd.read_csv("concrete_strength.csv").drop_duplicates()

# Prepare data
target = "Concrete compressive strength"

X = concrete.drop(columns=target)
y = concrete[target]

# Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# Five-fold cross-validation
folds = KFold(n_splits=5, shuffle=True, random_state=42)

scoring = "neg_root_mean_squared_error"

# Linear Regression baseline
linear_model = LinearRegression()

linear_rmse = -cross_val_score(
    linear_model, X_train, y_train, cv=folds, scoring=scoring
)

print("\nLinear Model Fold RMSE:", linear_rmse.round(2))
print("Linear Model Mean RMSE:", linear_rmse.mean().round(2))

# Create Polynomial and Ridge pipeline
ridge_model = Pipeline([
    ("poly", PolynomialFeatures(include_bias=False)),
    ("scale", StandardScaler()),
    ("ridge", Ridge())
])

# Parameter grid
grid = {
    "poly__degree": [1, 2, 3],
    "ridge__alpha": [0.01, 0.1, 1, 10, 100]
}

# Grid Search
search = GridSearchCV(
    ridge_model,
    grid,
    cv=folds,
    scoring=scoring,
    refit=True
)

search.fit(X_train, y_train)

# Print best parameters
print("\nBest Parameters:", search.best_params_)
print("Best Mean CV RMSE:", -search.best_score_)

# Display three highest-ranked settings
results = pd.DataFrame(search.cv_results_)

top_three = results.sort_values("rank_test_score").head(3)

print("\nTop Three Settings:")

for index, row in top_three.iterrows():
    print(
        "Rank:", row["rank_test_score"],
        "Degree:", row["param_poly__degree"],
        "Alpha:", row["param_ridge__alpha"],
        "Mean CV RMSE:", -row["mean_test_score"]
    )

# Predict test data using the best model
test_predictions = search.predict(X_test)

# Calculate test RMSE and MAE
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
test_mae = mean_absolute_error(y_test, test_predictions)

print("\nTest RMSE:", round(test_rmse, 2))
print("Test MAE:", round(test_mae, 2))

# Plot actual vs predicted values
plt.scatter(y_test, test_predictions)

plt.xlabel("Actual Concrete Strength (MPa)")
plt.ylabel("Predicted Concrete Strength (MPa)")
plt.title("Actual vs Predicted Concrete Strength")

# Reference line y = x
minimum = min(y_test.min(), test_predictions.min())
maximum = max(y_test.max(), test_predictions.max())

plt.plot([minimum, maximum], [minimum, maximum], color="red")

plt.savefig("task3_concrete.png")
plt.show()