import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
homes = pd.read_csv("real_estate.csv")

# Prepare data
homes["distance_100m"] = homes["X3 distance to the nearest MRT station"] / 100

X = homes[["distance_100m"]]
y = homes["Y house price of unit area"]

# Fit linear regression
model = LinearRegression()
model.fit(X, y)

# Print slope and intercept
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

# Predict price at 500 meters
prediction = model.predict(pd.DataFrame({"distance_100m": [5]}))
print("Predicted price at 500 meters:", prediction[0])

# Plot observations and fitted line
plt.scatter(X["distance_100m"], y, label="Actual Data")

plt.plot(
    X["distance_100m"].sort_values(),
    model.predict(X.loc[X["distance_100m"].sort_values().index]),
    color="red",
    label="Regression Line"
)

plt.xlabel("Distance to Metro Station (100 meters)")
plt.ylabel("House Price per Unit Area")
plt.title("Housing Prices Near Transit")
plt.legend()
plt.savefig("task1_housing.png")
plt.show()