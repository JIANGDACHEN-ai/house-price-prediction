# main.py
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------
# Step 1: Create Fake Data
# -----------------------
np.random.seed(42)
num_samples = 500

data = {
    "MedInc": np.random.normal(5, 2, num_samples),
    "HouseAge": np.random.randint(1, 50, num_samples),
    "AveRooms": np.random.normal(5, 1, num_samples),
    "AveBedrms": np.random.normal(1, 0.3, num_samples),
    "Population": np.random.randint(200, 5000, num_samples),
    "AveOccup": np.random.normal(3, 1, num_samples),
    "Latitude": np.random.uniform(32, 42, num_samples),
    "Longitude": np.random.uniform(-124, -114, num_samples),
}

# Simulate house prices (target variable)
target = (
    data["MedInc"] * 0.4 +
    data["HouseAge"] * 0.01 -
    data["AveBedrms"] * 0.5 +
    np.random.normal(0, 0.5, num_samples)
)

df = pd.DataFrame(data)
df["MedHouseVal"] = target

# -----------------------
# Step 2: Train a Model
# -----------------------
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
import joblib
print("About to save the model...")
file_path=joblib.dump(model,"model.pkl")
print("📦 Model saved at:",file_path)
# -----------------------
# Step 3: Evaluate the Model
# -----------------------
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("✅ Done! Model is trained.")
print("📉 Mean Squared Error:", mse)
print("📈 R² Score:", r2)
df ["MedHouseVal"]=target
# Visualize relationships between features and house value
plt.figure(figsize=(10, 6))
sns.scatterplot(x="MedInc", y="MedHouseVal", data=df)
plt.title("Median Income vs. House Value")
plt.xlabel("Median Income")
plt.ylabel("House Price")
plt.grid(True)
plt.tight_layout()
plt.savefig("scatter_income_vs_price.png")
plt.show()

# Optional: see heatmap of feature correlations
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("heatmap_correlation.png")
plt.show()
joblib.dump(model,"model.pkl")








