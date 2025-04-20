import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

file_path = "your csv file path"
data = pd.read_csv(file_path)

encoder = LabelEncoder()
data["Location"] = encoder.fit_transform(data["Location"])

scaler = StandardScaler()
X = data[["Area (sq ft)", "Bedrooms", "Location"]]
X_scaled = scaler.fit_transform(X)
y = data["Price (Lakhs)"]

model = SGDRegressor(max_iter=5000, learning_rate='adaptive', eta0=0.01, random_state=42)
model.fit(X_scaled, y)

model_path = "model path in pkl format"
scaler_path = "scaler model path in pkl format"
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

y_pred = model.predict(X_scaled)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

mean_price = np.mean(y)
if mean_price > 0:
    accuracy = max(0, 100 - (mae / mean_price * 100))
    print(f"Model Accuracy: {accuracy:.2f}%")
    print(f"R² Score: {r2:.4f}")
else:
    print("Invalid Mean Price")

save_graph_path = "path where you want to save the graph"

plt.figure(figsize=(8, 5))
plt.scatter(y, y_pred, color='blue', alpha=0.5)
plt.xlabel("Actual Price (Lakhs)")
plt.ylabel("Predicted Price (Lakhs)")
plt.title(f"Predictions - Accuracy: {accuracy:.2f}% | R²: {r2:.4f}")
plt.grid(True)
plt.savefig(save_graph_path)
plt.show()
