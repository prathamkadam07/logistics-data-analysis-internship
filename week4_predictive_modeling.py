import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Step 1: Simulate Logistics Dataset
np.random.seed(42)
n_samples = 1000

distance = np.random.uniform(20, 500, n_samples)
weight = np.random.uniform(50, 4000, n_samples)
traffic = np.random.uniform(1, 5, n_samples)
weather = np.random.uniform(0, 1, n_samples)

delivery_hours = (
    (distance / 60)
    + (weight / 2500)
    + (traffic * 0.8)
    + (weather * 1.5)
    + np.random.normal(0, 0.4, n_samples)
)

df = pd.DataFrame({
    'Distance_KM': distance,
    'Cargo_Weight_KG': weight,
    'Traffic_Index': traffic,
    'Weather_Impact': weather,
    'Actual_Delivery_Hours': delivery_hours,
})

# Step 2: Split Data
X = df[['Distance_KM', 'Cargo_Weight_KG', 'Traffic_Index', 'Weather_Impact']]
y = df['Actual_Delivery_Hours']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Model Training
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 4: Evaluation
y_pred = rf_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f'Random Forest MAE: {mae:.4f}')
print(f'Random Forest RMSE: {rmse:.4f}')
print(f'Random Forest R2 Score: {r2:.4f}')
