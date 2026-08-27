import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Step A: Sample logistics delivery dataset
data = pd.DataFrame(
    {
        'distance_km': [12.5, 4.2, 28.0, 15.3, 8.7, 33.1],
        'payload_kg': [150, 45, 600, 210, 80, 750],
        'traffic_level': [3, 1, 4, 2, 1, 5],  # Scale 1 (low) to 5 (heavy)
        'actual_delivery_minutes': [35, 14, 85, 42, 22, 110],
    }
)

# Step B: Feature matrix and Target variable
X = data[['distance_km', 'payload_kg', 'traffic_level']]
y = data['actual_delivery_minutes']

# Step C: Train-Test Split & Random Forest Training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Step D: Feature Importance Output
print('Model trained successfully.')
for feature, importance in zip(X.columns, model.feature_importances_):
  print(f'Feature: {feature}, Importance: {importance:.4f}')
