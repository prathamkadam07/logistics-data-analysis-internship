import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Step 1: Simulate Raw Logistics Dataset with Imperfections
raw_data = {
    'Shipment_ID': ['SH001', 'SH002', 'SH003', 'SH004', 'SH005', 'SH006'],
    'Cargo_Weight_KG': [1200, np.nan, 3400, 450, 15000, 2200],  # Outlier & Missing
    'Actual_Hours': [4.5, 2.1, np.nan, 1.2, 45.0, 3.8],  # Outlier (45.0) & Missing
    'Fuel_Liters': [35.0, 18.2, 78.5, 12.0, 250.0, 31.0],
    'Vehicle_Type': ['Truck', 'Van', 'Truck', 'Van', 'Truck', 'Truck'],
}
df = pd.DataFrame(raw_data)

# Step 2: Handling Missing Values
df['Cargo_Weight_KG'] = df['Cargo_Weight_KG'].fillna(
    df['Cargo_Weight_KG'].median()
)
df['Actual_Hours'] = df['Actual_Hours'].fillna(df['Actual_Hours'].median())

# Step 3: Outlier Treatment using IQR Method
Q1 = df['Actual_Hours'].quantile(0.25)
Q3 = df['Actual_Hours'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR
lower_bound = max(0, Q1 - 1.5 * IQR)

df['Actual_Hours'] = np.where(
    df['Actual_Hours'] > upper_bound,
    upper_bound,
    np.where(df['Actual_Hours'] < lower_bound, lower_bound, df['Actual_Hours']),
)

# Step 4: Categorical Encoding & Feature Scaling
df = pd.get_dummies(df, columns=['Vehicle_Type'], drop_first=True)

scaler = StandardScaler()
numeric_cols = ['Cargo_Weight_KG', 'Actual_Hours', 'Fuel_Liters']
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print('Preprocessed Clean Dataset:')
print(df)
