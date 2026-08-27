import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Step A: Generate Synthetic Logistics Dataset
np.random.seed(42)
n_records = 300

data = pd.DataFrame(
    {
        'Distance_KM': np.random.uniform(10, 300, n_records),
        'Cargo_Weight_KG': np.random.uniform(100, 5000, n_records),
        'Transport_Mode': np.random.choice(
            ['Light Van', 'Medium Truck', 'Heavy Freight'], n_records
        ),
    }
)

# Feature Calculations with noise
data['Delivery_Time_Hours'] = (
    data['Distance_KM'] / 50
    + (data['Cargo_Weight_KG'] / 2000)
    + np.random.normal(0.5, 0.2, n_records)
)
data['Fuel_Cost_USD'] = (
    data['Distance_KM'] * 0.45
    + (data['Cargo_Weight_KG'] * 0.08)
    + np.random.normal(10, 2, n_records)
)

# Step B: Correlation Analysis
numeric_df = data[
    ['Distance_KM', 'Cargo_Weight_KG', 'Delivery_Time_Hours', 'Fuel_Cost_USD']
]
correlation_matrix = numeric_df.corr()
print('Correlation Matrix:\n', correlation_matrix)

# Step C: Visualization Code (Matplotlib & Seaborn)
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='Blues', fmt='.2f')
plt.title('Logistics Metrics Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.boxplot(
    x='Transport_Mode',
    y='Delivery_Time_Hours',
    data=data,
    palette='Set2',
)
plt.title('Delivery Time Variance by Transport Mode')
plt.xlabel('Transport Mode')
plt.ylabel('Delivery Time (Hours)')
plt.tight_layout()
plt.savefig('transport_mode_delays.png')
plt.close()

print('Visualizations generated and saved successfully.')
