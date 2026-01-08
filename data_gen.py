import pandas as pd
import numpy as np

# Creating 2 years of daily inventory sales data
dates = pd.date_range(start='2024-01-01', periods=730, freq='D')
np.random.seed(42)

# Generate sales: Base of 50 + Seasonality + Random Noise
# This gives the AI a pattern to learn
sales = 50 + (20 * np.sin(2 * np.pi * dates.dayofyear / 365)) + np.random.normal(0, 5, 730)
df = pd.DataFrame({'Date': dates, 'Sales': sales})

# Save the data to your project folder
df.to_csv('inventory_data.csv', index=False)
print("✅ Success! 'inventory_data.csv' has been created in your folder.")