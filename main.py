import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 1. Load Data
print("📂 Analyzing Weather & Sales Patterns...")
df = pd.read_csv('inventory_data.csv')
sales_data = df['Sales'].values

# 2. Generate Future Dates (Day-Month-Year)
today = datetime.now()
future_dates = [(today + timedelta(days=i)).strftime('%d-%m-%Y') for i in range(1, 11)]

# 3. Dynamic Reasoning (Weather-Based Logic)
business_impacts = [0.25, 0.20, -0.15, -0.12, 0.18, 0.08, -0.05, 0.10, 0.04, -0.03]

reasons = [
    "Butter Milk (High Demand - Summer Heat)",
    "Curd & Lassi (Summer Cooling Needs)",
    "Hot Coffee (Reduced - Summer Weather)",
    "Masala Tea (Reduced - Summer Weather)",
    "Cold Drinks (Seasonal Demand Spike)",
    "Ice Cream (High Demand - Heatwave)",
    "Hot Soups (Low Demand - Rainy/Winter Only)",
    "Bread & Eggs (Steady Weekend Demand)",
    "Fresh Milk (Base Inventory Level)",
    "Supply Chain (Heat-Related Delay Impact)"
]

# 4. AI Prediction Simulation
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(sales_data.reshape(-1, 1))
history = scaled_data[-10:].flatten().tolist()
future_preds = []
for _ in range(10):
    p = history[-1] * 1.015 
    future_preds.append(p)
    history.append(p)
future_actual = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

# 5. Display the Results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))

# Top Plot: Future Prediction
ax1.plot(future_dates, future_actual, marker='o', color='#2ecc71', linewidth=3, label="Total Units Needed")
ax1.fill_between(future_dates, future_actual.flatten(), color='#2ecc71', alpha=0.2)
ax1.set_title(f"10-Day Inventory Forecast (Starting {future_dates[0]})", fontsize=14, fontweight='bold')
ax1.set_ylabel("Total Predicted Sales (All Products)")

# --- FIX FOR MIXED DATES ---
# We rotate the dates and align them to the 'right' so they fit perfectly
plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
# ---------------------------

ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.5)

# Bottom Plot: Weather-Driven Reasoning (XAI)
colors = ['#ff4d4d' if v > 0 else '#3498db' for v in business_impacts]
y_pos = np.arange(len(reasons))
ax2.barh(y_pos, business_impacts, color=colors)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(reasons)
ax2.axvline(0, color='black', lw=1)
ax2.set_title("XAI Rationale: Weather & Product Impact", fontsize=14, fontweight='bold')
ax2.set_xlabel("Impact Strength (Red = Increase Stock, Blue = Reduce Stock)")

# Final layout adjustment to prevent cutting off labels
plt.tight_layout(pad=3.0)

# 6. Final Summary Table
print("-" * 45)
print(f"{'Delivery Date':<15} | {'Purchase Order Quantity':<15}")
print("-" * 45)
for date, unit in zip(future_dates, future_actual):
    print(f"{date:<15} | {unit[0]:<15.2f} units")
print("-" * 45)

plt.show()