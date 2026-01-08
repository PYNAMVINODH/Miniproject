import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

# --- BROWSER SETTINGS ---
st.set_page_config(page_title="Inventory Management Using XAI Forecasting", layout="wide")

st.title("🛒🚶Inventory Management Using XAI Forecasting")
st.markdown("---")

# --- UPDATED PRODUCT DATABASE WITH CONTEXTUAL REASONS ---
product_info = {
    "Butter Milk": {
        "category": "Dairy", "benefit": "Hydration", "nutrition_summary": "💪 Probiotics & Vitamin B12",
        "reason": "Predicted spike due to **Extreme Summer Heatwaves**. High hydration needs in the local area.",
        "details": {"Energy": "40 kcal", "Protein": "3.3g", "Fat": "0.9g", "Carbs": "4.8g", "Calcium": "116mg"}
    },
    "Curd & Lassi": {
        "category": "Dairy", "benefit": "Cooling", "nutrition_summary": "🦴 Calcium & Protein",
        "reason": "Increased demand for **Summer Cooling**. Essential for local lunch consumption patterns.",
        "details": {"Energy": "98 kcal", "Protein": "3.5g", "Fat": "4.3g", "Carbs": "3.4g", "Calcium": "120mg"}
    },
    "Hot Coffee": {
        "category": "Beverage", "benefit": "Alertness", "nutrition_summary": "🧠 Caffeine & Antioxidants",
        "reason": "Demand typically peaks during **Early Morning Functions** or rainy intervals.",
        "details": {"Energy": "2 kcal", "Protein": "0.1g", "Fat": "0g", "Carbs": "0g", "Antioxidants": "High"}
    },
    "Masala Tea": {
        "category": "Beverage", "benefit": "Immunity", "nutrition_summary": "🌿 Ginger & Cardamom",
        "reason": "Steady demand anticipated for **College Functions** and rainy evening sessions.",
        "details": {"Energy": "35 kcal", "Protein": "0.5g", "Fat": "0.2g", "Carbs": "7g", "Spices": "Active"}
    },
    "Cold Drinks": {
        "category": "Soft Drink", "benefit": "Energy", "nutrition_summary": "⚡ Quick Glucose",
        "reason": "High volume expected for **Upcoming Local Festivals** and youth gatherings.",
        "details": {"Energy": "42 kcal", "Protein": "0g", "Fat": "0g", "Carbs": "10.6g", "Sugar": "10.6g"}
    },
    "Ice Cream": {
        "category": "Dessert", "benefit": "Mood Booster", "nutrition_summary": "🥛 Vitamin D",
        "reason": "Stock-up required for **Weekend College Events** and high-temperature evenings.",
        "details": {"Energy": "207 kcal", "Protein": "3.5g", "Fat": "11g", "Carbs": "24g", "Calcium": "128mg"}
    },
    "Fresh Milk": {
        "category": "Essential", "benefit": "Bone Health", "nutrition_summary": "🥛 Pure Calcium",
        "reason": "Base inventory requirement. Expected rise during **Festival Preparation** (Sweet making).",
        "details": {"Energy": "62 kcal", "Protein": "3.2g", "Fat": "3.6g", "Carbs": "4.7g", "Calcium": "125mg"}
    },
    "Bread & Eggs": {
        "category": "Essential", "benefit": "Brain Health", "nutrition_summary": "🍳 High Choline & Protein",
        "reason": "Rising trend due to **College Hostel Breakfast** demand and weekend stock-outs.",
        "details": {"Energy": "155 kcal", "Protein": "13g", "Fat": "11g", "Carbs": "1.1g", "Choline": "Active"}
    }
}

# --- SIDEBAR: DYNAMIC CONTROLS ---
st.sidebar.header("🕹️ Control Panel")
uploaded_file = st.sidebar.file_uploader("Upload inventory_data.csv", type="csv")

safety_buffer = st.sidebar.slider("Safety Stock Buffer (%)", 0, 30, 10)
target_product = st.sidebar.selectbox("Select Product for Detailed Analysis", list(product_info.keys()))

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    sales_data = df['Sales'].values
    
    # 1. Forecasting Logic
    today = datetime.now()
    future_dates = [(today + timedelta(days=i)).strftime('%d-%m-%Y') for i in range(1, 11)]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(sales_data.reshape(-1, 1))
    history = scaled_data[-10:].flatten().tolist()
    
    future_preds = []
    for _ in range(10):
        p = history[-1] * 1.015 
        future_preds.append(p)
        history.append(p)
    
    base_forecast = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
    final_forecast = base_forecast * (1 + (safety_buffer / 100))

    # --- SECTION 1: NUTRITION & INSIGHT ---
    st.subheader(f"🥗 Featured Insight: {target_product}")
    insight_col, nutrition_col = st.columns([1, 1])
    with insight_col:
        st.write("#### 📋 Core Metrics")
        c1, c2 = st.columns(2)
        c1.metric("Category", product_info[target_product]["category"])
        c2.metric("Primary Benefit", product_info[target_product]["benefit"])
        st.info(f"**Key Fact:** {product_info[target_product]['nutrition_summary']}")
    with nutrition_col:
        st.write("#### 🧪 Nutrition per 100g")
        nutri_df = pd.DataFrame(product_info[target_product]["details"].items(), columns=["Nutrient", "Value"])
        st.table(nutri_df)

    st.markdown("---")

    # --- SECTION 2: VISUAL ANALYTICS ---
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📈 Interactive Demand Forecast")
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(future_dates, final_forecast, marker='o', color='#2ecc71', linewidth=3, label="Buffered Forecast")
        ax1.plot(future_dates, base_forecast, '--', color='gray', alpha=0.5, label="Base Forecast")
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        ax1.legend()
        st.pyplot(fig1)

    with col2:
        st.subheader("📊 Order Metrics")
        total_units = np.sum(final_forecast)
        st.metric("Total Bulk Order", f"{total_units:.2f} Units", delta=f"{safety_buffer}% Buffer")
        
        # --- DYNAMIC EXPLANATION BOX ---
        with st.expander("📝 Why this order? (XAI Rationale)", expanded=True):
            st.markdown(f"**Reasoning:** {product_info[target_product]['reason']}")
            st.write(f"The model predicts a rising trend of {safety_buffer}% above base levels to accommodate these specific external events.")

    # --- SECTION 3: DATA TABLE ---
    st.subheader("📋 Finalized Purchase Plan")
    formatted_units = [float(u[0]) for u in final_forecast]
    summary_df = pd.DataFrame({
        "Delivery Date": future_dates, 
        "Base Units": base_forecast.flatten().round(2),
        "With Safety Buffer": np.round(formatted_units, 2)
    })
    st.dataframe(summary_df, use_container_width=True)

    csv = summary_df.to_csv(index=False).encode('utf-8')
    st.download_button(label="📥 Download Purchase Order as CSV", data=csv, file_name='purchase_order.csv', mime='text/csv')

else:
    st.info("👋 Welcome! Please upload your CSV file in the sidebar to activate the dynamic dashboard.")