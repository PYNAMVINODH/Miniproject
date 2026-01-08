## Inventory Management Using XAI Forecasting
This project introduces an innovative inventory management system utilizing Long Short-Term Memory (LSTM) networks integrated with Explainable AI (XAI) frameworks. It transforms "black-box" deep learning predictions into justified, actionable business intelligence to optimize supply chain costs and customer satisfaction.

## About
<!--Detailed Description about the project-->
Traditional inventory forecasting often relies on manual statistical models that fail to handle market volatility, while modern AI models are often "opaque," leading to a Trust Gap among managers. This system bridges that gap by providing high-accuracy demand forecasting paired with a SHAP interpretability layer. By explaining the "why" behind every "reorder" suggestion—attributing spikes to factors like seasonal heatwaves or price drops—the system empowers warehouse managers to mitigate the Bullwhip Effect and reduce financial risks.

## Features
<!--List the features of the project as shown below-->
- LSTM Forecasting Engine: Captures complex, non-linear demand patterns and long-term seasonality.
- SHAP Interpretability Layer: Breaks down forecasts into human-readable "Feature Contributions".
- Automated Reasoning Reports: Generates alerts that explain the mathematical logic behind reorder suggestions to build user trust.
- Dynamic Streamlit Dashboard: A real-time interface for visualizing future demand curves alongside XAI-driven "Waterfall Plots".
- Nutritional Intelligence Integration: Links product inventory to health-critical events (e.g., prioritizing Butter Milk during heatwaves).

## Requirements
<!--List the requirements of the project as shown below-->
*Operating System: Windows 10/11 or Linux (Ubuntu).
* Development Environment: Python 3.8 or later.
* Processor: Intel Core i5 (8th Gen) or higher.
* Memory: 8 GB minimum (16 GB recommended).
* Deep Learning Frameworks: TensorFlow, Keras, or PyTorch for the LSTM engine.
* Explainable AI: SHAP (SHapley Additive exPlanations) for model transparency.
* Data & Visualization: Pandas, NumPy, Matplotlib, and Plotly.
* Deployment: Streamlit for the interactive web-based dashboard.

## System Architecture
<!--Embed the system architecture diagram as shown below-->
The architecture consists of a Data Ingestion Layer, a Data Preprocessing/Feature Engineering module (handling normalization and time-lags), the LSTM Deep Learning Model, and the SHAP Explainer Module that feeds into an interactive Streamlit Dashboard.

<img width="1920" height="1080" alt="Screenshot 2026-01-07 100745" src="https://github.com/user-attachments/assets/fe619bf0-144c-487f-aab0-723161c54093" />


<img width="794" height="365" alt="Screenshot 2026-01-07 104729" src="https://github.com/user-attachments/assets/cb484e11-a6d6-4f8c-9e17-0ad5f15bd20b" />



## Output

<!--Embed the Output picture at respective places as shown below as shown below-->
#### Output1 - Featured Insights

<img width="802" height="368" alt="Screenshot 2026-01-07 104617" src="https://github.com/user-attachments/assets/d9c7cea8-964f-4e1b-93e3-4b1ed351502f" />


#### Output2 - Finalized Purchase Plan
<img width="803" height="395" alt="Screenshot 2026-01-07 104856" src="https://github.com/user-attachments/assets/3405b819-656e-4f35-aae0-2b210ff4f384" />


Forecast Accuracy: Achieved a Mean Absolute Percentage Error (MAPE) of 0.06 (94%) for seasonal demand and 0.08 (92%) for promotional spikes.


## Results and Impact
<!--Give the results and impact as shown below-->
The system achieves exceptional predictive accuracy and high computational efficiency, generating a 30-day forecast in under 0.5 seconds. By providing transparent, data-driven insights, it minimizes "over-ordering" of perishables, directly preventing food spoilage and resource waste in alignment with Sustainable Development Goal (SDG) 12.

## Articles published / References
1. S. Hochreiter and J. Schmidhuber, "Long short-term memory," Neural Computation, 1997.
2. S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," NIPS, 2017.
3. H. L. Lee et al., "Information distortion in a supply chain: The bullwhip effect," Management Science, 1997.




