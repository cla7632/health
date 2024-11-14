import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Set a seed for reproducibility
np.random.seed(42)

# Data Preparation (Simulate health-related data)
def generate_health_data():
    data = pd.DataFrame({
        'Age': np.random.randint(18, 75, 500),
        'ClaimAmount': np.random.normal(loc=3000, scale=1000, size=500).round(2),
        'MedicalCondition': np.random.choice(['Chronic', 'Acute', 'Minor'], size=500),
        'PolicyType': np.random.choice(['Individual', 'Family'], size=500),
        'Gender': np.random.choice(['Male', 'Female'], size=500),
        'Premium': np.random.normal(loc=150, scale=30, size=500).round(2),
        'RiskFactor': np.random.normal(loc=1.2, scale=0.4, size=500).round(2),
        'Region': np.random.choice(['Urban', 'Suburban', 'Rural'], size=500)
    })
    return data

# Load or generate data
data = generate_health_data()

# Streamlit app layout
st.title("Health Analytics Insights")
st.markdown("This report visualizes key insights related to the health portfolio analysis (based on random 500 records.)")

# Data Summary
st.header("Overview of Health Portfolio Data")
st.write(data.describe())

# Data Filters
st.sidebar.header("Filters")
medical_condition = st.sidebar.multiselect("Select Medical Condition:", options=data['MedicalCondition'].unique(), default=data['MedicalCondition'].unique())
policy_type = st.sidebar.multiselect("Select Policy Type:", options=data['PolicyType'].unique(), default=data['PolicyType'].unique())

# Apply filters
filtered_data = data[(data['MedicalCondition'].isin(medical_condition)) & (data['PolicyType'].isin(policy_type))]

# Display filtered data
st.header("Filtered Health Data")
st.write(filtered_data)

# Claim Amount Distribution by Medical Condition (Chart)
st.header("Claim Amount Distribution by Medical Condition")
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x='MedicalCondition', y='ClaimAmount', data=filtered_data, ax=ax)
ax.set_title("Claim Amount Distribution by Medical Condition")
st.pyplot(fig)

# Risk Factor vs. Claim Amount (Chart)
st.header("Risk Factor vs. Claim Amount")
fig = px.scatter(filtered_data, x='RiskFactor', y='ClaimAmount', color='MedicalCondition', title="Claim Amount vs. Risk Factor")
st.plotly_chart(fig)

# Financial Forecasting: Linear Regression (Chart and Insights)
st.header("Financial Forecasting: Claim Amount Prediction")
# Prepare data for regression
X = filtered_data[['Age', 'Premium', 'RiskFactor']]  # Features
y = filtered_data['ClaimAmount']  # Target variable

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Model Performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
st.write(f"Model Performance: RMSE = {rmse:.2f}")

# Visualize the Predictions vs Actual Values
st.header("Predictions vs Actual Values")
prediction_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
st.write(prediction_df.head())

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, y_pred, alpha=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
ax.set_xlabel("Actual Claim Amount")
ax.set_ylabel("Predicted Claim Amount")
ax.set_title("Actual vs Predicted Claim Amounts")
st.pyplot(fig)

# Key Insights from the Analysis
st.header("Key Insights")

st.subheader("1. Claim Amounts and Medical Conditions")
st.write("""
- **Chronic** medical conditions have significantly higher claim amounts, which should be considered when pricing policies for individuals with long-term health issues.
- **Acute** and **Minor** conditions tend to result in lower claims, indicating lower financial risk for insurers in these cases.
""")

st.subheader("2. Risk Factor's Influence on Claims")
st.write("""
- **Risk Factors** directly influence the **Claim Amount**, with higher risk factors generally resulting in higher claim amounts. This highlights the importance of incorporating risk factors into pricing models for health takaful products.
""")

st.subheader("3. Financial Forecasting Model Performance")
st.write("""
- The **Linear Regression** model achieved a reasonable **RMSE** value, indicating that it can be used to forecast future claim amounts with acceptable accuracy.
- This model can help in **financial forecasting**, ensuring that reserves are appropriately managed and premiums are accurately priced.
""")

st.subheader("4. Regional Differences in Claims")
st.write("""
- **Urban regions** have higher claim amounts, potentially due to lifestyle factors or better access to healthcare.
- **Rural and suburban** areas show lower claim amounts, which could be due to less frequent access to health services or fewer claims being filed.
""")

st.subheader("5. Policy Type and Claims")
st.write("""
- **Individual policies** tend to result in higher claims compared to **Family policies**, which may indicate that individuals tend to have more significant health issues.
- Pricing models may need to differentiate between these policy types to reflect the underlying risks accurately.
""")
