# Importing required libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer

# App Header
st.title("House Price Prediction App")
st.write("""
This app allows you to predict house prices based on various features of the dataset you uploaded. The app leverages machine learning techniques to provide accurate price predictions using a **Random Forest Regressor** model.
         """)

st.write("---")

# Upload Dataset
st.header("Upload Your Dataset")
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    data = pd.read_csv(uploaded_file)
    
    # Dataset Preview
    st.write("### Dataset Preview")
    st.dataframe(data.head(5))

    # Check for missing values
    st.write("### Missing Values in Data")
    st.write(data.isnull().sum())

    # Display summary statistics
    st.write("### Summary Statistics:")
    st.write(data.describe())

    # Display Correlation Matrix
    st.write("### Correlation Matrix:")
    correlation_matrix = data.corr()
    st.write(correlation_matrix)

    # Plot the Correlation Matrix using Seaborn
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    st.pyplot(plt)

    st.write("---")

    # Let user select the target column
    st.write("### Select the target column (the value you want to predict):")
    target_column = st.selectbox("Target Column", data.columns)

    # Preprocess Data
    if target_column:
        X = data.drop(columns=[target_column])  # Features
        y = data[[target_column]]  # Target
        
        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        st.write("### Preprocessed Features (X):")
        st.dataframe(X.head())
        st.write("### Target Variable (y):")
        st.dataframe(y.head())
else:
    st.write("Please upload a CSV file to proceed.")

st.write("---")
# Splitting Dataset
st.header("Split Data into training and testing sets")
test_size = st.slider("Select Test Size (as %):", 10, 50, 20) / 100
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
st.write(f"Training Features Shape (X_train): {X_train.shape}")
st.write(f"Testing Features Shape (X_test): {X_test.shape}")

st.write("---")

# Model Training
st.header("Train Model")
max_depth = st.slider("Max Depth of Random Forest:", 2, 20, 10)
n_estimators = st.slider("Number of Trees:", 10, 200, 100)
model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

st.write("---")

# Performance Metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.write("### Model Performance")
st.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
st.metric("R2 Score", f"{r2:.2f}")

st.write("---")
 # Visualization
st.header("Results Visualization")
results = pd.DataFrame({
     "Actual": y_test.values.flatten(),
     "Predicted": y_pred.flatten()
     })

st.write("#### Prediction vs Actual Values")
st.dataframe(results)

fig, ax = plt.subplots()
sns.scatterplot(x="Actual", y="Predicted", data=results, ax=ax)
ax.set_title("Actual vs Predicted")
st.pyplot(fig)

st.write("---")

# Feature Importance
st.header("Feature Importance")
importance = model.feature_importances_
importance_df = pd.DataFrame({
     "Feature": X.columns,
     "Importance": importance
     }).sort_values(by="Importance", ascending=False)

st.write(importance_df)

fig, ax = plt.subplots()
sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax)
ax.set_title("Feature Importance")
st.pyplot(fig)

st.write("---")

# Export Predictions
st.header("Export Predictions")
if st.button("Download Predictions as CSV"):
    results.to_csv("predictions.csv", index=False)
    st.write("Predictions saved as `predictions.csv`. Refresh page to reset.")

