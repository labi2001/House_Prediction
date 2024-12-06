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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV
import pickle

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
        y = data[target_column]  # Target

        # Encode categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns
        if not categorical_cols.empty:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        # Create new features
        X['Price_per_sqft'] = y.values / X['Area']  # Example: Price per square foot
        X['Room_Area_Ratio'] = X['Area'] / (X['Room'] + 1)  # Avoid division by zero

        # Log transformation for skewed features
        skewed_cols = ['Area', 'Price_per_sqft']  # Replace with actual skewed columns
        for col in skewed_cols:
            X[col] = np.log1p(X[col])

        # Scale the features
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        # Remove outliers from X using IQR
        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        IQR = Q3 - Q1
        outliers = (X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))

        # Drop the rows with outliers in X and corresponding rows in y
        X = X[~outliers.any(axis=1)]
        y = y[X.index]  # Ensure y matches X's index after outlier removal

        st.write("### Preprocessed Features (X):")
        st.dataframe(X.head())
        st.write("### Target Variable (y):")
        st.dataframe(y.head())

else:
    st.write("Please upload a CSV file to proceed.")

st.write("---")

# Check the shape of X and y
st.write(f"Features Shape (X): {X.shape}")
st.write(f"Target Shape (y): {y.shape}")


# Splitting Dataset
st.header("Split Data into Training and Testing Sets")
test_size = st.slider("Select Test Size (as %):", 10, 50, 20) / 100
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
st.write(f"Training Features Shape (X_train): {X_train.shape}")
st.write(f"Testing Features Shape (X_test): {X_test.shape}")

st.write("---")

# Model Training
# Section 1: GridSearchCV to find the best hyperparameters
st.header("Find the Best Hyperparameters with GridSearchCV")
if st.button("Run GridSearchCV"):
    # Define the parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }

    # Initialize RandomForestRegressor (no hyperparameters yet)
    rf = RandomForestRegressor(random_state=42)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1)

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    # Get and display the best parameters from grid search
    best_params = grid_search.best_params_
    st.write("Best Parameters from Grid Search:")
    st.write(best_params)

    # Use the best parameters to initialize the RandomForestRegressor
    model = grid_search.best_estimator_

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate and display the performance metrics (MSE and R²)
    mse_forgrid = mean_squared_error(y_test, y_pred)
    r2_forgrid= model.score(X_test, y_test)

    st.write("---")
    st.write(f"Mean Squared Error (MSE) from GridSearch CV: {mse_forgrid}")
    st.write(f"R² Score: {r2_forgrid}")

# Section 2: Manual Adjustment of Hyperparameters Using Sliders
st.header("Manually Adjust Hyperparameters")
max_depth = st.slider("Max Depth of Random Forest:", 2, 50, 10)
n_estimators = st.slider("Number of Trees:", 10, 300, 100)
min_samples_split = st.slider("Minimum Samples Split:", 2, 20, 2)
min_samples_leaf = st.slider("Minimum Samples per Leaf:", 1, 20, 1)
max_features = st.selectbox("Max Features:", ['sqrt', 'log2', None])

# Train the model with manually adjusted hyperparameters
st.header("Train the Model with Manually Adjusted Hyperparameters")
model = RandomForestRegressor(
    max_depth=max_depth,
    n_estimators=n_estimators,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    max_features=max_features,
    random_state=42
)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate and display the performance metrics (MSE and R²)
mse = mean_squared_error(y_test, y_pred)
r2 = model.score(X_test, y_test)

st.write(f"Mean Squared Error (MSE): {mse}")
st.write(f"R² Score: {r2}")


# Performance Metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.write("### Model Performance")
st.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
st.metric("R2 Score", f"{r2:.2f}")

# Residual Plot
st.write("### Residual Plot")
residuals = y_test - y_pred
fig, ax = plt.subplots()
sns.histplot(residuals, kde=True, ax=ax)
ax.set_title("Residuals Distribution")
st.pyplot(fig)

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

# After training your model, save it
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load the trained model
with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("House Price Prediction App")

# Input form for independent variables
Area = st.number_input("Area (in sqft)", value=1000)
Room = st.number_input("Number of Rooms", value=3)
Lon = st.number_input("Longitude", value=-122.4194)
Lat = st.number_input("Latitude", value=37.7749)
Price_per_sqft = st.number_input("Price per Square Foot ($)", value=200)
Room_Area_Ratio = st.number_input("Room Area Ratio", value=1.2)


# Create a DataFrame for the input values
input_data = pd.DataFrame([[Area, Room, Lon, Lat, Price_per_sqft, Room_Area_Ratio]],
                          columns=['Area', 'Room', 'Lon', 'Lat', 'Price_per_sqft', 'Room_Area_Ratio'])
# Predict the price when the button is pressed
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Price: ${prediction[0]:,.2f}")
