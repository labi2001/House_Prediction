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

def main():
    # Sidebar navigation
    st.sidebar.title("House Price Prediction")
    section = st.sidebar.radio(
        "Navigation",
        (
            "About Project",
            "Data Overview",
            "Feature Engineering",
            "Model Development",
            "Model Evaluation",
            "Predicting with the Model",
        ),
    )

    # Render the selected section
    if section == "About Project":
        about_project()
    elif section == "Data Overview":
        data_overview()
    elif section == "Feature Engineering":
        feature_engineering()
    elif section == "Model Development":
        model_development()
    elif section == "Model Evaluation":
        model_evaluation()
    elif section == "Predicting with the Model":
        predicting_with_model()

# SECTION 1
def about_project():
    st.title("House Price Prediction App")
    st.write("""
                This app allows you to predict house prices based on various features of the dataset you uploaded. The app leverages machine learning techniques to provide accurate price predictions using a **Random Forest Regressor** model.
         """)

    # Upload Dataset
    st.header("Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])
    
    # Store the uploaded file in session state
    if uploaded_file is not None:
        st.session_state["uploaded_file"] = uploaded_file
        st.success("File uploaded successfully!")

# SECTION 2
def data_overview():
    st.title("Data Overview")

    # Check if a file has been uploaded
    if "uploaded_file" in st.session_state and st.session_state["uploaded_file"] is not None:
        uploaded_file = st.session_state['uploaded_file']
            
        try: 
    # Load dataset
            data = pd.read_csv(uploaded_file)
            st.session_state["data"] = data

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
        except Exception as e:
            # Handle errors when reading or processing the file
            st.error(f"An error occurred while processing the file: {e}")
    else:
        # Notify the user to upload a file
        st.warning("Please upload a CSV file in the About section to proceed.")

# SECTION 3
def feature_engineering():
    st.title("Feature Engineering")

    if "data" in st.session_state:
        data = st.session_state["data"]

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

        # Store X and y in session state
        st.session_state["X"] = X
        st.session_state["y"] = y

def model_development():
    st.title("Model Development")
    if "X" in st.session_state and "y" in st.session_state:
        X = st.session_state["X"]
        y = st.session_state["y"]

        # Split Dataset
        st.header("Split Data into Training and Testing Sets")
        test_size = st.slider("Select Test Size (as %):", 10, 50, 20) / 100
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Store data in session_state for global access
        st.session_state["X_test"] = X_test
        st.session_state["y_test"] = y_test

        # GridSearchCV Tuning
        st.header("Find the Best Hyperparameters with GridSearchCV")
        if st.button("Run GridSearchCV"):
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            rf = RandomForestRegressor(random_state=42)
            grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1)
            grid_search.fit(X_train, y_train)

            # Save model, metrics, and best parameters in session state
            st.session_state["grid_model"] = grid_search.best_estimator_
            st.session_state["grid_mse"] = mean_squared_error(y_test, grid_search.best_estimator_.predict(X_test))
            st.session_state["grid_r2"] = grid_search.best_estimator_.score(X_test, y_test)
            st.session_state["y_pred_grid"] = grid_search.best_estimator_.predict(X_test)
            st.session_state["grid_params"] = grid_search.best_params_

        # Optional Manual Tuning
        st.header("Optionally Adjust Hyperparameters")
        if st.checkbox("Enable Manual Tuning"):
            max_depth = st.slider("Max Depth:", 2, 50, 10)
            n_estimators = st.slider("Number of Trees:", 10, 300, 100)
            min_samples_split = st.slider("Min Samples Split:", 2, 20, 2)
            min_samples_leaf = st.slider("Min Samples per Leaf:", 1, 20, 1)
            max_features = st.selectbox("Max Features:", ['sqrt', 'log2', None])

            if st.button("Train with Manual Parameters"):
                manual_model = RandomForestRegressor(
                    max_depth=max_depth,
                    n_estimators=n_estimators,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    random_state=42
                )
                manual_model.fit(X_train, y_train)
                # Save model and metrics in session state
                st.session_state["manual_model"] = manual_model
                st.session_state["manual_mse"] = mean_squared_error(y_test, manual_model.predict(X_test))
                st.session_state["manual_r2"] = manual_model.score(X_test, y_test)
                st.session_state["y_pred_manual"] = manual_model.predict(X_test)

        # Model Selection and Predictions
        st.header("Compare Models and Get Predictions")
        model_choice = st.radio("Select a Model for Evaluation and Predictions", ("GridSearchCV", "Manual Tuning"))

        # Store the model choice in st.session_state
        st.session_state.model_choice = model_choice

        if model_choice == "GridSearchCV" and "grid_model" in st.session_state:
            st.write("### Selected Model: GridSearchCV")
            st.write("Best Parameters from GridSearchCV:")
            st.write(st.session_state["grid_params"])
            st.write(f"Mean Squared Error (MSE): {st.session_state['grid_mse']}")
            st.write(f"R² Score: {st.session_state['grid_r2']}")
            st.write("### Predictions on Test Data:")
            st.write(st.session_state["y_pred_grid"])  # Use predictions for GridSearchCV

        elif model_choice == "Manual Tuning" and "manual_model" in st.session_state:
            st.write("### Selected Model: Manual Tuning")
            st.write(f"Mean Squared Error (MSE): {st.session_state['manual_mse']}")
            st.write(f"R² Score: {st.session_state['manual_r2']}")
            st.write("### Predictions on Test Data:")
            st.write(st.session_state["y_pred_manual"])  # Use predictions for Manual Tuning

        else:
            st.write("Train a model first to view metrics and predictions.")
    else:
        st.write("Please upload data and perform feature engineering first.")


def model_evaluation():
    """
    Evaluate the model performance using various metrics and visualizations.
    """
    # Retrieve necessary data from session_state
    X_test = st.session_state.get("X_test")
    y_test = st.session_state.get("y_test")
    
    # Check if the user selected GridSearchCV or Manual Tuning
    if st.session_state["model_choice"] == "GridSearchCV":
        model = st.session_state.get("grid_model")
        y_pred = st.session_state.get("y_pred_grid")
    elif st.session_state["model_choice"] == "Manual Tuning":
        model = st.session_state.get("manual_model")
        y_pred = st.session_state.get("y_pred_manual")

    # Performance Metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write("### Model Performance")
    st.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
    st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
    st.metric("R² Score", f"{r2:.2f}")

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
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        importance_df = pd.DataFrame({
            "Feature": X_test.columns,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        st.write(importance_df)

        fig, ax = plt.subplots()
        sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax)
        ax.set_title("Feature Importance")
        st.pyplot(fig)

    st.write("---")

    
def predicting_with_model():
    st.title("House Price Prediction App")
    # Retrieve model choice from session state
    model_choice = st.session_state.get("model_choice")
    
    # Retrieve the correct model (either GridSearchCV or manual model) from session_state
    if model_choice == "GridSearchCV":
        model = st.session_state.get("grid_model")  # GridSearchCV model
    elif model_choice == "Manual Tuning":
        model = st.session_state.get("manual_model")  # Manual tuning model
    
    # Check if model is available
    if model is not None:

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
#
        # If button is clicked, predict the result
        if st.button("Predict"):
            prediction = model.predict(input_data)
            st.success(f"Predicted Price: ${prediction[0]:,.2f}")
        
        # Saving the model after training
        with open('rf_model.pkl', 'wb') as f:
            pickle.dump(model, f)

            
        # Export Predictions
        st.header("Export Predictions")
        
        # Export predictions as CSV
        if st.button("Download Predictions as CSV"):
            results = pd.DataFrame({"Predicted Price": model.predict(input_data)})
            results.to_csv("predictions.csv", index=False)
            st.write("Predictions saved as `predictions.csv`. Refresh page to reset.")
    else:
        st.error("Model not found. Please train and select a model first.")


if __name__ == "__main__":
    main()
