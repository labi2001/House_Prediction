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
    st.sidebar.title("House Price Prediction")
    section = st.sidebar.radio(
        "Navigation",
        (   "About Project",
            "Data Overview",
            "Feature Engineering",
            "Model Development",
            "Model Evaluation",
            "Predicting with the Model"),
    )
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

def about_project():
    st.title("House Price Prediction App")
    st.write("""
            This app allows you to predict house prices based on various features of the dataset you uploaded. 
            The app leverages machine learning techniques to provide accurate price predictions using a Random Forest Regressor model.
         """)
    st.header("Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])
    if uploaded_file is not None:
        st.session_state["uploaded_file"] = uploaded_file
        st.success("File uploaded successfully!")

def data_overview():
    st.title("Data Overview")
    if "uploaded_file" in st.session_state and st.session_state["uploaded_file"] is not None:
        uploaded_file = st.session_state['uploaded_file']
        try: 
            data = pd.read_csv(uploaded_file)
            st.session_state["data"] = data
            st.write("### Dataset Preview")
            st.dataframe(data.head(5))

            st.write("### Missing Values in Data")
            st.write(data.isnull().sum())
            st.write("### Summary Statistics:")
            st.write(data.describe())

            st.write("### Correlation Matrix:")
            correlation_matrix = data.corr()
            st.write(correlation_matrix)
            plt.figure(figsize=(10, 6))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
            st.pyplot(plt)
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
    else:
        st.warning("Please upload a CSV file in the About section to proceed.")

def feature_engineering():
    st.title("Feature Engineering")
    if "data" in st.session_state:
        data = st.session_state["data"]
    st.write("### Select the target column (the value you want to predict):")
    target_column = st.selectbox("Target Column", data.columns)
    
    if target_column:
        X = data.drop(columns=[target_column])  
        y = data[target_column] 
        categorical_cols = X.select_dtypes(include=['object']).columns
        if not categorical_cols.empty:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        X['Price_per_sqft'] = y.values / X['Area']  
        X['Room_Area_Ratio'] = X['Area'] / (X['Room'] + 1)  
        skewed_cols = ['Area', 'Price_per_sqft']  
        for col in skewed_cols:
            X[col] = np.log1p(X[col])

        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        IQR = Q3 - Q1
        outliers = (X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))

        X = X[~outliers.any(axis=1)]
        y = y[X.index] 
        st.write("### Preprocessed Features (X):")
        st.dataframe(X.head())
        st.write("### Target Variable (y):")
        st.dataframe(y.head())
        st.session_state["X"] = X
        st.session_state["y"] = y

def model_development():
    st.title("Model Development")
    if "X" in st.session_state and "y" in st.session_state:
        X = st.session_state["X"]
        y = st.session_state["y"]
        
        st.header("Split Data into Training and Testing Sets")
        test_size = st.slider("Select Test Size (as %):", 10, 50, 20) / 100
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        st.session_state["X_test"] = X_test
        st.session_state["y_test"] = y_test

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
                st.session_state["manual_model"] = manual_model
                st.session_state["manual_mse"] = mean_squared_error(y_test, manual_model.predict(X_test))
                st.session_state["manual_r2"] = manual_model.score(X_test, y_test)
                st.session_state["y_pred_manual"] = manual_model.predict(X_test)

        st.header("Compare Models and Get Predictions")
        model_choice = st.radio("Select a Model for Evaluation and Predictions", ("GridSearchCV", "Manual Tuning"))
        st.session_state.model_choice = model_choice

        if model_choice == "GridSearchCV" and "grid_model" in st.session_state:
            st.write("### Selected Model: GridSearchCV")
            st.write("Best Parameters from GridSearchCV:")
            st.write(st.session_state["grid_params"])
            st.write(f"Mean Squared Error (MSE): {st.session_state['grid_mse']}")
            st.write(f"R² Score: {st.session_state['grid_r2']}")
            st.write("### Predictions on Test Data:")
            st.write(st.session_state["y_pred_grid"])  

        elif model_choice == "Manual Tuning" and "manual_model" in st.session_state:
            st.write("### Selected Model: Manual Tuning")
            st.write(f"Mean Squared Error (MSE): {st.session_state['manual_mse']}")
            st.write(f"R² Score: {st.session_state['manual_r2']}")
            st.write("### Predictions on Test Data:")
            st.write(st.session_state["y_pred_manual"]) 

        else:
            st.write("Train a model first to view metrics and predictions.")
    else:
        st.write("Please upload data and perform feature engineering first.")

def model_evaluation():
    X_test = st.session_state.get("X_test")
    y_test = st.session_state.get("y_test")
    if st.session_state["model_choice"] == "GridSearchCV":
        model = st.session_state.get("grid_model")
        y_pred = st.session_state.get("y_pred_grid")
    elif st.session_state["model_choice"] == "Manual Tuning":
        model = st.session_state.get("manual_model")
        y_pred = st.session_state.get("y_pred_manual")

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write("### Model Performance")
    st.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
    st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
    st.metric("R² Score", f"{r2:.2f}")

    st.write("### Residual Plot")
    residuals = y_test - y_pred
    fig, ax = plt.subplots()
    sns.histplot(residuals, kde=True, ax=ax)
    ax.set_title("Residuals Distribution")
    st.pyplot(fig)
    st.write("---")
    
    st.header("Results Visualization")
    results = pd.DataFrame({
        "Actual": y_test.values.flatten(),
        "Predicted": y_pred.flatten()})
    st.write("#### Prediction vs Actual Values")
    st.dataframe(results)

    fig, ax = plt.subplots()
    sns.scatterplot(x="Actual", y="Predicted", data=results, ax=ax)
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)
    st.write("---")

    st.header("Feature Importance")
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        importance_df = pd.DataFrame({
            "Feature": X_test.columns,
            "Importance": importance}).sort_values(by="Importance", ascending=False)
        st.write(importance_df)
        fig, ax = plt.subplots()
        sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax)
        ax.set_title("Feature Importance")
        st.pyplot(fig)
    st.write("---")

def predicting_with_model():
    st.title("House Price Prediction App")
    model_choice = st.session_state.get("model_choice")
    if model_choice == "GridSearchCV":
        model = st.session_state.get("grid_model")  
    elif model_choice == "Manual Tuning":
        model = st.session_state.get("manual_model") 
        
    if model is not None:
        Area = st.number_input("Area (in sqft)", value=1000)
        Room = st.number_input("Number of Rooms", value=3)
        Lon = st.number_input("Longitude", value=-122.4194)
        Lat = st.number_input("Latitude", value=37.7749)
        Price_per_sqft = st.number_input("Price per Square Foot ($)", value=200)
        Room_Area_Ratio = st.number_input("Room Area Ratio", value=1.2)

        input_data = pd.DataFrame([[Area, Room, Lon, Lat, Price_per_sqft, Room_Area_Ratio]],
                          columns=['Area', 'Room', 'Lon', 'Lat', 'Price_per_sqft', 'Room_Area_Ratio'])
        if st.button("Predict"):
            prediction = model.predict(input_data)
            st.success(f"Predicted Price: ${prediction[0]:,.2f}")
        with open('rf_model.pkl', 'wb') as f:
            pickle.dump(model, f)

        st.header("Export Predictions")
        if st.button("Download Predictions as CSV"):
            results = pd.DataFrame({"Predicted Price": model.predict(input_data)})
            results.to_csv("predictions.csv", index=False)
            st.write("Predictions saved as `predictions.csv`. Refresh page to reset.")
    else:
        st.error("Model not found. Please train and select a model first.")

if __name__ == "__main__":
    main()
