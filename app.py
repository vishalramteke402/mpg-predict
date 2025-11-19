import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.title("Auto MPG Prediction App 🚗")
st.write("Upload your cleaned Auto MPG dataset (CSV) to train and predict.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Clean text column
    if 'car name' in df.columns:
        df = df.drop(['car name'], axis=1)

    # Split features and target
    X = df.drop(['mpg'], axis=1)
    y = df['mpg']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    st.subheader("Select Model")
    model_choice = st.selectbox(
        "Choose regression model",
        ["Linear Regression", "Lasso", "Ridge", "ElasticNet", "Random Forest"]
    )

    if model_choice == "Linear Regression":
        model = LinearRegression()
    elif model_choice == "Lasso":
        alpha = st.slider("Lasso Alpha", 0.01, 1.0, 0.1)
        model = Lasso(alpha=alpha)
    elif model_choice == "Ridge":
        alpha = st.slider("Ridge Alpha", 0.01, 1.0, 0.1)
        model = Ridge(alpha=alpha)
    elif model_choice == "ElasticNet":
        alpha = st.slider("ElasticNet Alpha", 0.01, 1.0, 0.1)
        l1 = st.slider("ElasticNet L1 Ratio", 0.0, 1.0, 0.5)
        model = ElasticNet(alpha=alpha, l1_ratio=l1)
    elif model_choice == "Random Forest":
        n_estimators = st.slider("Number of Trees", 50, 300, 100)
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)

    # Fit model
    model.fit(X_train_scaled, y_train)
    st.success(f"{model_choice} model trained successfully!")

    # Evaluation
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("Model Performance")
    st.write(f"**MSE:** {mse}")
    st.write(f"**R² Score:** {r2}")

    st.subheader("Predict MPG for New Input")

    input_data = {}
    for col in X.columns:
        input_data[col] = st.number_input(f"Enter {col}", value=float(X[col].mean()))

    if st.button("Predict MPG"):
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        st.success(f"Predicted MPG: {prediction:.2f}")
