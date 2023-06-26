# import libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# load model from pickle file
with open("your_model.pkl", "rb") as f:
    rf = pickle.load(f)

# create sidebar title
st.sidebar.title("Random Forest Parameters")

# create sidebar input for n_estimators
n_estimators = st.sidebar.slider("Number of trees", 1, 1000)

# create sidebar button for model update
update_model = st.sidebar.button("Update Model")

# create main title
st.title("Random Forest Regression")

# display data upload option
st.write("Upload your data here:")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# load and display data if uploaded
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    X = df.drop("target", axis=1)
    y = df["target"]
    st.dataframe(df)
else:
    st.write("No data uploaded yet.")

# update model and display results if data is uploaded and button is clicked
if uploaded_file is not None and update_model:
    # show spinner while model is updating
    with st.spinner("Updating model..."):
        # create and fit new model with updated parameters
        rf = RandomForestRegressor(n_estimators=n_estimators)
        rf.fit(X, y)
        # save updated model to pickle file
        with open("your_model.pkl", "wb") as f:
            pickle.dump(rf, f)
    # show success message
    st.success("Model updated!")
else:
    st.write("No model update requested yet.")

# create inference title
st.title("Inference")

# display inference options
st.write("Choose one of the following options for inference:")
option = st.radio("Select an option", ("Enter values manually", "Upload a CSV file"))

# display manual input option
if option == "Enter values manually":
    # create input fields for each feature
    inputs = []
    for col in X.columns:
        value = st.number_input(f"Enter value for {col}", value=0.0)
        inputs.append(value)
    # create button for prediction
    predict = st.button("Predict")
    # make prediction and display result if button is clicked
    if predict:
        # convert inputs to numpy array and reshape to match model input shape
        inputs = np.array(inputs).reshape(1, -1)
        # make prediction using loaded model
        y_pred = rf.predict(inputs)
        # display prediction result
        st.write(f"The predicted target value is {y_pred[0]:.2f}")

# display csv upload option
if option == "Upload a CSV file":
    # display file upload option
    st.write("Upload your CSV file here:")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    # load and display data if uploaded
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        X_new = df.values
        st.dataframe(df)
    else:
        st.write("No CSV file uploaded yet.")
    # create button for prediction
    predict = st.button("Predict")
    # make prediction and display result if button is clicked and data is uploaded
    if uploaded_file is not None and predict:
        # make prediction using loaded model
        y_pred = rf.predict(X_new)
        # display prediction result as dataframe
        df["prediction"] = y_pred.round(2)
        st.dataframe(df)
