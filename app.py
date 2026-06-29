import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Page config
st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺")

st.title("🩺 Diabetes Prediction App")
st.write("This app predicts whether a person has diabetes based on health metrics.")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv('diabetes.csv')

df = load_data()

# Train model
@st.cache_resource
def train_model(df):
    X = df.drop(columns='Outcome')
    Y = df['Outcome']
    
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, stratify=Y, random_state=2)
    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train, Y_train)
    
    train_acc = accuracy_score(classifier.predict(X_train), Y_train)
    test_acc = accuracy_score(classifier.predict(X_test), Y_test)
    
    return classifier, scaler, train_acc, test_acc

classifier, scaler, train_acc, test_acc = train_model(df)

st.write(f"Model trained - Training Accuracy: {train_acc:.2%}, Test Accuracy: {test_acc:.2%}")

with st.form("prediction_form"):
    st.header("Enter your health metrics")
    
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
        glucose = st.number_input("Glucose", min_value=0, max_value=200, value=120)
        blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
        skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    
    with col2:
        insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
    
    submitted = st.form_submit_button("Predict")
    
    if submitted:
        input_data = (pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age)
        input_arr = np.asarray(input_data).reshape(1, -1)
        std_arr = scaler.transform(input_arr)
        pred = classifier.predict(std_arr)[0]

        st.subheader("Prediction Result")
        if pred == 0:
            st.success("✅ The person is **not diabetic**.")
        else:
            st.error("⚠️ The person is **diabetic**.")