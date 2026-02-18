import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go

# ======================
# CACHED MODEL LOADING FUNCTION
# ======================
@st.cache_resource
def load_model_and_data():
    # Load dataset
    df = pd.read_csv("student_performance_extended.csv")

    # Encode categorical features safely
    df["family_support"] = df["family_support"].map({"Yes": 1, "No": 0})
    df["internet_access"] = df["internet_access"].map({"Yes": 1, "No": 0})
    df["extracurricular"] = df["extracurricular"].map({"Yes": 1, "No": 0})
    df["gender"] = df["gender"].map({"M": 1, "F": 0})

    # Handle missing data
    df = df.fillna(0)

    X = df.drop(columns=["student_id", "pass"], errors="ignore")
    y = df["pass"]

    # Train model
    model = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced"
    )
    model.fit(X, y)

    return model, X.columns.tolist()


# Load model and column names once
model, feature_columns = load_model_and_data()

# ======================
# Streamlit Layout
# ======================
st.set_page_config(page_title="Student Performance Predictor", layout="wide")
st.title("🎓 Student Performance Prediction (Pass/Fail)")
st.markdown("Enter student details in the sidebar to predict the pass/fail outcome.")

# ======================
# Sidebar Inputs
# ======================
st.sidebar.header("🧍 Student Details Input")

gender = st.sidebar.selectbox("Gender", ["M", "F"])
age = st.sidebar.slider("Age", 17, 25, 18)
study_hours = st.sidebar.slider("Study Hours per Week", 0, 40, 10)
attendance = st.sidebar.slider("Attendance %", 40, 100, 75)
prior_grade = st.sidebar.slider("Prior Grade (0-100)", 0, 100, 60)
assignments = st.sidebar.slider("Assignments Submitted (0-10)", 0, 10, 5)
family_support = st.sidebar.selectbox("Family Support", ["Yes", "No"])
internet = st.sidebar.selectbox("Internet Access", ["Yes", "No"])
extracurricular = st.sidebar.selectbox("Extracurricular Activities", ["Yes", "No"])
health = st.sidebar.slider("Health Score (0-100)", 0, 100, 70)
sleep = st.sidebar.slider("Sleep Hours per Day", 4, 10, 7)

# ======================
# Convert Input to DataFrame
# ======================
input_data = pd.DataFrame([{
    "gender": 1 if gender == "M" else 0,
    "age": age,
    "study_hours_per_week": study_hours,
    "attendance_percent": attendance,
    "prior_grade": prior_grade,
    "assignments_submitted": assignments,
    "family_support": 1 if family_support == "Yes" else 0,
    "internet_access": 1 if internet == "Yes" else 0,
    "extracurricular": 1 if extracurricular == "Yes" else 0,
    "health_score": health,
    "sleep_hours": sleep
}])

# ======================
# Input Summary Table
# ======================
st.subheader("📋 Entered Student Details")
st.table(input_data.rename(columns={
    "gender": "Gender",
    "age": "Age",
    "study_hours_per_week": "Study Hours/Week",
    "attendance_percent": "Attendance %",
    "prior_grade": "Prior Grade",
    "assignments_submitted": "Assignments Submitted",
    "family_support": "Family Support",
    "internet_access": "Internet Access",
    "extracurricular": "Extracurricular",
    "health_score": "Health Score",
    "sleep_hours": "Sleep Hours/Day"
}))

# ======================
# Prediction & Dashboard
# ======================
if st.button("Predict"):
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    # Determine PASS/FAIL color and confidence
    if pred == 1:
        status = "PASS"
        color = "green"
        confidence = prob
    else:
        status = "FAIL"
        color = "red"
        confidence = 1 - prob

    # Two-column layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(f"## 🎯 **Result: {status}**")
        st.markdown(f"### Confidence: {confidence * 100:.2f}%")

        # Gauge chart for probability
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            title={'text': "Pass Probability"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 50], 'color': 'lightcoral'},
                    {'range': [50, 100], 'color': 'lightgreen'}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Feature bar chart
        fig2 = go.Figure(go.Bar(
            x=input_data.values[0],
            y=input_data.columns,
            orientation='h',
            marker_color='teal'
        ))
        fig2.update_layout(
            title_text="📊 Student Feature Metrics",
            xaxis_title="Value",
            yaxis_title="Feature",
            height=450
        )
        st.plotly_chart(fig2, use_container_width=True)

# ======================
# Footer
# ======================
st.markdown("---")
st.markdown("© 2025 Student Performance Predictor | Built with Streamlit & Python")
