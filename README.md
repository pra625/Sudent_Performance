Student Performance Prediction
Objective

Predict whether a student will pass or fail based on study hours, attendance, prior grades, and lifestyle factors. Helps educators and parents identify students needing support.

Dataset

Synthetic dataset (student_performance_extended.csv, 500+ records) with features:

Feature	Description
student_id	Unique ID
gender	M/F
age	Student age
study_hours_per_week	Avg study hours/week
attendance_percent	Class attendance %
prior_grade	Previous grade (0-100)
assignments_submitted	Number of assignments
family_support	Yes/No
internet_access	Yes/No
extracurricular	Yes/No
health_score	0-100
sleep_hours	Avg sleep/day
pass	Target: 1=Pass, 0=Fail
Libraries

pandas, numpy → Data manipulation

scikit-learn → Random Forest Classifier

joblib → Save/load model

streamlit → Interactive dashboard

plotly → Charts (gauge, bar)

matplotlib → Optional visualizations

Installation
pip install pandas numpy scikit-learn joblib streamlit plotly matplotlib

Run Dashboard
streamlit run app.py


Open browser at http://localhost:8501.

Methodology

Encode categorical features (Yes/No, M/F)

Train Random Forest Classifier (100 estimators, balanced weights)

Deploy interactive Streamlit dashboard

Visualize predictions with Plotly charts

Features

gender, age, study_hours_per_week, attendance_percent, prior_grade, assignments_submitted, family_support, internet_access, extracurricular, health_score, sleep_hours

Results & Insights

High prediction accuracy

Key factors: study hours, attendance, prior grades

Real-time predictions via dashboard

Future Work

Use real-world datasets (UCI, Kaggle)

Deploy online (Streamlit Cloud)

Explainable AI (SHAP)

Compare with other ML models

Conclusion

ML can effectively predict student performance and serve as an early warning system for educators.