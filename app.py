import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# تحميل النموذج
model_path = 'E:/data science/projact/Cardiovascular Disease Prediction/random_forest_model.pkl'
model = joblib.load(model_path)

# عنوان التطبيق
st.title("Cardiovascular Disease Prediction")
st.write("This application predicts the risk of cardiovascular disease based on patient data.")

# الأعمدة التي تحتاج إلى التشفير
categorical_cols = ['General_Health', 'Checkup', 'Exercise', 'Skin_Cancer', 
                    'Other_Cancer', 'Depression', 'Diabetes', 'Arthritis', 
                    'Sex', 'Age_Category', 'Smoking_History']

# واجهة المستخدم لجمع المعلومات
st.sidebar.header("Patient Information")
general_health = st.sidebar.selectbox("General_Health", ["Excellent", "Very Good", "Good", "Fair", "Poor"])
checkup = st.sidebar.selectbox("Last Checkup", ["Within last year", "Within last 2 years", "Within last 5 years", "5 or more years ago", "Never"])
exercise = st.sidebar.selectbox("Exercise", ["No", "Yes"])
skin_cancer = st.sidebar.selectbox("Skin Cancer", ["No", "Yes"])
other_cancer = st.sidebar.selectbox("Other Cancer", ["No", "Yes"])
depression = st.sidebar.selectbox("Depression", ["No", "Yes"])
diabetes = st.sidebar.selectbox("Diabetes", ["No", "Pre-Diabetes", "Gestational Diabetes", "Yes"])
arthritis = st.sidebar.selectbox("Arthritis", ["No", "Yes"])
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
age_category = st.sidebar.selectbox("Age Category", ["18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75-84", "85+"])
smoking_history = st.sidebar.selectbox("Smoking History", ["Never", "Former", "Current"])

# إضافة الأعمدة المفقودة
alcohol_consumption = st.sidebar.selectbox("Alcohol Consumption", ["No", "Yes"])
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
fried_potato_consumption = st.sidebar.number_input("Fried Potato Consumption (times/week)", min_value=0, max_value=7, value=1)
fruit_consumption = st.sidebar.number_input("Fruit Consumption (times/day)", min_value=0, max_value=5, value=3)
green_vegetables_consumption = st.sidebar.number_input("Green Vegetables Consumption (times/day)", min_value=0, max_value=5, value=4)

# إنشاء DataFrame باستخدام البيانات المُدخلة
data_dict = {
    'General_Health': [general_health],
    'Checkup': [checkup],
    'Exercise': [exercise],
    'Skin_Cancer': [skin_cancer],
    'Other_Cancer': [other_cancer],
    'Depression': [depression],
    'Diabetes': [diabetes],
    'Arthritis': [arthritis],
    'Sex': [sex],
    'Age_Category': [age_category],
    'Smoking_History': [smoking_history],
    'Alcohol_Consumption': [alcohol_consumption],
    'BMI': [bmi],
    'FriedPotato_Consumption': [fried_potato_consumption],
    'Fruit_Consumption': [fruit_consumption],
    'Green_Vegetables_Consumption': [green_vegetables_consumption],
    'Heart_Disease': [0]  # إضافة العمود المفقود بالقيمة الافتراضية
}

df = pd.DataFrame(data_dict)

# تطبيق التشفير على الأعمدة التصنيفية
label_encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

# ترتيب الأعمدة ليكون مطابقًا لما تم استخدامه أثناء التدريب
trained_columns = ['General_Health', 'Checkup', 'Exercise', 'Heart_Disease', 'Skin_Cancer', 'Other_Cancer', 
                   'Depression', 'Diabetes', 'Arthritis', 'Sex', 'Age_Category', 'Smoking_History', 
                   'Alcohol_Consumption', 'BMI', 'FriedPotato_Consumption', 'Fruit_Consumption', 
                   'Green_Vegetables_Consumption']

df = df[trained_columns]

# زر التنبؤ
if st.button('Predict'):
    try:
        prediction = model.predict(df)
        risk = "High Risk" if prediction[0] == 1 else "Low Risk"
        st.success(f"The predicted cardiovascular disease risk: **{risk}**")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
