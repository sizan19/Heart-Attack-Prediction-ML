"""
Heart Disease Prediction Web Application
Built with Streamlit
"""

import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# Load the trained model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    return model_data

# Main application
def main():
    # Title and description
    st.title("❤️ Heart Disease Prediction System")
    st.markdown("""
    This application predicts the risk of heart disease based on clinical parameters.
    Please enter the patient's information below.
    """)

    # Load model
    try:
        model_data = load_model()
        model = model_data['model']
        label_encoder = model_data['label_encoder']
        feature_names = model_data['feature_names']
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # Create two columns for input
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Patient Demographics")
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=55, step=1)
        gender = st.selectbox("Gender", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])

        st.subheader("Vital Signs")
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=30, max_value=200, value=80, step=1)
        systolic_bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=70, max_value=250, value=120, step=1)
        diastolic_bp = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=40, max_value=150, value=80, step=1)

    with col2:
        st.subheader("Laboratory Results")
        blood_sugar = st.number_input("Blood Sugar (mg/dL)", min_value=50, max_value=500, value=100, step=1)
        ck_mb = st.number_input("CK-MB (ng/mL)", min_value=0.0, max_value=100.0, value=3.0, step=0.1, format="%.1f")
        troponin = st.number_input("Troponin (ng/mL)", min_value=0.0, max_value=10.0, value=0.05, step=0.01, format="%.2f")

    # Prediction button
    st.markdown("---")
    if st.button("🔍 Predict Heart Disease Risk", type="primary", use_container_width=True):
        # Prepare input data
        # Feature order: Age, Gender, Heart rate, Systolic BP, Diastolic BP, Blood sugar, CK-MB, Troponin
        input_data = np.array([[
            age,
            gender[1],  # Get the numeric value (1 for Male, 0 for Female)
            heart_rate,
            systolic_bp,
            diastolic_bp,
            blood_sugar,
            ck_mb,
            troponin
        ]])

        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]

        # Get the label
        result_label = label_encoder.inverse_transform([prediction])[0]

        # Display results
        st.markdown("---")
        st.subheader("Prediction Results")

        # Create columns for results
        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            st.metric("Prediction", result_label.upper())

        with res_col2:
            negative_prob = prediction_proba[0] * 100
            st.metric("Negative Risk", f"{negative_prob:.2f}%")

        with res_col3:
            positive_prob = prediction_proba[1] * 100
            st.metric("Positive Risk", f"{positive_prob:.2f}%")

        # Visual indicator
        if result_label.lower() == 'positive':
            st.error("⚠️ HIGH RISK: This patient shows indicators of potential heart disease. Please consult with a healthcare professional immediately.")
        else:
            st.success("✅ LOW RISK: This patient shows lower indicators of heart disease. However, regular health checkups are still recommended.")

        # Display input summary
        with st.expander("📋 View Input Summary"):
            input_df = pd.DataFrame({
                'Parameter': ['Age', 'Gender', 'Heart Rate', 'Systolic BP', 'Diastolic BP', 'Blood Sugar', 'CK-MB', 'Troponin'],
                'Value': [age, gender[0], f"{heart_rate} bpm", f"{systolic_bp} mmHg", f"{diastolic_bp} mmHg", f"{blood_sugar} mg/dL", f"{ck_mb} ng/mL", f"{troponin} ng/mL"]
            })
            st.table(input_df)

    # Information section
    st.markdown("---")
    with st.expander("ℹ️ About this Application"):
        st.markdown("""
        ### Model Information
        - **Algorithm**: Random Forest Classifier
        - **Accuracy**: 97.35%
        - **F1 Score**: 97.86%

        ### Features Used
        - **Age**: Patient's age in years
        - **Gender**: Male (1) or Female (0)
        - **Heart Rate**: Beats per minute
        - **Systolic Blood Pressure**: Upper blood pressure reading
        - **Diastolic Blood Pressure**: Lower blood pressure reading
        - **Blood Sugar**: Blood glucose level
        - **CK-MB**: Creatine Kinase-MB (cardiac marker)
        - **Troponin**: Cardiac troponin level (cardiac marker)

        ### Important Notes
        - This is a predictive tool and should not replace professional medical advice
        - Always consult with healthcare professionals for proper diagnosis and treatment
        - The model is trained on historical data and predictions should be validated clinically
        """)

if __name__ == "__main__":
    main()
