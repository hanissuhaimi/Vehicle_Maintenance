import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
@st.cache_resource
def load_model_and_encoders():
    model = joblib.load('service_prediction_model.pkl')
    encoders = joblib.load('label_encoders.pkl')
    df = pd.read_csv("vehicle_data.csv")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Recompute problem_frequency if not present
    df['common_problem'] = df['common_problem'].astype(str)
    if 'problem_frequency' not in df.columns:
        problem_counts = df.groupby('common_problem').size().reset_index(name='problem_frequency')
        df = pd.merge(df, problem_counts, on='common_problem', how='left')

    return model, encoders, df

model, encoders, df = load_model_and_encoders()

st.title("Vehicle Service Prediction Tool")
st.markdown("Predict the recommended service based on mileage and known problems.")

with st.form("service_form"):
    mileage_last_service = st.number_input("Last service mileage (km)", min_value=0, value=10000)
    current_mileage = st.number_input("Current mileage (km)", min_value=0, value=15000)
    common_problem = st.text_input("Describe the common problem", value="Engine noise")

    submitted = st.form_submit_button("Predict Service")

if submitted:
    try:
        mileage_diff = current_mileage - mileage_last_service
        service_interval_ratio = mileage_diff / (mileage_last_service + 1)

        # Handle problem frequency safely
        if common_problem in df['common_problem'].values:
            problem_frequency = df[df['common_problem'] == common_problem]['problem_frequency'].mean()
        elif 'problem_frequency' in df.columns:
            problem_frequency = df['problem_frequency'].mean()
        else:
            problem_frequency = 0.0

        if common_problem in encoders['common_problem'].classes_:
            common_problem_encoded = encoders['common_problem'].transform([common_problem])[0]
        else:
            common_problem_encoded = df['common_problem_encoded'].mode()[0]

        features = [[mileage_last_service, current_mileage, mileage_diff,
                     service_interval_ratio, problem_frequency, common_problem_encoded]]

        predicted_service_code = model.predict(features)[0]
        predicted_service = encoders['primary_service'].inverse_transform([predicted_service_code])[0]

        st.success(f"Recommended Service: {predicted_service}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

 
