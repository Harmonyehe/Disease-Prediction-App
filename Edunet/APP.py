import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import joblib

# Set page configuration
st.set_page_config(page_title="Multi-Disease Prediction App", layout="wide", page_icon="❤️")

# Load models function
@st.cache_resource()
def load_models():
    # Define model paths (ensure files are saved in the "models" folder in the project directory)
    working_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(working_dir, 'saved_models')

    diabetes_model_path = os.path.join(models_dir, 'diabetes_model_rf.sav')
    heart_disease_model_path = os.path.join(models_dir, 'heart_disease_model.sav')
    parkinsons_model_path = os.path.join(models_dir, 'parkinsons_model_improved.sav')

    diabetes_model, diabetes_scaler = joblib.load(diabetes_model_path)
    heart_disease_model, heart_scaler = joblib.load(heart_disease_model_path)
    parkinsons_data = pickle.load(open(parkinsons_model_path, 'rb'))
    
    return (diabetes_model, diabetes_scaler), (heart_disease_model, heart_scaler), parkinsons_data

# Load models
(diabetes_model, diabetes_scaler), (heart_disease_model, heart_scaler), parkinsons_model_data = load_models()
parkinsons_model, parkinsons_scaler = parkinsons_model_data['model'], parkinsons_model_data['scaler']

# Sidebar navigation
with st.sidebar:
    selected = option_menu('Disease Prediction System',
                           ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
                           menu_icon='stethoscope',
                           icons=['activity', 'heart-pulse', 'person'],
                           default_index=0)

# Input helper
def get_user_input(fields):
    user_input = []
    for field in fields:
        value = st.text_input(field['label'])
        if value:
            try:
                user_input.append(float(value))
            except ValueError:
                st.error(f"Please enter a valid number for {field['label']}.")
                return None
        else:
            st.error(f"Please enter a value for {field['label']}.")
            return None
    return user_input

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction')
    diabetes_fields = [
        {'label': 'Number of Pregnancies'}, {'label': 'Glucose Level'},
        {'label': 'Blood Pressure'}, {'label': 'Skin Thickness'},
        {'label': 'Insulin Level'}, {'label': 'BMI'},
        {'label': 'Diabetes Pedigree Function'}, {'label': 'Age'}
    ]
    user_input = get_user_input(diabetes_fields)
    if user_input and st.button('Get Diabetes Prediction'):
        scaled_input = diabetes_scaler.transform([user_input])
        result = diabetes_model.predict(scaled_input)[0]
        st.success('Diabetic' if result == 1 else 'Not Diabetic')

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction')
    heart_disease_fields = [
        {'label': 'Age'}, {'label': 'Sex (1=Male, 0=Female)'},
        {'label': 'Chest Pain Type (0-3)'}, {'label': 'Resting Blood Pressure'},
        {'label': 'Serum Cholestoral'}, {'label': 'Fasting Blood Sugar (1=True, 0=False)'},
        {'label': 'Resting ECG'}, {'label': 'Max Heart Rate'},
        {'label': 'Exercise Induced Angina (1=True, 0=False)'},
        {'label': 'ST Depression'}, {'label': 'Slope'},
        {'label': 'Major Vessels (0-3)'}, {'label': 'Thal (0=Normal, 1=Fixed, 2=Reversible)'}
    ]
    user_input = get_user_input(heart_disease_fields)
    if user_input and st.button('Get Heart Disease Prediction'):
        scaled_input = heart_scaler.transform([user_input])
        result = heart_disease_model.predict(scaled_input)[0]
        st.success('Heart Disease Detected' if result == 1 else 'No Heart Disease')

# Parkinson's Prediction Page
if selected == 'Parkinsons Prediction':
    st.title("Parkinson's Disease Prediction")
    parkinsons_fields = [
        {'label': 'MDVP:Fo(Hz)'}, {'label': 'MDVP:Fhi(Hz)'}, {'label': 'MDVP:Flo(Hz)'},
        {'label': 'MDVP:Jitter(%)'}, {'label': 'MDVP:Jitter(Abs)'}, {'label': 'MDVP:RAP'},
        {'label': 'MDVP:PPQ'}, {'label': 'Jitter:DDP'}, {'label': 'MDVP:Shimmer'},
        {'label': 'MDVP:Shimmer(dB)'}, {'label': 'Shimmer:APQ3'}, {'label': 'Shimmer:APQ5'},
        {'label': 'MDVP:APQ'}, {'label': 'Shimmer:DDA'}, {'label': 'NHR'}, {'label': 'HNR'},
        {'label': 'RPDE'}, {'label': 'DFA'}, {'label': 'spread1'}, {'label': 'spread2'},
        {'label': 'D2'}, {'label': 'PPE'}
    ]
    user_input = get_user_input(parkinsons_fields)
    if user_input and st.button("Get Parkinson's Prediction"):
        scaled_input = parkinsons_scaler.transform([user_input])
        result = parkinsons_model.predict(scaled_input)[0]
        st.success("Parkinson's Disease Detected" if result == 1 else "No Parkinson's Disease")

