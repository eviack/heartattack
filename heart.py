import streamlit as st
import pickle as pkl
import plotly.graph_objects as go
import numpy as np
import pandas as pd

###########Setup###########
st.set_page_config(
    page_title="Heart-Attack Prediction",
    initial_sidebar_state='expanded',
    layout='wide'
)

with open('style.css', 'r') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

###############################

st.header("Heart Attack Prediction")
st.markdown('''
**Cardiovascular diseases (CVDs)** are the number **1** cause of death globally, 
taking an estimated **17.9** million lives each year, which accounts for **31%** of all deaths worldwide. 
Four out of 5CVD deaths are due to heart attacks and strokes, and one-third of these deaths occur prematurely in people under **70** years of age. 
Heart failure is a common event caused by CVDs and **this model is trained on dataset which contains 11 features that can be used to predict a possible heart disease.**
*It needs early detection and management !*   
            ''')

st.sidebar.header('Input Parameters')

age = st.sidebar.number_input('Age', min_value=0, max_value=100, value=50)
sex = st.sidebar.selectbox('Sex', ('Male', 'Female'))
chest_pain = st.sidebar.selectbox('Chest Pain Type', 
                                  ('Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'),
                                  help='Type of chest pain the patient experiences')
resting_bp = st.sidebar.number_input('Resting BP', min_value=0, max_value=200, value=120,
                                     help='Resting blood pressure in mm Hg')
cholesterol = st.sidebar.number_input('Cholesterol', min_value=0, max_value=580, value=200,
                                      help='Serum cholesterol in mg/dl')
fasting_bs = st.sidebar.selectbox('Fasting BS', ('< 120 mg/dl', '> 120 mg/dl'),
                                  help='Fasting blood sugar levels')
resting_ecg = st.sidebar.selectbox('Resting ECG', 
                                   ('Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'),
                                   help='''Results of the resting electrocardiogram. Normal, having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), 
                                   showing probable or definite left ventricular hypertrophy by Estes' criteria''')
max_hr = st.sidebar.number_input('Max HR', min_value=60, max_value=202, value=150,
                                 help='Maximum heart rate achieved')
exercise_angina = st.sidebar.selectbox('Exercise Angina', ('No', 'Yes'),
                                       help='Exercise-induced angina')
oldpeak = st.sidebar.number_input('Oldpeak', min_value=0.0, max_value=10.0, value=2.0,
                                  help='ST depression induced by exercise relative to rest')
st_slope = st.sidebar.selectbox('ST Slope', ('Upsloping', 'Flat', 'Downsloping'),
                                help='The slope of the peak exercise ST segment')

caa = st.sidebar.slider('Number of Major Vessels (0-3)', min_value=0, max_value=3, value=0,
                        help="This feature indicates the number of major blood vessels (0-3) colored by fluoroscopy. It helps assess the extent of coronary artery disease.")
thall = st.sidebar.slider('Thalassemia rate (0-3)', min_value=0, max_value=3, value=2,
                          help=" Thalassemia rate (0-3), which is used to describe various conditions of the heart. Higher values typically indicate more severe conditions.")

# Map categorical variables to numerical labels
sex_dict = {'Male': 1, 'Female': 0}
chest_pain_dict = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-Anginal Pain': 2, 'Asymptomatic': 3}
fasting_bs_dict = {'< 120 mg/dl': 0, '> 120 mg/dl': 1}
resting_ecg_dict = {'Normal': 0, 'ST-T wave abnormality': 1, 'Left ventricular hypertrophy': 2}
exercise_angina_dict = {'No': 0, 'Yes': 1}
st_slope_dict = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}

# Prepare input data
input_data = {
    'age': age,
    'sex': sex_dict[sex],
    'cp': chest_pain_dict[chest_pain],
    'trtbps': resting_bp,
    'chol': cholesterol,
    'fbs': fasting_bs_dict[fasting_bs],
    'restecg': resting_ecg_dict[resting_ecg],
    'thalachh': max_hr,
    'exng': exercise_angina_dict[exercise_angina],
    'oldpeak': round(oldpeak,2),
    'slp': st_slope_dict[st_slope],
    'caa': caa,
    'thall': thall
}

st.sidebar.header("Input structure")
st.sidebar.write(input_data)
import plotly.graph_objects as go

def create_radar_chart(input_data):
    categories = ['Age', 'Sex', 'Chest Pain', 'Resting BP', 'Cholesterol', 'Fasting BS', 
                  'Resting ECG', 'Max HR', 'Exercise Angina', 'Oldpeak', 'ST Slope', 'Caa', 'Thal Rate']
    
    values = [
        input_data['age'],
        input_data['sex'],
        input_data['cp'],
        input_data['trtbps'],
        input_data['chol'],
        input_data['fbs'],
        input_data['restecg'],
        input_data['thalachh'],
        input_data['exng'],
        input_data['oldpeak'],
        input_data['slp'],
        input_data['caa'],
        input_data['thall']
    ]
    
    # Normalize values for radar chart
    max_values = [100, 1, 3, 200, 580, 1, 2, 202, 1, 10, 2]
    normalized_values = [v / m for v, m in zip(values, max_values)]
    
    # Append the first value to the end to close the radar chart
    normalized_values.append(normalized_values[0])
    categories.append(categories[0])

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=normalized_values,
        theta=categories,
        fill='toself',
        name='Input Data',
        line=dict(color='#1f77b4')  # Use a single color for the filled area
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
            
        ),
        showlegend=True

    )

    return fig

with open('heartattckpred.pkl', 'rb') as f:
    best_rf_pipeline = pkl.load(f)

def get_pred(row):
    # Convert the row to DataFrame if it is a dictionary
    if isinstance(row, dict):
        row = pd.DataFrame([row])
    
    # Make prediction
    prediction = best_rf_pipeline.predict(row)
    
    prob = best_rf_pipeline.predict_proba(row)
    
    # The probability of class 1 (heart attack)
    prob_heart_attack = prob[0][1]
    
    if prediction[0]==1:
        return "High chances of having a heart attack!", f"{prob_heart_attack*100:.2f}"
    else:
        return "Low chances of having a heart attack", f"{prob_heart_attack*100:.2f}"

# Example usage in Streamlit
col1, col2 = st.columns([2,1])
with col1:
    st.plotly_chart(create_radar_chart(input_data), use_container_width=True)

with col2:
    st.subheader("Prediction results")
    st.caption('''The results say whether you got a high chance of getting a heart attack or not. 
               **It also shows the probability.**''')
    pred, chance = get_pred(input_data)

    if 'high' in pred.lower():
        st.error(pred)
    else:
        st.success(pred)

    st.write("**Chances are**")

    if float(chance)>50:
        st.markdown(f"## :red[{chance}%]")
    else:
        st.markdown(f"## :green[{chance}%]")
    


