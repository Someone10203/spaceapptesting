
import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# --- Custom CSS for Exoplanet Theme ---
st.markdown('''
    <style>
    body {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        color: #e0e6ed;
    }
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        color: #e0e6ed;
    }
    .main {
        background: transparent;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #aee2ff;
        font-family: 'Orbitron', sans-serif;
        letter-spacing: 2px;
    }
    /* Style number input label and box */
    label, .stNumberInput label {
        color: #aee2ff !important;
        font-weight: 600;
        font-size: 1.1em;
        margin-bottom: 0.2em;
        font-family: 'Orbitron', sans-serif;
        letter-spacing: 1px;
    }
    .stNumberInput > div > input {
        background: #1a2636 !important;
        color: #aee2ff !important;
        border-radius: 8px !important;
        border: 1px solid #3a506b !important;
        font-size: 1.1em;
        font-family: 'Orbitron', sans-serif;
    }
    .stButton > button {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: #fff;
        border-radius: 8px;
        border: none;
        font-weight: bold;
        box-shadow: 0 2px 8px rgba(30,60,114,0.2);
    }
    .stMarkdown, .stDataFrame, .stTable {
        background: rgba(20, 30, 48, 0.7);
        border-radius: 12px;
        padding: 1em;
        color: #e0e6ed;
    }
    .stPlotlyChart, .stImage, .stAltairChart, .stVegaLiteChart, .stPyplot {
        background: rgba(20, 30, 48, 0.7);
        border-radius: 12px;
        padding: 1em;
    }
    /* Custom font for exoplanet theme */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&display=swap');
    </style>
''', unsafe_allow_html=True)


# --- Exoura Header ---
st.markdown("""
<div style='text-align:center; margin-bottom: 2em;'>
    <h1 style='font-size:3em; margin-bottom:0.2em;'>🪐 Exoura AI</h1>
    <h3 style='font-weight:400; color:#aee2ff;'>Exoplanet False Positive Detector</h3>
    <p style='color:#b2becd; font-size:1.1em;'>Enter the planetary candidate's features below to see if it's likely a false positive.<br>Powered by Machine Learning and inspired by the cosmos.</p>
</div>
""", unsafe_allow_html=True)

# Load the pre-trained model
model = joblib.load('rf_model.joblib')


# --- Feature Input Card ---
col1, col2, col3 = st.columns([1,1,1])
with col1:
    koi_period = st.number_input('Orbital Period (days)', value=0.0, format="%f", key='koi_period', help='Time taken for one complete orbit around the star.')
    koi_prad = st.number_input('Planet Radius (Earth radii)', value=0.0, format="%f", key='koi_prad', help='Radius of the planet in Earth radii.')
with col2:
    koi_duration = st.number_input('Transit Duration (hours)', value=0.0, format="%f", key='koi_duration', help='Duration of the transit event in hours.')
    koi_model_snr = st.number_input('Model SNR', value=0.0, format="%f", key='koi_model_snr', help='Signal-to-noise ratio of the model.')
with col3:
    koi_depth = st.number_input('Transit Depth (ppm)', value=0.0, format="%f", key='koi_depth', help='Depth of the transit in parts per million.')

inputs = [koi_period, koi_duration, koi_depth, koi_prad, koi_model_snr]

# Only predict if any input is changed from default
if any(x != 0.0 for x in inputs):
    input_array = np.array([inputs])
    predicted_index = model.predict(input_array)[0]
    labels = ['Not False Positive', 'False Positive']
    predicted_label = labels[predicted_index]

    st.markdown(f"""
    <div style='background:rgba(20,30,48,0.85); border-radius:16px; padding:1.5em; margin-top:1em; box-shadow:0 2px 12px rgba(30,60,114,0.12); text-align:center;'>
        <h2 style='color:#aee2ff;'>Prediction</h2>
        <h3 style='color:{'#ff9999' if predicted_label == 'False Positive' else '#66b3ff'}; font-size:2em; margin:0.5em 0 0.2em 0;'>{predicted_label}</h3>
    </div>
    """, unsafe_allow_html=True)

    # Get prediction probabilities
    probs = model.predict_proba(input_array)[0]
    # Pie chart
    fig, ax = plt.subplots(figsize=(4,4))
    wedges, texts, autotexts = ax.pie(
        probs, labels=labels, autopct='%1.1f%%', startangle=90,
        colors=['#66b3ff','#ff9999'], textprops={'color':'#e0e6ed','fontsize':14})
    ax.axis('equal')
    plt.setp(autotexts, size=16, weight="bold")
    plt.setp(texts, size=14)
    fig.patch.set_alpha(0)
    st.markdown("<div style='text-align:center; margin-top:1em;'><b>Model Confidence</b></div>", unsafe_allow_html=True)
    st.pyplot(fig)
