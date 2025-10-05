
import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# --- Add stars to background using HTML, without extra boxes ---
# --- Custom CSS for Exoplanet Theme and Animated Stars ---
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
        background: transparent !important;
        border-radius: 0 !important;
        padding: 0 !important;
        color: #e0e6ed;
        box-shadow: none !important;
    }
    .stPlotlyChart, .stImage, .stAltairChart, .stVegaLiteChart, .stPyplot {
        background: transparent !important;
        border-radius: 0 !important;
        padding: 0 !important;
        box-shadow: none !important;
    }
    /* Animated stars background */
    .stars {
        position: fixed;
        width: 100vw;
        height: 100vh;
        top: 0;
        left: 0;
        z-index: 0;
        pointer-events: none;
    }
    .star {
        position: absolute;
        background: white;
        border-radius: 50%;
        opacity: 0.8;
        animation: twinkle 2s infinite ease-in-out;
    }
    @keyframes twinkle {
        0%, 100% { opacity: 0.8; }
        50% { opacity: 0.2; }
    }
    /* Custom font for exoplanet theme */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&display=swap');
    </style>
''', unsafe_allow_html=True)

import random
star_html = "<div class='stars' style='pointer-events:none; position:fixed; width:100vw; height:100vh; top:0; left:0; z-index:0;'>"
for _ in range(80):
    top = random.randint(0, 100)
    left = random.randint(0, 100)
    size = random.uniform(2, 5)
    star_html += f"<div class='star' style='top:{top}vh; left:{left}vw; width:{size}px; height:{size}px;'></div>"
star_html += "</div>"
st.markdown(star_html, unsafe_allow_html=True)


# --- Exoura Header ---
st.markdown("""
<div style='background:rgba(20,30,48,0.85); border-radius:16px; padding:2em 2em 1em 2em; margin-bottom: 2em; margin-top:0; box-shadow:0 2px 12px rgba(30,60,114,0.12); display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center;'>
    <h1 style='font-size:3em; margin-bottom:0.2em; text-align:center;'>🪐 Exoura AI</h1>
    <h3 style='font-weight:400; color:#aee2ff; text-align:center;'>Exoplanet False Positive Predictor</h3>
</div>
""", unsafe_allow_html=True)

# Load the pre-trained model
model = joblib.load('rf_model.joblib')


# --- Feature Input Card ---

# --- Feature Input Layout: Responsive 5 columns, always centered ---
input_cols = st.columns(5)
with input_cols[0]:
    koi_period = st.number_input('Orbital Period (days)', value=0.0, step=1.0, format="%f", key='koi_period', help='Time taken for one complete orbit around the star.')
with input_cols[1]:
    koi_duration = st.number_input('Transit Duration (hours)', value=0.0, step=1.0, format="%f", key='koi_duration', help='Duration of the transit event in hours.')
with input_cols[2]:
    koi_depth = st.number_input('Transit Depth (ppm)', value=0.0, step=1.0, format="%f", key='koi_depth', help='Depth of the transit in parts per million.')
with input_cols[3]:
    koi_prad = st.number_input('Planet Radius (Earth radii)', value=0.0, step=1.0, format="%f", key='koi_prad', help='Radius of the planet in Earth radii.')
with input_cols[4]:
    koi_model_snr = st.number_input('Signal Noise Ratio', value=0.0, step=1.0, format="%f", key='koi_model_snr', help='Signal-to-noise ratio of the model.')

inputs = [koi_period, koi_duration, koi_depth, koi_prad, koi_model_snr]

# Only predict if any input is changed from default
if any(x != 0.0 for x in inputs):
    input_array = np.array([inputs])
    predicted_index = model.predict(input_array)[0]
    labels = ['Not False Positive', 'False Positive']
    predicted_label = labels[predicted_index]

    # Get prediction probabilities
    probs = model.predict_proba(input_array)[0]
    fig, ax = plt.subplots(figsize=(4,4))
    wedges, texts, autotexts = ax.pie(
        probs, labels=labels, autopct='%1.1f%%', startangle=90,
        colors=['#66b3ff','#ff9999'], textprops={'color':'#e0e6ed','fontsize':14})
    ax.axis('equal')
    plt.setp(autotexts, size=16, weight="bold")
    plt.setp(texts, size=14)
    fig.patch.set_alpha(0)

    # Combine prediction and chart in one bottom box
    color = '#ff9999' if predicted_label == 'False Positive' else '#66b3ff'
    # Prediction box with pie chart in its own inner box
    st.markdown(f"""
    <div style='background:rgba(20,30,48,0.85); border-radius:16px; padding:2em 2em 2em 2em; margin-top:1.5em; box-shadow:0 2px 12px rgba(30,60,114,0.12); max-width: 500px; margin-left:auto; margin-right:auto; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center;'>
        <h2 style='color:#aee2ff; text-align:center; width:100%;'>Prediction</h2>
        <h3 style='color:{color}; font-size:2em; margin:0.5em 0 0.1em 0; text-align:center; width:100%;'>{predicted_label}</h3>
        <div style='display:flex; justify-content:center; margin-top:0.5em; width:100%;'>
    """, unsafe_allow_html=True)
    st.pyplot(fig)
    st.markdown(f"""
        </div>
    </div>
    """, unsafe_allow_html=True)
