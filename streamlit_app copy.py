import streamlit as st
import time
import sklearn
import pandas as pd
import numpy as np
import joblib
if "currentRequest" not in st.session_state:
    st.session_state.currentRequest = 0
if "userInputs" not in st.session_state:
    st.session_state.userInputs = []

st.title("Exoura AI")

feature_names = ['koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_model_snr']
userInputs = []
chatRequests = ["Please input koi_period", "Please input koi_duration", "Please input koi_depth", "Please input koi_prad", "Please input koi_model_snr"]
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": f"Exoura: Hello! I am Exoura, an Exoplanet AI false positive detector."}, {"role": "assistant", "content": f"Exoura: Lets start with a few questions."}]
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



# Load the pre-trained model
model = joblib.load('rf_model.joblib')

# Show the request before user input ONLY if this is a new request (avoid duplicate messages)
if st.session_state.currentRequest < len(chatRequests):
    # Only add the message if it's not already the last assistant message
    if not (st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant" and chatRequests[st.session_state.currentRequest] in st.session_state.messages[-1]["content"]):
        with st.chat_message("assistant"):
            st.markdown(f"Exoura: {chatRequests[st.session_state.currentRequest]}")
        st.session_state.messages.append({"role": "assistant", "content": f"Exoura: {chatRequests[st.session_state.currentRequest]}"})

if prompt := st.chat_input("Upon BOT request, please provide data here."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.userInputs.append(float(prompt))
    st.session_state.currentRequest += 1
    # Show the next request immediately after input, if there are more
    if st.session_state.currentRequest < len(chatRequests):
        with st.chat_message("assistant"):
            st.markdown(f"Exoura: {chatRequests[st.session_state.currentRequest]}")
        st.session_state.messages.append({"role": "assistant", "content": f"Exoura: {chatRequests[st.session_state.currentRequest]}"})
    if st.session_state.currentRequest == len(chatRequests):
        input_array = np.array([st.session_state.userInputs])
        predicted_index = model.predict(input_array)[0]
        labels = ['Not False Positive', 'False Positive']
        predicted_label = labels[predicted_index]
        response = f"Exoura: Based on the inputs, the prediction is: {predicted_label}."
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.currentRequest = 0
        st.session_state.userInputs = []
        time.sleep(1)
        response = f"Exoura: Let's start over. {chatRequests[st.session_state.currentRequest]}"
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
