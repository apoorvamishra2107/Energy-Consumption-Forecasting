import streamlit as st
import torch
import numpy as np
import pandas as pd
import joblib
from model import LSTMModel

SEQ_LEN = 24
MODEL_PATH = "model/lstm_model.pt"
SCALER_PATH = "model/scaler.save"

@st.cache_resource
def load_model():
    model = LSTMModel(input_size=1, hidden_size=50, num_layers=1)
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()
scaler = joblib.load(SCALER_PATH)

st.title("Energy Consumption Forecast")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    values = df.iloc[:, 1].values.reshape(-1, 1)
    scaled = scaler.transform(values)

    seq = scaled[-SEQ_LEN:]
    seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        prediction = model(seq).item()

    prediction = scaler.inverse_transform([[prediction]])[0][0]
    st.success(f"Predicted Energy Consumption: {prediction:.2f}")
