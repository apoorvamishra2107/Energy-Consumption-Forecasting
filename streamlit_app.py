import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import joblib
import numpy as np

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Energy Consumption Forecasting",
    layout="wide"
)

# -----------------------------
# MODEL DEFINITION (MUST MATCH train.py)
# -----------------------------
SEQ_LEN = 24
HIDDEN_SIZE = 32

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, HIDDEN_SIZE, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

# -----------------------------
# CACHED LOADERS
# -----------------------------
@st.cache_resource
def load_model():
    model = LSTMModel()
    state_dict = torch.load(
        "model/lstm_model.pt",
        map_location="cpu"
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model

@st.cache_resource
def load_scaler():
    return joblib.load("model/scaler.save")

@st.cache_data
def load_data():
    return pd.read_csv("data/energy_small.csv", sep=";")

model = load_model()
scaler = load_scaler()
data = load_data()

# -----------------------------
# UI
# -----------------------------
st.title("⚡ Energy Consumption Forecasting")

st.sidebar.header("Controls")
seq_len = st.sidebar.slider("Sequence Length", 12, 48, 24)

st.subheader("Recent Energy Usage")
st.line_chart(
    data["Global_active_power"]
    .astype(float)
    .tail(300)
)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict Next Consumption"):
    recent = data["Global_active_power"].astype(float).values[-seq_len:]
    recent = recent.reshape(-1, 1)

    scaled = scaler.transform(recent)
    seq_tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        pred_scaled = model(seq_tensor).item()

    prediction = scaler.inverse_transform([[pred_scaled]])[0][0]

    st.success(f"Predicted Next Value: {prediction:.3f} kW")

