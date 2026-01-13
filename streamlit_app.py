import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import joblib
import plotly.graph_objects as go
import io

# -----------------------------
# Config
# -----------------------------
SEQ_LEN = 24
HIDDEN_SIZE = 50
DATA_PATH = "data/energy.csv"
MODEL_PATH = "model/lstm_model.pt"
SCALER_PATH = "model/scaler.save"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# LSTM Model
# -----------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path, sep=";", na_values=["?"])
    df.dropna(inplace=True)
    if 'Date' in df.columns and 'Time' in df.columns:
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
        df.drop(['Date','Time'], axis=1, inplace=True)
    df['Global_active_power'] = df['Global_active_power'].astype(float)
    return df

data = load_data(DATA_PATH)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Options")
show_chart = st.sidebar.checkbox("Show Recent Consumption Chart", value=True)
seq_len_slider = st.sidebar.slider("Sequence Length for Prediction", 12, 48, SEQ_LEN, step=1)
future_steps = st.sidebar.slider("Number of Future Steps to Predict", 1, 24, 3)
chart_theme = st.sidebar.radio("Chart Theme", ("Light","Dark"))

# -----------------------------
# Dynamic background
# -----------------------------
if chart_theme == "Light":
    bg_css = """
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top left, #ffecd2, #fcb69f, #ff7e5f, #ff758c, #ffecd2);
        background-size: 400% 400%;
        animation: gradientBG 20s ease infinite;
    }
    """
else:
    bg_css = """
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top left, #0f0c29, #302b63, #24243e, #151515);
        background-size: 400% 400%;
        animation: gradientBG 25s ease infinite;
    }
    """
st.markdown(f"""
<style>
{bg_css}

@keyframes gradientBG {{
    0%{{background-position:0% 50%;}}
    50%{{background-position:100% 50%;}}
    100%{{background-position:0% 50%;}}
}}

.card {{
    background-color: rgba(255, 255, 255, 0.85);
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0 0 15px rgba(255,87,51,0.5), 0 0 30px rgba(255,87,51,0.3);
    margin-bottom: 10px;
    display: inline-block;
    margin-right: 10px;
    transition: all 0.3s ease-in-out;
    text-align: center;
}}
.card:hover {{
    box-shadow: 0 0 25px rgba(255,87,51,0.7), 0 0 50px rgba(255,87,51,0.5);
    transform: scale(1.05);
}}
[data-testid="stPlotlyChart"] {{
    background: transparent !important;
}}
h1,h2,h3,h4 {{ color:#333; }}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Streamlit layout
# -----------------------------
st.set_page_config(page_title="Energy Forecasting", page_icon="⚡", layout="wide")
st.title("⚡ Energy Consumption Prediction")
st.markdown("LSTM-based time series forecasting with **PyTorch**")

# -----------------------------
# Load scaler only (model deferred)
# -----------------------------
if os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)
else:
    st.error(f"Scaler not found at {SCALER_PATH}")

scaled_values = scaler.transform(data["Global_active_power"].values.reshape(-1,1))

# -----------------------------
# Helper for gradient colors
# -----------------------------
def gradient_colors(n, color1="#1f77b4", color2="#ff5733"):
    return [f"rgba({int(int(color1[1:3],16)*(1-i/n)+int(color2[1:3],16)*(i/n))},"
            f"{int(int(color1[3:5],16)*(1-i/n)+int(color2[3:5],16)*(i/n))},"
            f"{int(int(color1[5:7],16)*(1-i/n)+int(color2[5:7],16)*(i/n))},1)" for i in range(n)]

# -----------------------------
# Recent consumption chart
# -----------------------------
if show_chart:
    st.subheader("📈 Recent Energy Consumption")
    y_values = data["Global_active_power"].tail(100).tolist()
    x_values = data["datetime"].tail(100).tolist()
    colors = gradient_colors(len(y_values))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_values,
        mode='lines+markers',
        name="Actual",
        line=dict(color="#1f77b4", width=2),
        marker=dict(color=colors, size=6)
    ))
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Global Active Power (kW)",
        template="plotly_white" if chart_theme=="Light" else "plotly_dark",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Prediction button
# -----------------------------
st.subheader("⚡ Predict Next Energy Consumption")

@st.cache_resource
def load_model(path):
    model = LSTMModel().to(device)
    if os.path.exists(path):
        state_dict = torch.load(path, map_location=device)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            print("RuntimeError loading state_dict:", e)
            model.load_state_dict(state_dict, strict=False)
        model.eval()
    else:
        st.error(f"Model not found at {path}")
    return model

if st.button("Predict"):
    # Load model only when needed
    model = load_model(MODEL_PATH)

    seq_len = seq_len_slider
    steps = future_steps
    recent_seq = scaled_values[-seq_len:].reshape(-1,1)
    predictions = []

    for _ in range(steps):
        seq_tensor = torch.tensor(recent_seq[-seq_len:], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(seq_tensor)
        pred_val = pred.cpu().numpy()[0][0]
        predictions.append(pred_val)
        recent_seq = np.vstack([recent_seq, [[pred_val]]])

    predictions_scaled = scaler.inverse_transform(np.array(predictions).reshape(-1,1)).flatten()

    # Metric cards
    st.markdown("<div style='display:flex; flex-wrap: wrap;'>", unsafe_allow_html=True)
    for i,val in enumerate(predictions_scaled,1):
        st.markdown(f"""
        <div class="card">
            <h4>Step +{i}</h4>
            <h2 style="color:#ff5733;">{val:.2f} kW</h2>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Overlay chart
    recent_actual = data["Global_active_power"].tail(seq_len).tolist()
    times = data["datetime"].tail(seq_len).tolist()
    freq = data["datetime"].diff().mode()[0]

    for i in range(steps):
        recent_actual.append(predictions_scaled[i])
        times.append(times[-1] + freq)

    colors_actual = gradient_colors(seq_len)
    colors_pred = gradient_colors(steps, color1="#ff5733", color2="#ffbd69")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=times[:seq_len], y=recent_actual[:seq_len],
        mode='lines+markers', name="Actual",
        line=dict(color="#1f77b4", width=2),
        marker=dict(color=colors_actual, size=6)
    ))
    fig2.add_trace(go.Scatter(
        x=times[seq_len-1:], y=recent_actual[seq_len-1:],
        mode='lines+markers', name="Prediction",
        line=dict(color="#ff5733", width=3, dash='dash'),
        marker=dict(color=colors_pred, size=6)
    ))
    fig2.update_layout(
        xaxis_title="Time",
        yaxis_title="Global Active Power (kW)",
        template="plotly_white" if chart_theme=="Light" else "plotly_dark",
        height=400
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Download CSV
    pred_df = pd.DataFrame({"datetime": times[-steps:], "prediction_kW": predictions_scaled})
    csv_buffer = io.StringIO()
    pred_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download Predictions as CSV",
        data=csv_buffer.getvalue(),
        file_name="energy_predictions.csv",
        mime="text/csv"
    )

