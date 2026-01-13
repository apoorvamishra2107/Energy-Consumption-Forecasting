import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import joblib
from model import LSTMModel

# -----------------------------
# Configuration
# -----------------------------
SEQ_LEN = 24
EPOCHS = 20
LR = 0.001
BATCH_SIZE = 32

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("data/energy.csv", sep=";")
values = data.iloc[:, 1].values.reshape(-1, 1)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(values)
joblib.dump(scaler, "model/scaler.save")

# -----------------------------
# Create sequences
# -----------------------------
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled, SEQ_LEN)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -----------------------------
# Model
# -----------------------------
model = LSTMModel(input_size=1, hidden_size=50, num_layers=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -----------------------------
# Training
# -----------------------------
for epoch in range(EPOCHS):
    for xb, yb in loader:
        preds = model(xb)
        loss = criterion(preds, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

# -----------------------------
# Save model
# -----------------------------
torch.save(model.state_dict(), "model/lstm_model.pt")
print("Model saved successfully")
