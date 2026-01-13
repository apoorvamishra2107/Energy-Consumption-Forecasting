import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# -----------------------------
# CONFIG (FAST MODE)
# -----------------------------
SEQ_LEN = 24
EPOCHS = 1              # VERY FAST
HIDDEN_SIZE = 32        # smaller model
LR = 0.001

# -----------------------------
# LOAD SMALL DATASET
# -----------------------------
data = pd.read_csv("data/energy_small.csv", sep=";")
data = data.tail(5000)  # keep last 5k rows only

values = data["Global_active_power"].astype(float).values.reshape(-1, 1)

# -----------------------------
# SCALE
# -----------------------------
scaler = MinMaxScaler()
scaled = scaler.fit_transform(values)

# -----------------------------
# CREATE SEQUENCES
# -----------------------------
X, y = [], []
for i in range(len(scaled) - SEQ_LEN):
    X.append(scaled[i:i + SEQ_LEN])
    y.append(scaled[i + SEQ_LEN])

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# -----------------------------
# MODEL
# -----------------------------
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, HIDDEN_SIZE, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -----------------------------
# TRAIN (VERY FAST)
# -----------------------------
model.train()
optimizer.zero_grad()
output = model(X)
loss = criterion(output, y)
loss.backward()
optimizer.step()

print("Training complete")

# -----------------------------
# SAVE
# -----------------------------
os.makedirs("model", exist_ok=True)
torch.save(model, "model/lstm_model.pt")
joblib.dump(scaler, "model/scaler.save")

print("Model and scaler saved")
