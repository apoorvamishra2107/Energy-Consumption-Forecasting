import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import joblib

# -----------------------------
# Config
# -----------------------------
SEQ_LEN = 24       # sequence length for LSTM
EPOCHS = 20
LR = 0.001
BATCH_SIZE = 32
HIDDEN_SIZE = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# Load dataset
# -----------------------------
data = pd.read_csv("data/energy.csv", sep=";", na_values=["?"])
data.dropna(inplace=True)
data['datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], dayfirst=True)
data.drop(['Date','Time'], axis=1, inplace=True)

values = data["Global_active_power"].astype(float).values.reshape(-1,1)

# -----------------------------
# Scale values
# -----------------------------
scaler = MinMaxScaler()
scaled = scaler.fit_transform(values)

# -----------------------------
# Create sequences
# -----------------------------
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

X_np, y_np = create_sequences(scaled, SEQ_LEN)

# Optional: reduce dataset for testing
# X_np = X_np[:5000]
# y_np = y_np[:5000]

# Convert to torch tensors
X_tensor = torch.tensor(X_np, dtype=torch.float32)
y_tensor = torch.tensor(y_np, dtype=torch.float32)

# -----------------------------
# DataLoader
# -----------------------------
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

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

model = LSTMModel().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -----------------------------
# Training loop
# -----------------------------
for epoch in range(EPOCHS):
    epoch_loss = 0
    for batch_X, batch_y in loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * batch_X.size(0)

    epoch_loss /= len(loader.dataset)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.6f}")

# -----------------------------
# Save model and scaler
# -----------------------------
os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/lstm_model.pt")
print("Model saved to model/lstm_model.pt")

joblib.dump(scaler, "model/scaler.save")
print("Scaler saved to model/scaler.save")

