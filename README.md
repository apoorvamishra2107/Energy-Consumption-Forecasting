# 🔌 Energy Consumption Forecasting using Deep Learning (LSTM)

A deep learning–based time series forecasting project that predicts future household energy consumption using an **LSTM neural network implemented in PyTorch**, with an interactive **Streamlit web application** for real-time inference.

---

## 📌 Project Overview

Electricity consumption forecasting is a critical task in energy management systems.  
This project uses historical household electricity usage data to train an **LSTM (Long Short-Term Memory)** model capable of predicting future energy consumption patterns.

The trained model is deployed as a **web-based prediction app** using Streamlit.

---

## 🚀 Key Features

- Time-series forecasting using **LSTM (PyTorch)**
- Handles real-world noisy data with missing values
- Data normalization using **MinMaxScaler**
- Offline model training + online inference
- Interactive **Streamlit web app**
- Clean ML lifecycle: preprocessing → training → saving → inference

---

## 🧠 Tech Stack

- **Programming Language:** Python  
- **Deep Learning Framework:** PyTorch  
- **Data Processing:** Pandas, NumPy  
- **Scaling:** Scikit-learn  
- **Visualization & App:** Streamlit  
- **Model Type:** LSTM (Recurrent Neural Network)

---

## 📊 Dataset

**Household Electric Power Consumption Dataset**  
Source: UCI Machine Learning Repository / Kaggle

- Sampling rate: 1 minute
- Duration: ~4 years
- Format: Semicolon (`;`) separated text file
- Target variable: `Global_active_power`
