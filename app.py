import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import pennylane as qml
from torch import nn
from sklearn.preprocessing import StandardScaler

# ----------- QUANTUM MODEL -----------
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QuantumLayer(nn.Module):
    def __init__(self):
        super().__init__()
        weight_shapes = {"weights": (6, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

    def forward(self, x):
        return self.qlayer(x)

class HybridClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(n_qubits, 8)
        self.q_layer = QuantumLayer()
        self.out = nn.Linear(8, 1)

    def forward(self, x):
        x = self.q_layer(x)
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.out(x))

# ----------- FETCH FUNCTIONS -----------
@st.cache_data
def fetch_stock_data(symbol):
    df = yf.download(symbol, period='6mo')
    df['Return'] = df['Close'].pct_change()
    df['Volume_Change'] = df['Volume'].pct_change()
    df['7D_Return'] = df['Close'].pct_change(7)
    df = df.dropna()
    return df

def analyze_stock(symbol):
    try:
        df = fetch_stock_data(symbol)
        df['Date'] = df.index.date

        # Prepare features for quantum model
        features = ['Return', 'Volume_Change', '7D_Return']
        X = df[features].fillna(0).values
        X = StandardScaler().fit_transform(X)
        X_torch = torch.tensor(X, dtype=torch.float32)

        # Initialize and run quantum model
        model = HybridClassifier()
        with torch.no_grad():
            preds = model(X_torch).numpy().flatten()
            df['Quantum_Anomaly'] = (preds > 0.5).astype(int)

        return df[['Date', 'Close', 'Quantum_Anomaly']]
    except Exception as e:
        st.error(f"Error analyzing {symbol}: {str(e)}")
        return None

# ----------- STREAMLIT UI -----------
st.set_page_config(page_title="Quantector", layout="wide")
st.title("🧠 Quantector – Quantum-enhanced Stock Anomaly Detector")

# Main analysis function
def run_analysis(selected_sheet):
    try:
        stock_df = pd.read_excel("stocklist.xlsx", sheet_name=selected_sheet)

        if 'Symbol' not in stock_df.columns:
            st.error("Error: The selected sheet doesn't have a 'Symbol' column.")
            return

        symbols = stock_df['Symbol'].tolist()
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, symbol in enumerate(symbols):
            status_text.text(f"🔍 Analyzing {symbol} ({i+1}/{len(symbols)})...")
            result = analyze_stock(symbol)
            if result is not None:
                results.append((symbol, result))
            progress_bar.progress((i + 1) / len(symbols))

        if not results:
            st.warning("No valid stock data could be fetched. Please try again later.")
        else:
            for symbol, df in results:
                st.subheader(f"📈 {symbol}")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Price Movement**")
                    st.line_chart(df.set_index('Date')['Close'])
                with col2:
                    st.markdown("**Anomaly Detection**")
                    st.bar_chart(df.set_index('Date')['Quantum_Anomaly'])

    except Exception as e:
        st.error(f"Error analyzing stocks: {str(e)}")

# Load Excel and sheet list
try:
    xl = pd.ExcelFile("stocklist.xlsx")
    stock_sheets = xl.sheet_names

    # User inputs
    selected_sheet = st.selectbox("Select Stock List", stock_sheets)
    analyze_button = st.button("Analyze Stocks")

    if analyze_button:
        run_analysis(selected_sheet)

except FileNotFoundError:
    st.error("⚠️ File 'stocklist.xlsx' not found. Please ensure it's in the same directory.")
except Exception as e:
    st.error(f"⚠️ Error loading Excel file: {str(e)}")

# Add documentation
with st.expander("About this App"):
    st.markdown("""
    ## Quantector - Quantum Stock Anomaly Detection
    
    This app analyzes stock data using a hybrid quantum-classical machine learning model to detect potential anomalies.
    
    **How it works:**
    1. Fetches 6 months of historical data for each stock
    2. Calculates daily returns, volume changes, and 7-day returns
    3. Processes the data through a quantum neural network
    4. Flags potential anomalies (values > 0.5)
    
    **Requirements:**
    - An Excel file named 'stocklist.xlsx' with sheets containing 'Symbol' columns
    - Python packages listed in requirements.txt
    
    The quantum model uses {n_qubits} qubits with angle embedding and strongly entangling layers.
    """)
