import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import pennylane as qml
from torch import nn
from sklearn.preprocessing import StandardScaler
from io import BytesIO

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
    
    # Calculate percentage changes with error handling
    df['Return'] = df['Close'].pct_change().replace([np.inf, -np.inf], np.nan)
    df['Volume_Change'] = df['Volume'].pct_change().replace([np.inf, -np.inf], np.nan)
    df['7D_Return'] = df['Close'].pct_change(7).replace([np.inf, -np.inf], np.nan)
    
    # Drop rows with NA values
    df = df.dropna()
    
    # Clip extreme values to prevent overflow
    for col in ['Return', 'Volume_Change', '7D_Return']:
        df[col] = df[col].clip(lower=-1, upper=1)  # Clipping to ±100% change
    
    # Reset index and rename date column
    df = df.reset_index()
    df = df.rename(columns={'Date': 'TradeDate'})  # Avoid conflict with index
    
    return df

def analyze_stock(symbol):
    try:
        df = fetch_stock_data(symbol)
        if df.empty:
            st.warning(f"No valid data for {symbol}")
            return None
            
        df['Symbol'] = symbol  # Add symbol column for reference

        # Prepare features for quantum model
        features = ['Return', 'Volume_Change', '7D_Return']
        X = df[features].values
        
        # Additional check for infinite values
        if not np.isfinite(X).all():
            st.warning(f"Non-finite values detected in {symbol}, cleaning data...")
            X = np.nan_to_num(X, nan=0, posinf=1, neginf=-1)  # Replace any remaining inf/nan
            
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Convert to tensor
        X_torch = torch.tensor(X_scaled, dtype=torch.float32)

        # Initialize and run quantum model
        model = HybridClassifier()
        with torch.no_grad():
            preds = model(X_torch).numpy().flatten()
            df['Quantum_Anomaly'] = preds  # Store raw probability
            df['Anomaly_Flag'] = (preds > 0.5).astype(int)  # Binary flag

        return df[['Symbol', 'TradeDate', 'Close', 'Quantum_Anomaly', 'Anomaly_Flag']]
    except Exception as e:
        st.error(f"Error analyzing {symbol}: {str(e)}")
        return None

# ----------- STREAMLIT UI -----------
st.set_page_config(page_title="Quantector", layout="wide")
st.title("🧠 Quantector – Quantum-enhanced Stock Anomaly Detector")

def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Anomaly_Results')
    return output.getvalue()

# Main analysis function
def run_analysis(selected_sheet):
    try:
        stock_df = pd.read_excel("stocklist.xlsx", sheet_name=selected_sheet)

        if 'Symbol' not in stock_df.columns:
            st.error("Error: The selected sheet doesn't have a 'Symbol' column.")
            return None, None

        symbols = stock_df['Symbol'].tolist()
        all_results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, symbol in enumerate(symbols):
            status_text.text(f"🔍 Analyzing {symbol} ({i+1}/{len(symbols)})...")
            result = analyze_stock(symbol)
            if result is not None:
                all_results.append(result)
            progress_bar.progress((i + 1) / len(symbols))

        if not all_results:
            st.warning("No valid stock data could be fetched. Please try again later.")
            return None, None
        
        # Combine all results
        combined_df = pd.concat(all_results)
        
        # Find top anomalies
        top_anomalies = combined_df[combined_df['Anomaly_Flag'] == 1].sort_values(
            'Quantum_Anomaly', ascending=False).head(10)
        
        return combined_df, top_anomalies

    except Exception as e:
        st.error(f"Error analyzing stocks: {str(e)}")
        return None, None

# Load Excel and sheet list
try:
    xl = pd.ExcelFile("stocklist.xlsx")
    stock_sheets = xl.sheet_names

    # User inputs
    selected_sheet = st.selectbox("Select Stock List", stock_sheets)
    analyze_button = st.button("Analyze Stocks")

    if analyze_button:
        combined_df, top_anomalies = run_analysis(selected_sheet)
        
        if combined_df is not None:
            # Show top anomalies
            st.subheader("🚨 Top 10 Anomalies Detected")
            st.dataframe(top_anomalies[['Symbol', 'TradeDate', 'Close', 'Quantum_Anomaly']]
                        .rename(columns={
                            'Quantum_Anomaly': 'Anomaly Score',
                            'TradeDate': 'Date'
                        }).style.format({'Anomaly Score': '{:.3f}'}))
            
            # Show all results with expander
            with st.expander("View All Results"):
                st.dataframe(combined_df.sort_values(['Symbol', 'TradeDate'])
            
            # Download button
            st.subheader("📥 Download Results")
            excel_data = to_excel(combined_df.rename(columns={'TradeDate': 'Date'}))
            st.download_button(
                label="Download Excel File",
                data=excel_data,
                file_name="quantum_anomaly_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

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
    
    **Results Interpretation:**
    - Anomaly Score: Probability between 0-1 (higher = more anomalous)
    - Anomaly Flag: 1 if score > 0.5, 0 otherwise
    
    **Requirements:**
    - An Excel file named 'stocklist.xlsx' with sheets containing 'Symbol' columns
    - Python packages listed in requirements.txt
    
    The quantum model uses {n_qubits} qubits with angle embedding and strongly entangling layers.
    """)
