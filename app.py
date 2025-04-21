# 📦 Required packages: streamlit, yfinance, pandas, numpy, torch, scikit-learn, requests, transformers, openpyxl, pennylane

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pennylane as qml
from torch import nn

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

# ----------- SENTIMENT SETUP -----------
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model_sent = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

def get_sentiment(text):
    tokens = tokenizer(text, return_tensors='pt', truncation=True)
    with torch.no_grad():
        output = model_sent(**tokens)
    probs = torch.nn.functional.softmax(output.logits, dim=1)
    return probs[0][0].item(), probs[0][2].item()  # negative, positive

# ----------- FETCH FUNCTIONS -----------
def fetch_stock_data(symbol):
    df = yf.download(symbol, period='6mo')
    df['Return'] = df['Close'].pct_change()
    df['Volume_Change'] = df['Volume'].pct_change()
    df = df.dropna()
    return df

def fetch_news():
    url = "https://service.upstox.com/content/open/v5/news/sub-category/news/list//market-news/stocks?page=1&pageSize=100"
    res = requests.get(url).json()
    return res.get('data', {}).get('items', [])

def analyze_stock(symbol):
    try:
        df = fetch_stock_data(symbol)
        news_items = fetch_news()
        df['Date'] = df.index.date

        # Sentiment grouping by date
        news_df = pd.DataFrame(news_items)
        news_df['publishedDate'] = pd.to_datetime(news_df['publishedDate']).dt.date
        news_df['neg'], news_df['pos'] = zip(*news_df['title'].map(get_sentiment))
        sent_df = news_df.groupby('publishedDate')[['neg', 'pos']].mean().reset_index()
        sent_df.rename(columns={'publishedDate': 'Date'}, inplace=True)

        df = pd.merge(df, sent_df, on='Date', how='left').fillna(0)
        df['7D_Return'] = df['Close'].pct_change(7).fillna(0)

        # Quantum input
        features = ['Return', 'Volume_Change', '7D_Return', 'pos']
        X = df[features].fillna(0).values
        X = StandardScaler().fit_transform(X)
        X_torch = torch.tensor(X, dtype=torch.float32)

        model = HybridClassifier()
        with torch.no_grad():
            preds = model(X_torch).numpy().flatten()
            df['Quantum_Anomaly'] = (preds > 0.5).astype(int)

        return df[['Date', 'Close', 'Quantum_Anomaly']]
    except Exception as e:
        return None

# ----------- STREAMLIT UI -----------
st.set_page_config(page_title="Quantector", layout="wide")
st.title("🧠 Quantector – Quantum-enhanced Stock Anomaly Detector")

# Load Excel and sheet list
try:
    xl = pd.ExcelFile("stocklist.xlsx")
    stock_sheets = xl.sheet_names

    # User inputs
    selected_sheet = st.selectbox("Select Stock List", stock_sheets)
    analyze_button = st.button("Analyze Stocks")

    if analyze_button:
        try:
            stock_df = pd.read_excel("stocklist.xlsx", sheet_name=selected_sheet)

            if 'Symbol' not in stock_df.columns:
                st.error("Error: The selected sheet doesn't have a 'Symbol' column.")
            else:
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
                        st.line_chart(df.set_index('Date')[['Close']])
                        st.bar_chart(df.set_index('Date')[['Quantum_Anomaly']])
        except Exception as e:
            st.error(f"Error analyzing stocks: {str(e)}")
except Exception as e:
    st.error("⚠️ Error loading 'stocklist.xlsx'. Please check the file.")
