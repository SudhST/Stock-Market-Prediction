import numpy as np
import pandas as pd
import yfinance as yf

def load_data(filepath, column='Close'):
    df = pd.read_csv(filepath)
    # Remove any row where the specified column contains non-numeric values
    df = df[pd.to_numeric(df[column], errors='coerce').notna()]
    # Convert the column to float32
    return df[column].astype(np.float32).values

def create_non_overlapping_windows(data, window_size):
    X, y = [], []
    for i in range(0, len(data) - window_size, window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    # Ensure arrays are float32 for compatibility with torch
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def download_dji_data(start='2014-01-01', end='2022-12-31', filename=None):
    """
    Download DJI daily data from Yahoo Finance using yfinance.
    If filename is provided, saves the data as CSV.
    Returns the DataFrame.
    """
    dji = yf.download('^DJI', start=start, end=end, progress=False)
    if filename:
        dji.to_csv(filename)
    return dji
    if filename:
        dji.to_csv(filename)
    return dji
