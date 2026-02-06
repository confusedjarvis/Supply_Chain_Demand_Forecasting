import pandas as pd
from pathlib import Path

def preprocess_data():
    # Get project root directory
    BASE_DIR = Path(__file__).resolve().parent.parent

    # Absolute path to CSV
    csv_path = BASE_DIR / "data" / "supply_chain_data.csv"

    print("Reading data from:", csv_path)  # Debug print

    df = pd.read_csv(csv_path)

    df.dropna(inplace=True)

    if "Date" in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Quarter'] = df['Date'].dt.quarter
        df.drop(columns=['Date'], inplace=True)

    df = pd.get_dummies(df, drop_first=True)
    return df
