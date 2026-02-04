import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.getcwd())

from src.gnosis.quantum import PriceTimeManifold
from scripts.evaluate_manifold import prints_to_ohlcv

def get_hourly_prediction():
    data_path = "data/large_history/prints.parquet"
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    # 1. Load data
    df = pd.read_parquet(data_path)
    symbol = "BTCUSDT"
    symbol_df = df[df['symbol'] == symbol].copy()
    
    if symbol_df.empty:
        print(f"Error: No data for {symbol}")
        return

    # 2. Process to 1m bars
    ohlcv_1m = prints_to_ohlcv(symbol_df, bar_minutes=1)
    current_p = ohlcv_1m['close'].iloc[-1]
    
    # 3. Predict for 60 minutes ahead (Engine's standard horizon)
    results = []
    timeframes = [5, 15, 30, 60]
    
    for tf in timeframes:
        tf_df = ohlcv_1m.resample(f'{tf}min', on='timestamp').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
            'volume': 'sum', 'buy_volume': 'sum', 'sell_volume': 'sum'
        }).dropna().reset_index()

        manifold = PriceTimeManifold()
        manifold.fit_from_1m_bars(tf_df)

        h_bars = 60 // tf
        res = manifold.predict_probabilistic(len(manifold._states) - 1, [h_bars], n_sims=5000)
        sim_data = res[h_bars]
        results.append(sim_data)

    # 4. Aggregate Results
    avg_median = sum(r['median'] for r in results) / len(results)
    avg_upper = sum(r['upper_90'] for r in results) / len(results)
    avg_lower = sum(r['lower_90'] for r in results) / len(results)
    avg_std = sum(r['std'] for r in results) / len(results)
    
    print(f"Current Price (last data): ${current_p:,.2f}")
    print(f"Predicted Median (1h): ${avg_median:,.2f}")
    print(f"90% Confidence Interval: [${avg_lower:,.2f}, ${avg_upper:,.2f}]")
    print(f"Expected Volatility: ${avg_std:,.2f}")
    
    # Confidence as a percentage
    # (High/Medium/Low based on standard deviation relative to price)
    rel_std = (avg_std / avg_median) * 100
    confidence = "HIGH" if rel_std < 0.5 else "MEDIUM" if rel_std < 1.5 else "LOW"
    print(f"Confidence Level: {confidence} ({rel_std:.2f}% relative std)")

if __name__ == "__main__":
    get_hourly_prediction()
