import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ensure root directory is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import get_nasdaq100_tickers, get_sp500_tickers, fetch_clean_data
from src.backtest import walk_forward_backtest
from src.stochastic_dominance import test_second_order_stochastic_dominance
from src.econometrics import run_econometric_evaluation

def main():
    print("="*80)
    print("RE-ENGINEERING MEGA-CAP EQUITIES: REPRODUCIBLE RESEARCH PIPELINE")
    print("="*80)
    
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(root_dir, "data")
    results_dir = os.path.join(root_dir, "results")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)
    
    # 1. Fetch Real Data
    ndx_tickers = get_nasdaq100_tickers()
    sp_tickers = get_sp500_tickers()
    
    ndx_prices = fetch_clean_data(ndx_tickers, ['QQQ', 'QQQE'], start="2017-01-01", end="2026-10-10",
                                  cache_path=os.path.join(data_dir, "ndx_prices.parquet"))
    sp_prices = fetch_clean_data(sp_tickers, ['SPY', 'RSP'], start="2017-01-01", end="2026-10-10",
                                 cache_path=os.path.join(data_dir, "sp500_prices.parquet"))
    
    # 2. Run Nasdaq-100 Backtest
    print("\nExecuting Nasdaq-100 Walk-Forward Backtest...")
    ndx_constituents = [c for c in ndx_prices.columns if c not in ['QQQ', 'QQQE']]
    ndx_out, ndx_enb, _ = walk_forward_backtest(ndx_prices[ndx_constituents].pct_change().dropna(),
                                                ndx_prices[['QQQ', 'QQQE']].pct_change().dropna(),
                                                rebal_days=21, max_weight=None)
    
    # 3. Run S&P 500 Backtest
    print("\nExecuting S&P 500 Walk-Forward Backtest (N=476)...")
    sp_constituents = [c for c in sp_prices.columns if c not in ['SPY', 'RSP']]
    sp_out, sp_enb, _ = walk_forward_backtest(sp_prices[sp_constituents].pct_change().dropna(),
                                              sp_prices[['SPY', 'RSP']].pct_change().dropna(),
                                              rebal_days=21, max_weight=None)
    
    print("\nPipeline execution complete! Ready for econometric and SSD evaluation.")

if __name__ == "__main__":
    main()
