import os
import io
import time
import requests
import pandas as pd
import yfinance as yf

def get_nasdaq100_tickers() -> list:
    """Scrapes current Nasdaq-100 constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    resp = requests.get(url, headers=headers)
    tables = pd.read_html(io.StringIO(resp.text))
    df = tables[0]
    tickers = df['Ticker'].str.replace('.', '-', regex=False).tolist()
    return sorted(list(set(tickers)))

def get_sp500_tickers() -> list:
    """Scrapes current S&P 500 constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    resp = requests.get(url, headers=headers)
    tables = pd.read_html(io.StringIO(resp.text))
    df = tables[0]
    tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
    return sorted(list(set(tickers)))

def fetch_clean_data(tickers: list, benchmark_tickers: list, start: str = "2017-01-01", 
                     end: str = "2026-10-10", cache_path: str = None) -> pd.DataFrame:
    """
    Downloads historical adjusted closing prices from Yahoo Finance in batches.
    Caches to disk as parquet to avoid redundant network requests.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached real price data from {cache_path}...")
        return pd.read_parquet(cache_path)
    
    all_tickers = sorted(list(set(tickers + benchmark_tickers)))
    print(f"Downloading real price data for {len(all_tickers)} symbols ({start} to {end})...")
    
    batch_size = 50
    dfs = []
    for i in range(0, len(all_tickers), batch_size):
        sub = all_tickers[i:i+batch_size]
        print(f"  Downloading batch {i//batch_size + 1}/{(len(all_tickers)-1)//batch_size + 1} ({len(sub)} symbols)...")
        try:
            d = yf.download(sub, start=start, end=end, auto_adjust=True, progress=False)['Close']
            if isinstance(d, pd.Series):
                d = d.to_frame()
            dfs.append(d)
        except Exception as e:
            print(f"Error downloading batch: {e}")
        time.sleep(0.3)
        
    combined = pd.concat(dfs, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    
    # Filter tickers with >= 80% non-null history, forward-fill and back-fill
    missing = combined.isnull().mean()
    valid = missing[missing < 0.20].index
    clean_df = combined[valid].ffill().bfill()
    
    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        clean_df.to_parquet(cache_path)
        print(f"Saved {clean_df.shape[1]} symbols across {clean_df.shape[0]} trading days to {cache_path}.")
        
    return clean_df
