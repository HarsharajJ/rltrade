"""
SPY Data Loader Module
Downloads historical SPY OHLCV data via yfinance and provides train/test splitting.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Optional


def download_spy_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    years: int = 2
) -> pd.DataFrame:
    """
    Download SPY historical OHLCV data from Yahoo Finance.
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format. If None, calculated from years.
        end_date: End date in 'YYYY-MM-DD' format. If None, uses today.
        years: Number of years of data to fetch (used if start_date is None).
    
    Returns:
        DataFrame with OHLCV data indexed by date.
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    if start_date is None:
        start = datetime.now() - timedelta(days=years * 365)
        start_date = start.strftime('%Y-%m-%d')
    
    print(f"Downloading SPY data from {start_date} to {end_date}...")
    
    ticker = yf.Ticker("SPY")
    df = ticker.history(start=start_date, end=end_date, interval="1d")
    
    # Clean up the DataFrame
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = 'Date'
    
    # Remove any NaN values
    df = df.dropna()
    
    print(f"Downloaded {len(df)} bars of SPY data")
    return df


def split_train_test(
    df: pd.DataFrame,
    train_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-based train/test split to avoid data leakage.
    
    Args:
        df: Full DataFrame with features.
        train_ratio: Fraction of data to use for training.
    
    Returns:
        Tuple of (train_df, test_df).
    """
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    print(f"Train set: {len(train_df)} bars ({train_df.index[0]} to {train_df.index[-1]})")
    print(f"Test set: {len(test_df)} bars ({test_df.index[0]} to {test_df.index[-1]})")
    
    return train_df, test_df


def load_spy_data(years: int = 2) -> pd.DataFrame:
    """
    Convenience function to load SPY data with default settings.
    
    Args:
        years: Number of years of historical data.
    
    Returns:
        DataFrame with SPY OHLCV data.
    """
    return download_spy_data(years=years)


def download_vix_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    years: int = 2
) -> pd.DataFrame:
    """
    Download VIX (CBOE Volatility Index) data for implied volatility proxy.
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format.
        end_date: End date in 'YYYY-MM-DD' format.
        years: Number of years of data to fetch.
    
    Returns:
        DataFrame with VIX close values.
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    if start_date is None:
        start = datetime.now() - timedelta(days=years * 365)
        start_date = start.strftime('%Y-%m-%d')
    
    print(f"Downloading VIX data from {start_date} to {end_date}...")
    
    ticker = yf.Ticker("^VIX")
    df = ticker.history(start=start_date, end=end_date, interval="1d")
    
    df = df[['Close']].copy()
    df.columns = ['VIX']
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = 'Date'
    df = df.dropna()
    
    # Convert VIX to decimal (e.g., 18 -> 0.18)
    df['IV'] = df['VIX'] / 100.0
    
    print(f"Downloaded {len(df)} bars of VIX data")
    return df


def load_spy_with_vix(years: int = 2) -> pd.DataFrame:
    """
    Load SPY data with VIX for implied volatility.
    
    Returns:
        DataFrame with SPY OHLCV and IV columns.
    """
    spy_df = download_spy_data(years=years)
    vix_df = download_vix_data(years=years)
    
    # Merge on date index
    df = spy_df.join(vix_df[['VIX', 'IV']], how='left')
    
    # Forward fill any missing VIX values
    df['VIX'] = df['VIX'].ffill().bfill()
    df['IV'] = df['IV'].ffill().bfill()
    
    print(f"Combined SPY+VIX data: {len(df)} bars")
    return df


if __name__ == "__main__":
    # Test the data loader
    df = load_spy_data(years=2)
    print("\nSample data:")
    print(df.head())
    print(f"\nData range: {df.index[0]} to {df.index[-1]}")
    print(f"Total bars: {len(df)}")
