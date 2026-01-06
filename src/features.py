"""
Feature Engineering Module
Computes technical indicators and builds normalized state vectors for the RL agent.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import pickle
from pathlib import Path


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index (RSI).
    
    Args:
        prices: Series of closing prices.
        period: RSI lookback period (default 14).
    
    Returns:
        RSI values normalized to [0, 1].
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    
    return rsi / 100.0  # Normalize to [0, 1]


def compute_macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> pd.Series:
    """
    Compute MACD (Moving Average Convergence Divergence).
    Returns the MACD histogram (MACD line - Signal line).
    
    Args:
        prices: Series of closing prices.
        fast: Fast EMA period.
        slow: Slow EMA period.
        signal: Signal line period.
    
    Returns:
        MACD histogram values.
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    
    return macd_histogram


def compute_ema(prices: pd.Series, period: int = 20) -> pd.Series:
    """
    Compute Exponential Moving Average.
    
    Args:
        prices: Series of closing prices.
        period: EMA period.
    
    Returns:
        EMA values.
    """
    return prices.ewm(span=period, adjust=False).mean()


def compute_volatility(prices: pd.Series, period: int = 20) -> pd.Series:
    """
    Compute rolling volatility (standard deviation of returns).
    
    Args:
        prices: Series of closing prices.
        period: Rolling window for volatility calculation.
    
    Returns:
        Rolling volatility values.
    """
    returns = prices.pct_change()
    volatility = returns.rolling(window=period, min_periods=1).std()
    return volatility


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators to the DataFrame.
    
    Args:
        df: DataFrame with OHLCV data.
    
    Returns:
        DataFrame with additional feature columns.
    """
    df = df.copy()
    
    # Compute indicators
    df['RSI'] = compute_rsi(df['Close'], period=14)
    df['MACD'] = compute_macd(df['Close'], fast=12, slow=26, signal=9)
    df['EMA20'] = compute_ema(df['Close'], period=20)
    df['Volatility'] = compute_volatility(df['Close'], period=20)
    
    # Forward fill any NaN values from indicator warm-up
    df = df.ffill().bfill()
    
    return df


def compute_iv_rank(iv_series: pd.Series, lookback: int = 252) -> pd.Series:
    """
    Compute IV Rank: where current IV stands in its 52-week range.
    
    Returns:
        IV Rank as 0-100 value.
    """
    def calc_rank(window):
        if len(window) < 10:
            return 50.0
        current = window.iloc[-1]
        low = window.min()
        high = window.max()
        if high == low:
            return 50.0
        return (current - low) / (high - low) * 100
    
    return iv_series.rolling(window=lookback, min_periods=10).apply(calc_rank, raw=False).fillna(50)


def compute_put_call_ratio_proxy(df: pd.DataFrame) -> pd.Series:
    """
    Estimate put/call ratio from price action.
    Uses volume on down days vs up days as proxy.
    
    Returns:
        Ratio where > 1 = bearish, < 1 = bullish.
    """
    returns = df['Close'].pct_change()
    volume = df['Volume']
    
    # Volume on down days
    down_vol = volume.where(returns < 0, 0).rolling(10).sum()
    # Volume on up days
    up_vol = volume.where(returns > 0, 0).rolling(10).sum()
    
    pcr = down_vol / (up_vol + 1)  # Avoid division by zero
    return pcr.fillna(1.0)


def compute_volume_surge(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """
    Compute volume surge: current volume vs average.
    
    Returns:
        Ratio where > 2 = unusual activity.
    """
    avg_volume = df['Volume'].rolling(lookback).mean()
    surge = df['Volume'] / (avg_volume + 1)
    return surge.fillna(1.0)


def compute_iv_skew_proxy(df: pd.DataFrame) -> pd.Series:
    """
    Estimate IV skew from price action (proxy).
    Uses realized volatility asymmetry.
    
    Positive = puts more expensive (fear)
    Negative = calls more expensive (greed)
    """
    returns = df['Close'].pct_change()
    
    # Volatility of down moves
    down_vol = returns.where(returns < 0, 0).rolling(20).std()
    # Volatility of up moves
    up_vol = returns.where(returns > 0, 0).rolling(20).std()
    
    skew = (down_vol - up_vol) * 100  # Scale up
    return skew.fillna(0)


def compute_gamma_proxy(df: pd.DataFrame) -> pd.Series:
    """
    Estimate gamma exposure from price action.
    
    Negative when at round numbers (dealer short gamma)
    Positive when between strikes (dealer long gamma)
    """
    close = df['Close']
    
    # Distance to nearest $5 strike
    nearest_strike = (close / 5).round() * 5
    distance = (close - nearest_strike).abs() / 5
    
    # Gamma highest near strikes, lower between
    gamma = distance - 0.5  # Range: -0.5 to +0.5
    
    return gamma.fillna(0)


def add_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add enhanced options-focused features to DataFrame.
    
    New features:
    - IV_Rank: Where IV stands in 52-week range
    - PCR: Put/Call ratio proxy
    - Volume_Surge: Unusual volume detection
    - IV_Skew: Put vs call IV proxy
    - Gamma_Proxy: Dealer gamma exposure estimate
    
    Args:
        df: DataFrame with OHLCV and IV columns.
    
    Returns:
        DataFrame with enhanced features (15 total).
    """
    df = df.copy()
    
    # First add basic features
    df = add_features(df)
    
    # Ensure IV column exists
    if 'IV' not in df.columns:
        df['IV'] = 0.18  # Default
    
    # Add enhanced features
    df['IV_Rank'] = compute_iv_rank(df['IV'], lookback=252)
    df['PCR'] = compute_put_call_ratio_proxy(df)
    df['Volume_Surge'] = compute_volume_surge(df, lookback=20)
    df['IV_Skew'] = compute_iv_skew_proxy(df)
    df['Gamma_Proxy'] = compute_gamma_proxy(df)
    
    # EMA ratios
    df['EMA_Ratio'] = df['Close'] / df['EMA20']
    
    # Momentum
    df['Momentum'] = df['Close'].pct_change(5) * 100  # 5-day momentum
    
    # Forward fill NaN
    df = df.ffill().bfill()
    
    return df


def normalize_features(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    feature_columns: Optional[list] = None,
    scaler_path: Optional[str] = None
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], StandardScaler]:
    """
    Normalize features using StandardScaler fitted only on training data.
    
    Args:
        train_df: Training DataFrame with features.
        test_df: Optional test DataFrame to normalize with same scaler.
        feature_columns: List of columns to normalize.
        scaler_path: Optional path to save the scaler.
    
    Returns:
        Tuple of (normalized_train_df, normalized_test_df, scaler).
    """
    if feature_columns is None:
        feature_columns = ['Close', 'RSI', 'MACD', 'EMA20', 'Volatility']
    
    scaler = StandardScaler()
    
    # Fit on training data only
    train_normalized = train_df.copy()
    train_values = scaler.fit_transform(train_df[feature_columns])
    
    # Create normalized column names
    for i, col in enumerate(feature_columns):
        train_normalized[f'{col}_norm'] = train_values[:, i]
    
    # Transform test data with the same scaler
    test_normalized = None
    if test_df is not None:
        test_normalized = test_df.copy()
        test_values = scaler.transform(test_df[feature_columns])
        for i, col in enumerate(feature_columns):
            test_normalized[f'{col}_norm'] = test_values[:, i]
    
    # Save scaler if path provided
    if scaler_path:
        Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved to {scaler_path}")
    
    return train_normalized, test_normalized, scaler


def build_state_vector(
    row: pd.Series,
    balance_normalized: float = 0.5
) -> np.ndarray:
    """
    Build the normalized state vector for a single timestep.
    
    State vector: [Close_norm, RSI_norm, MACD_norm, EMA20_norm, Volatility_norm, Balance_norm]
    
    Args:
        row: DataFrame row with normalized features.
        balance_normalized: Normalized balance value (0-1).
    
    Returns:
        NumPy array of shape (6,) with state values.
    """
    state = np.array([
        np.clip(row['Close_norm'], -3, 3) / 6 + 0.5,  # Scale to ~[0, 1]
        row['RSI_norm'] if 'RSI_norm' in row else row['RSI'],  # Already 0-1
        np.clip(row['MACD_norm'], -3, 3) / 6 + 0.5,  # Scale to ~[0, 1]
        np.clip(row['EMA20_norm'], -3, 3) / 6 + 0.5,  # Scale to ~[0, 1]
        np.clip(row['Volatility_norm'], -3, 3) / 6 + 0.5,  # Scale to ~[0, 1]
        balance_normalized
    ], dtype=np.float32)
    
    return np.clip(state, 0, 1)


def prepare_training_data(
    years: int = 2,
    train_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline to prepare training and test data with features.
    
    Args:
        years: Years of historical data to download.
        train_ratio: Fraction of data for training.
    
    Returns:
        Tuple of (train_df, test_df) with normalized features.
    """
    from src.data_loader import download_spy_data, split_train_test
    
    # Download data
    df = download_spy_data(years=years)
    
    # Add technical indicators
    df = add_features(df)
    
    # Split into train/test (time-based)
    train_raw, test_raw = split_train_test(df, train_ratio=train_ratio)
    
    # Normalize features (fit only on training data)
    train_df, test_df, _ = normalize_features(
        train_raw,
        test_raw,
        scaler_path='models/feature_scaler.pkl'
    )
    
    return train_df, test_df


if __name__ == "__main__":
    # Test feature engineering
    train_df, test_df = prepare_training_data(years=2)
    print("\nTraining data sample:")
    print(train_df[['Close', 'RSI', 'MACD', 'EMA20', 'Volatility', 
                    'Close_norm', 'RSI_norm']].head())
    print(f"\nFeature columns: {train_df.columns.tolist()}")
