"""
Real Options Data Module
Fetches actual SPY options chains via yfinance with bid/ask, volume, OI.
"""

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import yfinance as yf


# Cache directory
CACHE_DIR = Path("cache/options")
CACHE_EXPIRY_HOURS = 1  # Refresh cache after 1 hour


def get_cache_path(symbol: str, expiry: str) -> Path:
    """Get cache file path for options chain."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = f"{symbol}_{expiry}_{datetime.now().strftime('%Y%m%d')}"
    return CACHE_DIR / f"{cache_key}.json"


def load_from_cache(cache_path: Path) -> Optional[Dict]:
    """Load options data from cache if fresh."""
    if not cache_path.exists():
        return None
    
    # Check if cache is expired
    mod_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
    if datetime.now() - mod_time > timedelta(hours=CACHE_EXPIRY_HOURS):
        return None
    
    try:
        with open(cache_path, 'r') as f:
            return json.load(f)
    except:
        return None


def save_to_cache(cache_path: Path, data: Dict):
    """Save options data to cache."""
    with open(cache_path, 'w') as f:
        json.dump(data, f)


def get_spy_expirations() -> List[str]:
    """Get available SPY expiration dates."""
    spy = yf.Ticker("SPY")
    return list(spy.options)


def get_next_weekly_expiry(target_dte: int = 7) -> str:
    """Get the next weekly expiration closest to target DTE."""
    expirations = get_spy_expirations()
    
    if not expirations:
        # Fallback: calculate approximate next Friday
        today = datetime.now()
        days_until_friday = (4 - today.weekday()) % 7
        if days_until_friday == 0:
            days_until_friday = 7
        next_friday = today + timedelta(days=days_until_friday)
        return next_friday.strftime('%Y-%m-%d')
    
    today = datetime.now().date()
    
    # Find expiration closest to target DTE
    best_expiry = expirations[0]
    best_diff = abs((datetime.strptime(expirations[0], '%Y-%m-%d').date() - today).days - target_dte)
    
    for exp in expirations:
        exp_date = datetime.strptime(exp, '%Y-%m-%d').date()
        dte = (exp_date - today).days
        diff = abs(dte - target_dte)
        
        if diff < best_diff and dte > 0:
            best_diff = diff
            best_expiry = exp
    
    return best_expiry


def fetch_options_chain(
    symbol: str = "SPY",
    expiry: Optional[str] = None,
    use_cache: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Fetch real options chain data.
    
    Returns:
        Dict with 'calls' and 'puts' DataFrames.
    """
    ticker = yf.Ticker(symbol)
    
    if expiry is None:
        expiry = get_next_weekly_expiry()
    
    # Check cache
    if use_cache:
        cache_path = get_cache_path(symbol, expiry)
        cached = load_from_cache(cache_path)
        if cached:
            return {
                'calls': pd.DataFrame(cached['calls']),
                'puts': pd.DataFrame(cached['puts']),
                'expiry': cached['expiry'],
                'spot': cached['spot']
            }
    
    # Fetch fresh data
    try:
        chain = ticker.option_chain(expiry)
        calls = chain.calls
        puts = chain.puts
        
        # Get current spot price
        spot = ticker.history(period='1d')['Close'].iloc[-1]
        
        result = {
            'calls': calls,
            'puts': puts,
            'expiry': expiry,
            'spot': float(spot)
        }
        
        # Cache the data
        if use_cache:
            cache_data = {
                'calls': calls.to_dict('records'),
                'puts': puts.to_dict('records'),
                'expiry': expiry,
                'spot': float(spot)
            }
            save_to_cache(cache_path, cache_data)
        
        return result
        
    except Exception as e:
        print(f"Error fetching options chain: {e}")
        return {'calls': pd.DataFrame(), 'puts': pd.DataFrame(), 'expiry': expiry, 'spot': 0}


def get_atm_options(
    chain: Dict[str, pd.DataFrame],
    spot: Optional[float] = None
) -> Dict[str, Dict]:
    """
    Get ATM call and put from options chain.
    
    Returns:
        Dict with 'call' and 'put' option details.
    """
    if spot is None:
        spot = chain.get('spot', 0)
    
    calls = chain['calls']
    puts = chain['puts']
    
    if calls.empty or puts.empty:
        return {'call': None, 'put': None}
    
    # Find ATM strike (closest to spot)
    calls['dist'] = abs(calls['strike'] - spot)
    puts['dist'] = abs(puts['strike'] - spot)
    
    atm_call = calls.loc[calls['dist'].idxmin()]
    atm_put = puts.loc[puts['dist'].idxmin()]
    
    return {
        'call': {
            'strike': float(atm_call['strike']),
            'bid': float(atm_call.get('bid', 0)),
            'ask': float(atm_call.get('ask', 0)),
            'mid': float((atm_call.get('bid', 0) + atm_call.get('ask', 0)) / 2),
            'volume': int(atm_call.get('volume', 0) or 0),
            'openInterest': int(atm_call.get('openInterest', 0) or 0),
            'impliedVolatility': float(atm_call.get('impliedVolatility', 0.2)),
            'delta': float(atm_call.get('delta', 0.5) if 'delta' in atm_call else 0.5),
        },
        'put': {
            'strike': float(atm_put['strike']),
            'bid': float(atm_put.get('bid', 0)),
            'ask': float(atm_put.get('ask', 0)),
            'mid': float((atm_put.get('bid', 0) + atm_put.get('ask', 0)) / 2),
            'volume': int(atm_put.get('volume', 0) or 0),
            'openInterest': int(atm_put.get('openInterest', 0) or 0),
            'impliedVolatility': float(atm_put.get('impliedVolatility', 0.2)),
            'delta': float(atm_put.get('delta', -0.5) if 'delta' in atm_put else -0.5),
        },
        'spot': spot
    }


def calculate_put_call_ratio(chain: Dict[str, pd.DataFrame]) -> float:
    """
    Calculate put/call ratio based on volume.
    
    > 1.0 = bearish sentiment
    < 1.0 = bullish sentiment
    """
    calls = chain['calls']
    puts = chain['puts']
    
    if calls.empty or puts.empty:
        return 1.0
    
    call_volume = calls['volume'].sum() or 1
    put_volume = puts['volume'].sum() or 1
    
    return put_volume / call_volume


def calculate_iv_rank(current_iv: float, iv_history: pd.Series) -> float:
    """
    Calculate IV Rank (0-100).
    
    IV Rank = (Current IV - 52w Low) / (52w High - 52w Low) * 100
    """
    if iv_history.empty or len(iv_history) < 10:
        return 50.0  # Default to middle
    
    iv_high = iv_history.max()
    iv_low = iv_history.min()
    
    if iv_high == iv_low:
        return 50.0
    
    rank = (current_iv - iv_low) / (iv_high - iv_low) * 100
    return max(0, min(100, rank))


def calculate_gamma_exposure(
    chain: Dict[str, pd.DataFrame],
    spot: float
) -> float:
    """
    Calculate net gamma exposure (GEX) - simplified.
    
    Positive GEX = dealers are long gamma (stabilizing)
    Negative GEX = dealers are short gamma (amplifying moves)
    """
    calls = chain['calls']
    puts = chain['puts']
    
    if calls.empty or puts.empty:
        return 0.0
    
    # Simplified: use OI * (proximity to spot) as gamma proxy
    def calc_gamma_proxy(options: pd.DataFrame, is_call: bool):
        gex = 0.0
        for _, row in options.iterrows():
            strike = row['strike']
            oi = row.get('openInterest', 0) or 0
            
            # Gamma is highest near ATM
            distance = abs(strike - spot) / spot
            gamma_factor = np.exp(-distance * 10)  # Higher near ATM
            
            # Calls: dealers short = negative gamma to them
            # Puts: dealers short = positive gamma to them
            if is_call:
                gex -= oi * gamma_factor * 100  # Dealers short calls
            else:
                gex += oi * gamma_factor * 100  # Dealers short puts
        
        return gex
    
    call_gex = calc_gamma_proxy(calls, is_call=True)
    put_gex = calc_gamma_proxy(puts, is_call=False)
    
    return call_gex + put_gex


def calculate_volume_surge(
    current_volume: int,
    avg_volume: float
) -> float:
    """
    Calculate volume surge ratio.
    
    > 2.0 = unusual activity
    > 3.0 = very unusual
    """
    if avg_volume <= 0:
        return 1.0
    
    return current_volume / avg_volume


def get_options_features(
    symbol: str = "SPY",
    expiry: Optional[str] = None
) -> Dict[str, float]:
    """
    Get all options-derived features for the model.
    
    Returns:
        Dict with IV, IV_rank, put_call_ratio, gamma_exposure, etc.
    """
    chain = fetch_options_chain(symbol, expiry)
    atm = get_atm_options(chain)
    
    spot = chain.get('spot', 0)
    
    # Calculate features
    pcr = calculate_put_call_ratio(chain)
    gex = calculate_gamma_exposure(chain, spot)
    
    # Get IV from ATM options
    call_iv = atm['call']['impliedVolatility'] if atm['call'] else 0.2
    put_iv = atm['put']['impliedVolatility'] if atm['put'] else 0.2
    avg_iv = (call_iv + put_iv) / 2
    
    # IV skew (puts more expensive = fear)
    iv_skew = put_iv - call_iv
    
    # Volume
    call_vol = atm['call']['volume'] if atm['call'] else 0
    put_vol = atm['put']['volume'] if atm['put'] else 0
    total_vol = call_vol + put_vol
    
    return {
        'spot': spot,
        'iv': avg_iv,
        'iv_rank': 50.0,  # Would need historical IV to calculate properly
        'put_call_ratio': pcr,
        'gamma_exposure': gex,
        'iv_skew': iv_skew,
        'atm_call_bid': atm['call']['bid'] if atm['call'] else 0,
        'atm_call_ask': atm['call']['ask'] if atm['call'] else 0,
        'atm_put_bid': atm['put']['bid'] if atm['put'] else 0,
        'atm_put_ask': atm['put']['ask'] if atm['put'] else 0,
        'atm_volume': total_vol,
        'expiry': chain.get('expiry', '')
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Real Options Data Test")
    print("=" * 60)
    
    # Get expirations
    print("\nAvailable expirations:")
    expirations = get_spy_expirations()
    for exp in expirations[:5]:
        print(f"  {exp}")
    print(f"  ... and {len(expirations) - 5} more")
    
    # Get next weekly
    next_exp = get_next_weekly_expiry(7)
    print(f"\nNext weekly expiry (~7 DTE): {next_exp}")
    
    # Fetch chain
    print("\nFetching options chain...")
    chain = fetch_options_chain("SPY", next_exp)
    print(f"Spot price: ${chain['spot']:.2f}")
    print(f"Calls: {len(chain['calls'])} strikes")
    print(f"Puts: {len(chain['puts'])} strikes")
    
    # Get ATM
    atm = get_atm_options(chain)
    print(f"\nATM Call: ${atm['call']['strike']:.0f}")
    print(f"  Bid: ${atm['call']['bid']:.2f}")
    print(f"  Ask: ${atm['call']['ask']:.2f}")
    print(f"  IV: {atm['call']['impliedVolatility']*100:.1f}%")
    
    print(f"\nATM Put: ${atm['put']['strike']:.0f}")
    print(f"  Bid: ${atm['put']['bid']:.2f}")
    print(f"  Ask: ${atm['put']['ask']:.2f}")
    print(f"  IV: {atm['put']['impliedVolatility']*100:.1f}%")
    
    # Get features
    print("\nOptions Features:")
    features = get_options_features()
    for key, value in features.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n✓ Real options data test passed!")
