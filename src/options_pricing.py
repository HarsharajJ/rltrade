"""
Options Pricing Module
Black-Scholes model for simulating SPY options prices and Greeks.
"""

import numpy as np
from scipy.stats import norm
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class OptionContract:
    """Represents an options contract."""
    option_type: str  # 'call' or 'put'
    strike: float
    expiry_days: int  # Days to expiration
    premium: float
    delta: float
    gamma: float
    theta: float
    vega: float
    
    @property
    def dte(self) -> int:
        """Days to expiration."""
        return self.expiry_days


def black_scholes_call(
    S: float,      # Current stock price
    K: float,      # Strike price
    r: float,      # Risk-free rate (annual)
    sigma: float,  # Volatility (annual)
    T: float       # Time to expiration (in years)
) -> float:
    """
    Calculate Black-Scholes call option price.
    
    Args:
        S: Current stock price
        K: Strike price
        r: Risk-free interest rate (annual, e.g., 0.05 for 5%)
        sigma: Volatility (annual, e.g., 0.20 for 20%)
        T: Time to expiration in years (e.g., 30/365 for 30 days)
    
    Returns:
        Call option price.
    """
    if T <= 0:
        return max(S - K, 0)  # Intrinsic value at expiration
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return max(call_price, 0.01)  # Minimum premium


def black_scholes_put(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float
) -> float:
    """
    Calculate Black-Scholes put option price.
    
    Returns:
        Put option price.
    """
    if T <= 0:
        return max(K - S, 0)  # Intrinsic value at expiration
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return max(put_price, 0.01)


def calculate_greeks(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = 'call'
) -> Dict[str, float]:
    """
    Calculate option Greeks.
    
    Returns:
        Dictionary with delta, gamma, theta, vega.
    """
    if T <= 0:
        # At expiration
        if option_type == 'call':
            delta = 1.0 if S > K else 0.0
        else:
            delta = -1.0 if S < K else 0.0
        return {'delta': delta, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0}
    
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    
    # Delta
    if option_type == 'call':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1
    
    # Gamma (same for calls and puts)
    gamma = norm.pdf(d1) / (S * sigma * sqrt_T)
    
    # Theta (daily)
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * sqrt_T)
    if option_type == 'call':
        term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
    theta = (term1 + term2) / 365  # Convert to daily
    
    # Vega (for 1% change in volatility)
    vega = S * sqrt_T * norm.pdf(d1) / 100
    
    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega
    }


def price_option(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = 'call'
) -> Tuple[float, Dict[str, float]]:
    """
    Price an option and calculate its Greeks.
    
    Returns:
        Tuple of (premium, greeks_dict).
    """
    if option_type == 'call':
        premium = black_scholes_call(S, K, r, sigma, T)
    else:
        premium = black_scholes_put(S, K, r, sigma, T)
    
    greeks = calculate_greeks(S, K, r, sigma, T, option_type)
    
    return premium, greeks


def select_atm_strike(spot_price: float, strike_interval: float = 1.0) -> float:
    """
    Select the at-the-money strike price.
    
    Args:
        spot_price: Current underlying price.
        strike_interval: Strike price interval (e.g., 1 for SPY).
    
    Returns:
        ATM strike price.
    """
    return round(spot_price / strike_interval) * strike_interval


def create_option_contract(
    spot_price: float,
    volatility: float,
    days_to_expiry: int,
    option_type: str = 'call',
    moneyness: float = 0.0,  # 0 = ATM, positive = OTM, negative = ITM
    risk_free_rate: float = 0.05
) -> OptionContract:
    """
    Create an option contract with pricing and Greeks.
    
    Args:
        spot_price: Current underlying price.
        volatility: Implied volatility (annual, e.g., 0.20).
        days_to_expiry: Days until expiration.
        option_type: 'call' or 'put'.
        moneyness: Strike offset as percentage (0 = ATM).
        risk_free_rate: Annual risk-free rate.
    
    Returns:
        OptionContract with all details.
    """
    # Calculate strike based on moneyness
    strike = select_atm_strike(spot_price)
    if option_type == 'call':
        strike = strike * (1 + moneyness)
    else:
        strike = strike * (1 - moneyness)
    
    T = days_to_expiry / 365.0
    
    premium, greeks = price_option(
        S=spot_price,
        K=strike,
        r=risk_free_rate,
        sigma=volatility,
        T=T,
        option_type=option_type
    )
    
    return OptionContract(
        option_type=option_type,
        strike=strike,
        expiry_days=days_to_expiry,
        premium=premium,
        delta=greeks['delta'],
        gamma=greeks['gamma'],
        theta=greeks['theta'],
        vega=greeks['vega']
    )


def reprice_contract(
    contract: OptionContract,
    new_spot: float,
    new_volatility: float,
    days_passed: int = 1,
    risk_free_rate: float = 0.05
) -> OptionContract:
    """
    Reprice an existing contract after time has passed.
    
    Args:
        contract: Existing option contract.
        new_spot: New underlying price.
        new_volatility: New implied volatility.
        days_passed: Days since last pricing.
        risk_free_rate: Risk-free rate.
    
    Returns:
        Updated OptionContract.
    """
    new_dte = max(contract.expiry_days - days_passed, 0)
    T = new_dte / 365.0
    
    premium, greeks = price_option(
        S=new_spot,
        K=contract.strike,
        r=risk_free_rate,
        sigma=new_volatility,
        T=T,
        option_type=contract.option_type
    )
    
    return OptionContract(
        option_type=contract.option_type,
        strike=contract.strike,
        expiry_days=new_dte,
        premium=premium,
        delta=greeks['delta'],
        gamma=greeks['gamma'],
        theta=greeks['theta'],
        vega=greeks['vega']
    )


if __name__ == "__main__":
    # Test Black-Scholes pricing
    print("=" * 50)
    print("Black-Scholes Options Pricing Test")
    print("=" * 50)
    
    # SPY at $580, ATM options, 30 DTE, 18% IV
    S, K, r, sigma, T = 580, 580, 0.05, 0.18, 30/365
    
    call_price = black_scholes_call(S, K, r, sigma, T)
    put_price = black_scholes_put(S, K, r, sigma, T)
    
    print(f"\nUnderlying: ${S}")
    print(f"Strike: ${K}")
    print(f"IV: {sigma*100:.0f}%")
    print(f"DTE: {int(T*365)} days")
    
    print(f"\nCall Price: ${call_price:.2f}")
    print(f"Put Price: ${put_price:.2f}")
    
    # Test Greeks
    call_greeks = calculate_greeks(S, K, r, sigma, T, 'call')
    print(f"\nCall Greeks:")
    print(f"  Delta: {call_greeks['delta']:.4f}")
    print(f"  Gamma: {call_greeks['gamma']:.6f}")
    print(f"  Theta: ${call_greeks['theta']:.4f}/day")
    print(f"  Vega: ${call_greeks['vega']:.4f}/1% IV")
    
    # Test contract creation
    print("\n" + "=" * 50)
    print("Contract Creation Test")
    print("=" * 50)
    
    contract = create_option_contract(
        spot_price=580,
        volatility=0.18,
        days_to_expiry=30,
        option_type='call'
    )
    
    print(f"\nCreated {contract.option_type.upper()} contract:")
    print(f"  Strike: ${contract.strike}")
    print(f"  Premium: ${contract.premium:.2f}")
    print(f"  Delta: {contract.delta:.4f}")
    print(f"  DTE: {contract.dte}")
    
    # Test repricing after 1 day
    new_contract = reprice_contract(
        contract,
        new_spot=582,  # SPY moved up $2
        new_volatility=0.17,  # IV dropped
        days_passed=1
    )
    
    print(f"\nAfter 1 day (SPY +$2, IV -1%):")
    print(f"  New Premium: ${new_contract.premium:.2f}")
    print(f"  P&L: ${(new_contract.premium - contract.premium) * 100:.2f}")
    print(f"  New DTE: {new_contract.dte}")
    
    print("\n✓ Options pricing test passed!")
