"""
Production Live Signal Generator
Uses real options data and production model for trading signals.
"""

import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.data_loader import load_spy_with_vix
from src.features import add_enhanced_features
from src.options_data import (
    fetch_options_chain, get_atm_options, get_next_weekly_expiry,
    calculate_put_call_ratio, get_options_features
)


# Action constants (7 actions)
ACTION_NAMES = {
    0: "[+] BUY CALL",
    1: "[-] BUY PUT",
    2: "[^] BULL SPREAD",
    3: "[v] BEAR SPREAD",
    4: "[=] IRON CONDOR",
    5: "[.] HOLD",
    6: "[X] CLOSE"
}


class ProductionSignalGenerator:
    """
    Generates production-grade trading signals.
    
    Features:
    - Real options data from yfinance
    - 7-action support (spreads, condors)
    - Detailed trade setups
    - PostgreSQL storage for RL retraining
    """
    
    def __init__(
        self,
        model_path: str = "models/production/ppo_production_best.zip",
        stop_loss_pct: float = 0.30,
        target_pct: float = 0.50,
        default_dte: int = 7,
        spread_width: int = 5,
        capital: float = 10_000,
        risk_per_trade: float = 0.02,
        use_database: bool = True
    ):
        self.stop_loss_pct = stop_loss_pct
        self.target_pct = target_pct
        self.default_dte = default_dte
        self.spread_width = spread_width
        self.capital = capital
        self.risk_per_trade = risk_per_trade
        
        # Database connection
        self.db = None
        self.use_database = use_database
        if use_database:
            try:
                from src.database import TradingDatabase
                self.db = TradingDatabase()
                self.db.connect()
                print("Connected to PostgreSQL database")
            except Exception as e:
                print(f"Database connection failed: {e}")
                self.db = None
        
        # Try loading production model, fall back to basic
        if os.path.exists(model_path):
            self.model = PPO.load(model_path)
            self.model_type = "production"
        elif os.path.exists("models/ppo_spy_options.zip"):
            self.model = PPO.load("models/ppo_spy_options.zip")
            self.model_type = "basic"
        else:
            raise FileNotFoundError("No trained model found. Run training first.")
        
        print(f"Loaded {self.model_type} model")
    
    def get_market_state(self) -> Dict[str, Any]:
        """Get current market data with real options."""
        # Historical data for features
        df = load_spy_with_vix(years=1)
        df = add_enhanced_features(df)
        latest = df.iloc[-1]
        
        # Real options data
        try:
            options = get_options_features()
            has_real_options = True
        except Exception as e:
            print(f"Warning: Could not fetch options data: {e}")
            options = {
                'spot': latest['Close'],
                'iv': latest.get('IV', 0.18),
                'put_call_ratio': 1.0,
                'gamma_exposure': 0,
                'iv_skew': 0
            }
            has_real_options = False
        
        return {
            'df': df,
            'latest': latest,
            'options': options,
            'has_real_options': has_real_options,
            'timestamp': datetime.now()
        }
    
    def build_state(self, market: Dict) -> np.ndarray:
        """Build 15-dimensional state vector."""
        latest = market['latest']
        options = market['options']
        
        state = np.array([
            np.clip((latest['Close'] - 400) / 400, 0, 1),
            latest['RSI'],
            np.clip(latest['MACD'] / 10 + 0.5, 0, 1),
            np.clip((latest.get('IV', 0.18) - 0.10) / 0.40, 0, 1),
            np.clip(latest.get('IV_Rank', 50) / 100, 0, 1),
            np.clip(options.get('put_call_ratio', 1) / 3, 0, 1),
            np.clip(latest.get('Volume_Surge', 1) / 5, 0, 1),
            np.clip(options.get('iv_skew', 0) / 10 + 0.5, 0, 1),
            np.clip(latest.get('Gamma_Proxy', 0) + 0.5, 0, 1),
            np.clip(latest.get('Momentum', 0) / 10 + 0.5, 0, 1),
            0.5,  # Balance (normalized)
            0.0,  # No position
            0.0,  # Position type
            0.5,  # Position P&L
            0.0   # Holding time
        ], dtype=np.float32)
        
        # Handle 10-dim model (basic model)
        if self.model_type == "basic":
            state = state[:10]
        
        return np.clip(state, 0, 1)
    
    def get_trade_setup(self, action: int, spot: float, chain: Dict) -> Optional[Dict]:
        """Get detailed trade setup for an action."""
        if action >= 5:  # HOLD or CLOSE
            return None
        
        atm = get_atm_options(chain) if chain else None
        expiry = get_next_weekly_expiry(self.default_dte)
        expiry_date = datetime.strptime(expiry, '%Y-%m-%d')
        
        setup = {
            'action': ACTION_NAMES[action],
            'expiry': expiry,
            'expiry_dte': (expiry_date - datetime.now()).days,
            'spot': spot
        }
        
        if action == 0:  # BUY CALL
            if atm and atm.get('call'):
                call = atm['call']
                strike = call['strike']
                entry = (call['bid'] + call['ask']) / 2 if call['ask'] > 0 else call['bid'] * 1.02
            else:
                strike = round(spot)
                entry = spot * 0.015  # Estimate 1.5% of spot
            
            setup.update({
                'type': 'CALL',
                'strike': strike,
                'symbol': f"SPY {expiry_date.strftime('%y%m%d')} C {int(strike)}",
                'entry': round(entry, 2),
                'stop_loss': round(entry * (1 - self.stop_loss_pct), 2),
                'target': round(entry * (1 + self.target_pct), 2),
                'max_loss': round(entry * 100, 2),
                'direction': 'BULLISH'
            })
            
        elif action == 1:  # BUY PUT
            if atm and atm.get('put'):
                put = atm['put']
                strike = put['strike']
                entry = (put['bid'] + put['ask']) / 2 if put['ask'] > 0 else put['bid'] * 1.02
            else:
                strike = round(spot)
                entry = spot * 0.015
            
            setup.update({
                'type': 'PUT',
                'strike': strike,
                'symbol': f"SPY {expiry_date.strftime('%y%m%d')} P {int(strike)}",
                'entry': round(entry, 2),
                'stop_loss': round(entry * (1 - self.stop_loss_pct), 2),
                'target': round(entry * (1 + self.target_pct), 2),
                'max_loss': round(entry * 100, 2),
                'direction': 'BEARISH'
            })
            
        elif action == 2:  # BULL SPREAD
            long_strike = round(spot)
            short_strike = long_strike + self.spread_width
            # Estimate debit (cost) as 40% of width
            debit = self.spread_width * 0.4
            breakeven = long_strike + debit
            
            setup.update({
                'type': 'BULL CALL SPREAD',
                'long_strike': long_strike,
                'short_strike': short_strike,
                'symbol': f"SPY {expiry_date.strftime('%y%m%d')} C {long_strike}/{short_strike}",
                'entry': round(debit, 2),
                'max_profit': round(self.spread_width * 100 - debit * 100, 2),
                'max_loss': round(debit * 100, 2),
                'breakeven': round(breakeven, 2),
                # SPY levels for analysis
                'spy_target': round(short_strike, 2),  # Max profit at short strike
                'spy_stop': round(long_strike - debit, 2),  # Close if below breakeven - buffer
                'direction': 'BULLISH',
                'risk_reward': f"1:{round((self.spread_width - debit) / debit, 1)}"
            })
            
        elif action == 3:  # BEAR SPREAD
            long_strike = round(spot)
            short_strike = long_strike - self.spread_width
            # Estimate debit as 40% of width
            debit = self.spread_width * 0.4
            breakeven = long_strike - debit
            
            setup.update({
                'type': 'BEAR PUT SPREAD',
                'long_strike': long_strike,
                'short_strike': short_strike,
                'symbol': f"SPY {expiry_date.strftime('%y%m%d')} P {long_strike}/{short_strike}",
                'entry': round(debit, 2),
                'max_profit': round(self.spread_width * 100 - debit * 100, 2),
                'max_loss': round(debit * 100, 2),
                'breakeven': round(breakeven, 2),
                # SPY levels for analysis
                'spy_target': round(short_strike, 2),  # Max profit at short strike
                'spy_stop': round(long_strike + debit, 2),  # Close if above breakeven + buffer
                'direction': 'BEARISH',
                'risk_reward': f"1:{round((self.spread_width - debit) / debit, 1)}"
            })
            
        elif action == 4:  # IRON CONDOR
            call_short = round(spot) + self.spread_width
            call_long = call_short + self.spread_width
            put_short = round(spot) - self.spread_width
            put_long = put_short - self.spread_width
            # Estimate credit as 30% of one wing width
            credit = self.spread_width * 0.3
            
            setup.update({
                'type': 'IRON CONDOR',
                'call_spread': f"{call_short}/{call_long}",
                'put_spread': f"{put_long}/{put_short}",
                'symbol': f"SPY {expiry_date.strftime('%y%m%d')} IC {put_long}/{put_short}/{call_short}/{call_long}",
                'entry': round(credit, 2),  # Credit received
                'max_profit': round(credit * 100, 2),
                'max_loss': round((self.spread_width - credit) * 100, 2),
                # SPY levels for analysis
                'spy_upper_stop': round(call_short + credit, 2),  # Stop if above
                'spy_lower_stop': round(put_short - credit, 2),   # Stop if below
                'spy_profit_high': round(call_short, 2),
                'spy_profit_low': round(put_short, 2),
                'direction': 'NEUTRAL',
                'profit_zone': f"${put_short} - ${call_short}"
            })
        
        return setup
    
    def generate_signal(self) -> Dict[str, Any]:
        """Generate complete trading signal."""
        market = self.get_market_state()
        state = self.build_state(market)
        
        # Get prediction
        action, _ = self.model.predict(state, deterministic=False)
        action = int(action)
        
        # Get probabilities
        obs_tensor = self.model.policy.obs_to_tensor(state.reshape(1, -1))[0]
        dist = self.model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.detach().numpy()[0]
        
        spot = market['options'].get('spot', 0)
        if spot == 0:
            spot = market['latest']['Close']  # Fallback to latest close
        
        # Get options chain for trade setup
        try:
            chain = fetch_options_chain()
        except:
            chain = None
        
        signal = {
            'timestamp': market['timestamp'].isoformat(),
            'spot': spot,
            'iv': market['options'].get('iv', 0.18) * 100,
            'put_call_ratio': market['options'].get('put_call_ratio', 1),
            'rsi': market['latest']['RSI'] * 100,
            'macd': market['latest']['MACD'],
            'iv_rank': market['latest'].get('IV_Rank', 50),
            'volume_surge': market['latest'].get('Volume_Surge', 1),
            'action': action,
            'action_name': ACTION_NAMES.get(action, f"Action {action}"),
            'confidence': float(probs[action] * 100) if action < len(probs) else 0,
            'probabilities': {ACTION_NAMES.get(i, f"Action {i}"): float(p * 100) 
                            for i, p in enumerate(probs)},
            'trade_setup': self.get_trade_setup(action, spot, chain),
            'has_real_options': market['has_real_options']
        }
        
        # Store signal in database for RL retraining
        if self.db:
            try:
                signal_id = self.db.store_signal(
                    signal=signal,
                    state_vector=state,
                    model_version=self.model_type
                )
                signal['db_id'] = signal_id
                print(f"Signal stored in DB (id={signal_id})")
            except Exception as e:
                print(f"Failed to store signal: {e}")
        
        return signal
    
    def print_signal(self, signal: Dict):
        """Print formatted signal."""
        print("")
        print("=" * 60)
        print(">>> SPY OPTIONS SIGNAL <<<")
        print("=" * 60)
        print(f"Time: {signal['timestamp']}")
        print("")
        print("MARKET DATA")
        print(f"   SPY Price:     ${signal['spot']:.2f}")
        print(f"   IV:            {signal['iv']:.1f}%")
        print(f"   IV Rank:       {signal['iv_rank']:.0f}")
        print(f"   Put/Call:      {signal['put_call_ratio']:.2f}")
        print(f"   RSI:           {signal['rsi']:.1f}")
        print(f"   MACD:          {signal['macd']:.2f}")
        print("")
        print("AI RECOMMENDATION")
        print(f"   Signal:        {signal['action_name']}")
        print(f"   Confidence:    {signal['confidence']:.1f}%")
        print("")
        print("PROBABILITIES")
        for name, prob in sorted(signal['probabilities'].items(), key=lambda x: -x[1]):
            bar = "#" * int(prob / 5) + "-" * (20 - int(prob / 5))
            print(f"   {name[:12]:12s} [{bar}] {prob:.1f}%")
        
        if signal['trade_setup']:
            setup = signal['trade_setup']
            spot = signal['spot']
            
            print("")
            print("=" * 60)
            print(">>> OPTION CONTRACT <<<")
            print("=" * 60)
            print(f"   SYMBOL:     {setup['symbol']}")
            print(f"   Type:       {setup['type']}")
            
            if 'strike' in setup:
                # Single leg options
                print(f"   Strike:     ${setup['strike']:.0f}")
                print(f"   Expiry:     {setup['expiry']} ({setup['expiry_dte']} DTE)")
                print(f"   Delta:      0.50 (ATM est.)")
                print("")
                print(">>> OPTION PRICE LEVELS <<<")
                print(f"   [ENTRY]      ${setup['entry']:.2f}")
                print(f"   [STOP LOSS]  ${setup['stop_loss']:.2f} (-{self.stop_loss_pct*100:.0f}%)")
                print(f"   [TARGET]     ${setup['target']:.2f} (+{self.target_pct*100:.0f}%)")
                print("")
                print(">>> SPY PRICE LEVELS <<<")
                print(f"   Current:    ${spot:.2f}")
                # Calculate SPY targets based on option direction
                if setup['direction'] == 'BULLISH':
                    spy_target = spot * 1.012  # ~1.2% move for 50% option gain
                    spy_stop = spot * 0.992
                else:
                    spy_target = spot * 0.988
                    spy_stop = spot * 1.008
                print(f"   Target:     ${spy_target:.2f}")
                print(f"   Stop:       ${spy_stop:.2f}")
                print("")
                print(">>> POSITION SIZE <<<")
                risk_amount = self.capital * self.risk_per_trade
                max_loss_per = setup['entry'] * self.stop_loss_pct * 100
                contracts = max(1, int(risk_amount / max_loss_per)) if max_loss_per > 0 else 1
                print(f"   Contracts:   {contracts}")
                print(f"   Max Risk:    ${setup['entry'] * self.stop_loss_pct * 100 * contracts:.2f}")
                print(f"   Max Profit:  ${setup['entry'] * self.target_pct * 100 * contracts:.2f}")
                
            elif 'long_strike' in setup:
                # Vertical spread
                print(f"   Long:       ${setup['long_strike']:.0f}")
                print(f"   Short:      ${setup['short_strike']:.0f}")
                print(f"   Width:      ${self.spread_width}")
                print(f"   Expiry:     {setup['expiry']} ({setup['expiry_dte']} DTE)")
                print("")
                print(">>> SPREAD PRICING <<<")
                print(f"   [ENTRY]      ${setup.get('entry', 2.0):.2f} (debit)")
                print(f"   [BREAKEVEN]  ${setup.get('breakeven', 0):.2f}")
                print("")
                print(">>> DEFINED RISK <<<")
                print(f"   [MAX PROFIT] ${setup['max_profit']:.2f}")
                print(f"   [MAX LOSS]   ${setup['max_loss']:.2f}")
                print(f"   R:R:         {setup.get('risk_reward', '1:1.5')}")
                print("")
                print(">>> SPY PRICE TARGETS <<<")
                print(f"   Current:    ${spot:.2f}")
                print(f"   [TARGET]    ${setup.get('spy_target', setup['short_strike']):.2f}  ← Max profit here")
                print(f"   [STOP]      ${setup.get('spy_stop', spot):.2f}  ← Exit if breached")
                print(f"   Breakeven:  ${setup.get('breakeven', 0):.2f}")
                print("")
                print(">>> POSITION SIZE <<<")
                contracts = max(1, int(self.capital * self.risk_per_trade / setup['max_loss']))
                print(f"   Contracts:   {contracts}")
                print(f"   Max Risk:    ${setup['max_loss'] * contracts:.2f}")
                print(f"   Max Profit:  ${setup['max_profit'] * contracts:.2f}")
                
            elif setup['type'] == 'IRON CONDOR':
                print(f"   Call Spread: {setup['call_spread']}")
                print(f"   Put Spread:  {setup['put_spread']}")
                print(f"   Expiry:      {setup['expiry']} ({setup['expiry_dte']} DTE)")
                print("")
                print(">>> CREDIT RECEIVED <<<")
                print(f"   [ENTRY]      ${setup.get('entry', 1.5):.2f} (credit)")
                print("")
                print(">>> DEFINED RISK <<<")
                print(f"   [MAX PROFIT] ${setup['max_profit']:.2f}")
                print(f"   [MAX LOSS]   ${setup['max_loss']:.2f}")
                print("")
                print(">>> SPY PRICE TARGETS <<<")
                print(f"   Current:     ${spot:.2f}")
                print(f"   [UPPER STOP] ${setup.get('spy_upper_stop', spot + 6):.2f}  ← Exit if SPY above")
                print(f"   [LOWER STOP] ${setup.get('spy_lower_stop', spot - 6):.2f}  ← Exit if SPY below")
                print(f"   Profit Zone: {setup['profit_zone']}")
                print("")
                print(">>> POSITION SIZE <<<")
                contracts = max(1, int(self.capital * self.risk_per_trade / setup['max_loss']))
                print(f"   Contracts:   {contracts}")
                print(f"   Max Risk:    ${setup['max_loss'] * contracts:.2f}")
                print(f"   Max Profit:  ${setup['max_profit'] * contracts:.2f}")
        
        print("")
        print("=" * 60)
    
    def run_loop(self, interval: int = 300, max_signals: Optional[int] = None):
        """Run continuous signal generation."""
        print("Starting Production Signal Generator...")
        print(f"Interval: {interval}s")
        
        count = 0
        try:
            while max_signals is None or count < max_signals:
                signal = self.generate_signal()
                self.print_signal(signal)
                
                count += 1
                
                if max_signals is None or count < max_signals:
                    print(f"\nNext signal in {interval}s...")
                    time.sleep(interval)
        
        except KeyboardInterrupt:
            print("\n\nSignal generator stopped.")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Production Signal Generator")
    parser.add_argument("--model", type=str, default="models/production/ppo_production_best.zip")
    parser.add_argument("--capital", type=float, default=10_000)
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--stop-loss", type=float, default=0.30)
    parser.add_argument("--target", type=float, default=0.50)
    
    args = parser.parse_args()
    
    generator = ProductionSignalGenerator(
        model_path=args.model,
        stop_loss_pct=args.stop_loss,
        target_pct=args.target,
        capital=args.capital
    )
    
    if args.once:
        signal = generator.generate_signal()
        generator.print_signal(signal)
    else:
        generator.run_loop(interval=args.interval)


if __name__ == "__main__":
    main()
