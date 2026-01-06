"""
Smart Trading Advisor - Optimized Version
Single file that does everything:
- Caches data (no repeated downloads)
- Tracks positions in memory
- Alerts on entry/exit
- Monitors positions continuously
"""

import os
import time
import argparse
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

import numpy as np
import yfinance as yf
from stable_baselines3 import PPO

from src.features import add_enhanced_features


# Entry criteria thresholds
MIN_CONFIDENCE = 65.0
MIN_VOLUME_SURGE = 1.0  # Reduced for now
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# Action mapping
ACTION_NAMES = {
    0: "BUY_CALL", 1: "BUY_PUT", 2: "BULL_SPREAD",
    3: "BEAR_SPREAD", 4: "IRON_CONDOR", 5: "HOLD", 6: "CLOSE"
}


@dataclass
class Position:
    """In-memory position tracking."""
    id: int
    action_name: str
    entry_time: datetime
    entry_spy: float
    spy_target: float
    spy_stop: float
    max_profit: float
    max_loss: float
    contracts: int = 1
    
    def check_exit(self, current_spy: float) -> tuple[bool, str, float]:
        """Check if position should exit."""
        is_bullish = self.action_name in ['BUY_CALL', 'BULL_SPREAD']
        is_bearish = self.action_name in ['BUY_PUT', 'BEAR_SPREAD']
        
        if is_bullish:
            if current_spy >= self.spy_target:
                return True, "🎯 TARGET HIT", self.max_profit * 0.9
            if current_spy <= self.spy_stop:
                return True, "🛑 STOP LOSS", -self.max_loss
            pnl = (current_spy - self.entry_spy) / self.entry_spy * self.max_profit * 5
            
        elif is_bearish:
            if current_spy <= self.spy_target:
                return True, "🎯 TARGET HIT", self.max_profit * 0.9
            if current_spy >= self.spy_stop:
                return True, "🛑 STOP LOSS", -self.max_loss
            pnl = (self.entry_spy - current_spy) / self.entry_spy * self.max_profit * 5
            
        else:  # Iron Condor
            pnl = self.max_profit * 0.5
        
        return False, "HOLDING", pnl


class SmartAdvisor:
    """Optimized trading advisor with caching and in-memory tracking."""
    
    def __init__(
        self,
        model_path: str = "models/ppo_retrained.zip",
        min_confidence: float = MIN_CONFIDENCE,
        spread_width: int = 5
    ):
        self.min_confidence = min_confidence
        self.spread_width = spread_width
        
        # Load model
        if os.path.exists(model_path):
            self.model = PPO.load(model_path)
            print(f"✓ Loaded model: {model_path}")
        elif os.path.exists("models/ppo_spy_options.zip"):
            self.model = PPO.load("models/ppo_spy_options.zip")
            print("✓ Loaded fallback model")
        else:
            raise FileNotFoundError("No trained model found")
        
        # In-memory position tracking
        self.positions: List[Position] = []
        self.closed_positions: List[Dict] = []
        self.position_counter = 0
        
        # Data cache
        self.cached_df = None
        self.cached_spot = None
        self.cache_time = None
        self.cache_duration = 300  # 5 minutes
        
        # Stats
        self.total_pnl = 0.0
        self.wins = 0
        self.losses = 0
    
    def get_data(self, force_refresh: bool = False) -> tuple:
        """Get market data with caching."""
        now = datetime.now()
        
        # Use cache if fresh
        if not force_refresh and self.cache_time and self.cached_df is not None:
            age = (now - self.cache_time).total_seconds()
            if age < self.cache_duration:
                # Just update spot price
                try:
                    spy = yf.Ticker("SPY")
                    self.cached_spot = float(spy.history(period="1d")['Close'].iloc[-1])
                except:
                    pass
                return self.cached_df, self.cached_spot
        
        # Download fresh data
        print("  Downloading market data...")
        
        try:
            spy = yf.Ticker("SPY")
            vix = yf.Ticker("^VIX")
            
            # Get historical data
            spy_data = spy.history(period="1y")
            vix_data = vix.history(period="1y")
            
            # Create df
            df = spy_data.copy()
            
            # Add VIX
            df['VIX'] = vix_data['Close'].reindex(df.index, method='ffill')
            df['IV'] = df['VIX'].fillna(20) / 100
            
            # Calculate RSI
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0.0)
            loss = (-delta).where(delta < 0, 0.0)
            avg_gain = gain.rolling(14, min_periods=1).mean()
            avg_loss = loss.rolling(14, min_periods=1).mean()
            rs = avg_gain / (avg_loss + 1e-10)
            df['RSI'] = (100 - (100 / (1 + rs))) / 100
            
            # Calculate MACD
            ema12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema26 = df['Close'].ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            df['MACD'] = macd - signal
            
            # Volume surge
            df['Volume_Surge'] = df['Volume'] / df['Volume'].rolling(20).mean()
            
            # IV Rank
            df['IV_Rank'] = df['VIX'].rolling(252).apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 0.01) * 100 if len(x) > 10 else 50,
                raw=False
            )
            
            # PCR placeholder
            df['PCR'] = 1.0
            
            # Fill NaN
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0.5)
            
            # Get spot
            spot = float(df['Close'].iloc[-1])
            
            # Cache
            self.cached_df = df
            self.cached_spot = spot
            self.cache_time = now
            
            print(f"  ✓ Cached {len(df)} bars, SPY: ${spot:.2f}")
            
            return df, spot
            
        except Exception as e:
            print(f"  ⚠ Data error: {e}")
            if self.cached_df is not None:
                return self.cached_df, self.cached_spot
            raise
    
    def build_state(self, df) -> np.ndarray:
        """Build state vector."""
        row = df.iloc[-1]
        
        try:
            expected_dim = self.model.observation_space.shape[0]
        except:
            expected_dim = 15
        
        # Safely get values with defaults
        close = float(row.get('Close', row.get('close', 500)))
        rsi = float(row.get('RSI', 0.5))
        macd = float(row.get('MACD', 0))
        iv = float(row.get('IV', 0.2))
        iv_rank = float(row.get('IV_Rank', 50))
        pcr = float(row.get('PCR', 1))
        vol_surge = float(row.get('Volume_Surge', 1))
        
        if expected_dim == 15:
            state = np.array([
                np.clip((close - 400) / 400, 0, 1),
                rsi if 0 <= rsi <= 1 else rsi / 100,
                np.clip(macd / 10 + 0.5, 0, 1),
                np.clip((iv - 0.10) / 0.40, 0, 1),
                np.clip(iv_rank / 100, 0, 1),
                np.clip(pcr / 3, 0, 1),
                np.clip(vol_surge / 5, 0, 1),
                0.5, 0.5, 0.5,  # Placeholders
                0.5, 0.0, 0.0, 0.0, 0.0  # Portfolio state
            ], dtype=np.float32)
        else:
            state = np.array([
                np.clip((close - 400) / 400, 0, 1),
                rsi if 0 <= rsi <= 1 else rsi / 100,
                np.clip(macd / 10 + 0.5, 0, 1),
                np.clip((iv - 0.10) / 0.40, 0, 1),
                0.5, 0.5, 0.0, 0.0, 0.0, 0.0
            ], dtype=np.float32)
        
        return np.clip(state, 0, 1)
    
    def get_confidence(self, state: np.ndarray, action: int) -> float:
        """Get model confidence for action."""
        try:
            obs_tensor = self.model.policy.obs_to_tensor(state.reshape(1, -1))[0]
            dist = self.model.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.detach().numpy()[0]
            return float(probs[action]) * 100
        except:
            return 50.0
    
    def check_entry(self, action: int, confidence: float, df) -> tuple[bool, str]:
        """Check entry criteria."""
        if action >= 5:  # HOLD or CLOSE
            return False, "No trade action"
        
        if confidence < self.min_confidence:
            return False, f"Low confidence ({confidence:.1f}%)"
        
        row = df.iloc[-1]
        rsi = row['RSI'] * 100
        
        is_bullish = action in [0, 2]
        is_bearish = action in [1, 3]
        
        if is_bullish and rsi > RSI_OVERBOUGHT:
            return False, f"RSI overbought ({rsi:.0f})"
        
        if is_bearish and rsi < RSI_OVERSOLD:
            return False, f"RSI oversold ({rsi:.0f})"
        
        # Check for duplicate positions
        for pos in self.positions:
            if pos.action_name == ACTION_NAMES[action]:
                return False, f"Already have {ACTION_NAMES[action]}"
        
        return True, "✓ All criteria met"
    
    def open_position(self, action: int, spot: float) -> Position:
        """Open a new position."""
        self.position_counter += 1
        
        # Calculate levels based on action
        if action == 2:  # BULL SPREAD
            spy_target = spot + self.spread_width
            spy_stop = spot - self.spread_width * 0.4
            max_profit = self.spread_width * 100 * 0.6
            max_loss = self.spread_width * 100 * 0.4
        elif action == 3:  # BEAR SPREAD
            spy_target = spot - self.spread_width
            spy_stop = spot + self.spread_width * 0.4
            max_profit = self.spread_width * 100 * 0.6
            max_loss = self.spread_width * 100 * 0.4
        elif action == 0:  # BUY CALL
            spy_target = spot * 1.012
            spy_stop = spot * 0.992
            max_profit = 300
            max_loss = 200
        elif action == 1:  # BUY PUT
            spy_target = spot * 0.988
            spy_stop = spot * 1.008
            max_profit = 300
            max_loss = 200
        else:  # IRON CONDOR
            spy_target = spot
            spy_stop = spot
            max_profit = 150
            max_loss = 350
        
        pos = Position(
            id=self.position_counter,
            action_name=ACTION_NAMES[action],
            entry_time=datetime.now(),
            entry_spy=spot,
            spy_target=spy_target,
            spy_stop=spy_stop,
            max_profit=max_profit,
            max_loss=max_loss
        )
        
        self.positions.append(pos)
        return pos
    
    def close_position(self, pos: Position, reason: str, pnl: float, spot: float):
        """Close a position."""
        self.positions.remove(pos)
        self.closed_positions.append({
            'id': pos.id,
            'action': pos.action_name,
            'entry_spy': pos.entry_spy,
            'exit_spy': spot,
            'pnl': pnl,
            'reason': reason
        })
        
        self.total_pnl += pnl
        if pnl > 0:
            self.wins += 1
        else:
            self.losses += 1
    
    def print_entry(self, pos: Position, confidence: float, df):
        """Print entry alert."""
        row = df.iloc[-1]
        
        print("\n" + "=" * 60)
        print("🚨 ENTRY ALERT 🚨")
        print("=" * 60)
        print(f"  Time:       {datetime.now().strftime('%H:%M:%S')}")
        print(f"  Action:     {pos.action_name}")
        print(f"  Confidence: {confidence:.1f}%")
        print("")
        print(f"  SPY:        ${pos.entry_spy:.2f}")
        print(f"  [TARGET]    ${pos.spy_target:.2f}")
        print(f"  [STOP]      ${pos.spy_stop:.2f}")
        print("")
        print(f"  Max Profit: ${pos.max_profit:.2f}")
        print(f"  Max Loss:   ${pos.max_loss:.2f}")
        print("")
        print(f"  RSI: {row['RSI']*100:.1f} | IV: {row['IV']*100:.1f}%")
        print("=" * 60)
    
    def print_exit(self, pos: Position, reason: str, pnl: float, spot: float):
        """Print exit alert."""
        print("\n" + "=" * 60)
        print(f"{reason}")
        print("=" * 60)
        print(f"  Time:       {datetime.now().strftime('%H:%M:%S')}")
        print(f"  Action:     Close {pos.action_name}")
        print("")
        print(f"  Entry SPY:  ${pos.entry_spy:.2f}")
        print(f"  Exit SPY:   ${spot:.2f}")
        print(f"  P&L:        ${pnl:+.2f}")
        print("")
        print(f"  Total P&L:  ${self.total_pnl + pnl:.2f}")
        print(f"  W/L:        {self.wins + (1 if pnl > 0 else 0)}/{self.losses + (0 if pnl > 0 else 1)}")
        print("=" * 60)
    
    def print_position_status(self, spot: float):
        """Print current position status."""
        if not self.positions:
            return
        
        print("\n📊 OPEN POSITIONS:")
        for pos in self.positions:
            _, _, pnl = pos.check_exit(spot)
            holding_mins = (datetime.now() - pos.entry_time).total_seconds() / 60
            
            print(f"  [{pos.id}] {pos.action_name}")
            print(f"      Entry: ${pos.entry_spy:.2f} → Now: ${spot:.2f}")
            print(f"      Target: ${pos.spy_target:.2f} | Stop: ${pos.spy_stop:.2f}")
            print(f"      P&L: ${pnl:+.2f} | Holding: {holding_mins:.0f}m")
    
    def analyze(self) -> Optional[str]:
        """Run one analysis cycle."""
        # Get cached data
        df, spot = self.get_data()
        state = self.build_state(df)
        
        # Get model prediction
        action, _ = self.model.predict(state, deterministic=True)
        action = int(action)
        confidence = self.get_confidence(state, action)
        
        # Check exits first
        for pos in list(self.positions):
            should_exit, reason, pnl = pos.check_exit(spot)
            
            if should_exit:
                self.print_exit(pos, reason, pnl, spot)
                self.close_position(pos, reason, pnl, spot)
                return "EXIT"
        
        # Check entry
        can_enter, reason = self.check_entry(action, confidence, df)
        
        if can_enter:
            pos = self.open_position(action, spot)
            self.print_entry(pos, confidence, df)
            return "ENTRY"
        
        # Show position status if we have positions
        if self.positions:
            self.print_position_status(spot)
        
        print(f"  → {reason}")
        return None
    
    def run(self, interval: int = 60, duration_minutes: Optional[int] = None):
        """Run the advisor loop."""
        print("\n" + "=" * 60)
        print("🤖 SMART TRADING ADVISOR")
        print("=" * 60)
        print(f"  Confidence: ≥{self.min_confidence}%")
        print(f"  Interval:   {interval}s")
        print(f"  Positions:  Tracked in memory")
        print("=" * 60)
        print("\nWaiting for strategic opportunities...")
        
        start_time = datetime.now()
        count = 0
        
        try:
            while True:
                # Check duration
                if duration_minutes:
                    elapsed = (datetime.now() - start_time).total_seconds() / 60
                    if elapsed >= duration_minutes:
                        print(f"\n⏱ Duration limit reached ({duration_minutes} min)")
                        break
                
                count += 1
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Analyzing... (#{count})")
                
                try:
                    self.analyze()
                except Exception as e:
                    print(f"  ⚠ Error: {e}")
                
                # Force data refresh every 5 cycles
                if count % 5 == 0:
                    self.cache_time = None
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n\n👋 Advisor stopped")
        
        # Print summary
        print("\n" + "=" * 60)
        print("SESSION SUMMARY")
        print("=" * 60)
        print(f"  Open positions:  {len(self.positions)}")
        print(f"  Closed trades:   {len(self.closed_positions)}")
        print(f"  Total P&L:       ${self.total_pnl:.2f}")
        print(f"  Win/Loss:        {self.wins}/{self.losses}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Trading Advisor")
    parser.add_argument("--interval", type=int, default=60, help="Seconds between analysis")
    parser.add_argument("--duration", type=int, default=None, help="Max duration in minutes")
    parser.add_argument("--confidence", type=float, default=65, help="Min confidence %")
    
    args = parser.parse_args()
    
    advisor = SmartAdvisor(min_confidence=args.confidence)
    advisor.run(interval=args.interval, duration_minutes=args.duration)
