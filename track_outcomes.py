"""
Trade Outcome Tracker
Checks past signals and records whether they were profitable.
This feedback loop is critical for RL model improvement.
"""

import argparse
from datetime import datetime, timedelta
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import yfinance as yf

from src.database import TradingDatabase


def get_spy_price_at_time(timestamp: datetime) -> Optional[float]:
    """Get SPY price at a specific time."""
    try:
        spy = yf.Ticker("SPY")
        # Get data around the timestamp
        start = timestamp - timedelta(days=1)
        end = timestamp + timedelta(days=1)
        hist = spy.history(start=start, end=end, interval="1h")
        
        if hist.empty:
            return None
        
        # Find closest price
        hist.index = hist.index.tz_localize(None)
        ts_naive = timestamp.replace(tzinfo=None) if timestamp.tzinfo else timestamp
        
        closest = hist.index.get_indexer([ts_naive], method='nearest')[0]
        return float(hist.iloc[closest]['Close'])
    except Exception as e:
        print(f"Error fetching price: {e}")
        return None


def get_current_spy_price() -> float:
    """Get current SPY price."""
    spy = yf.Ticker("SPY")
    return float(spy.history(period="1d")['Close'].iloc[-1])


def evaluate_signal_outcome(
    signal: Dict,
    current_price: float,
    entry_price: float
) -> Dict:
    """
    Evaluate if a signal's trade recommendation was profitable.
    
    For BUY signals:
    - Profitable if SPY moved in the predicted direction
    - Loss if SPY moved against prediction
    
    Returns outcome details.
    """
    action = signal.get('action', 5)
    action_name = signal.get('action_name', 'HOLD')
    spot_at_signal = float(signal.get('spot_price', current_price))
    
    # Calculate SPY movement since signal
    spy_change = current_price - spot_at_signal
    spy_change_pct = (spy_change / spot_at_signal) * 100 if spot_at_signal else 0
    
    outcome = {
        'signal_id': signal.get('id'),
        'action': action_name,
        'spot_at_signal': spot_at_signal,
        'current_price': current_price,
        'spy_change': spy_change,
        'spy_change_pct': spy_change_pct,
        'was_profitable': False,
        'exit_reason': 'pending',
        'estimated_pnl': 0.0
    }
    
    # Skip HOLD signals
    if action >= 5:
        outcome['exit_reason'] = 'no_trade'
        return outcome
    
    # Determine if profitable based on direction
    # 0 = BUY_CALL (bullish), 1 = BUY_PUT (bearish)
    # 2 = BULL_SPREAD (bullish), 3 = BEAR_SPREAD (bearish)
    # 4 = IRON_CONDOR (neutral)
    
    is_bullish = action in [0, 2]  # CALL or BULL_SPREAD
    is_bearish = action in [1, 3]  # PUT or BEAR_SPREAD
    is_neutral = action == 4       # IRON_CONDOR
    
    stop_loss_pct = 0.30
    target_pct = 0.50
    
    if is_bullish:
        # Bullish trade: profitable if SPY went up
        if spy_change_pct >= 1.0:  # SPY up 1%+ = target likely hit
            outcome['was_profitable'] = True
            outcome['exit_reason'] = 'target'
            outcome['estimated_pnl'] = target_pct * entry_price * 100
        elif spy_change_pct <= -0.8:  # SPY down 0.8% = stop likely hit
            outcome['was_profitable'] = False
            outcome['exit_reason'] = 'stop_loss'
            outcome['estimated_pnl'] = -stop_loss_pct * entry_price * 100
        else:
            outcome['exit_reason'] = 'open'
            outcome['estimated_pnl'] = spy_change_pct * 5 * entry_price  # Rough delta estimate
            outcome['was_profitable'] = outcome['estimated_pnl'] > 0
            
    elif is_bearish:
        # Bearish trade: profitable if SPY went down
        if spy_change_pct <= -1.0:  # SPY down 1%+ = target likely hit
            outcome['was_profitable'] = True
            outcome['exit_reason'] = 'target'
            outcome['estimated_pnl'] = target_pct * entry_price * 100
        elif spy_change_pct >= 0.8:  # SPY up 0.8% = stop likely hit
            outcome['was_profitable'] = False
            outcome['exit_reason'] = 'stop_loss'
            outcome['estimated_pnl'] = -stop_loss_pct * entry_price * 100
        else:
            outcome['exit_reason'] = 'open'
            outcome['estimated_pnl'] = -spy_change_pct * 5 * entry_price
            outcome['was_profitable'] = outcome['estimated_pnl'] > 0
            
    elif is_neutral:
        # Iron Condor: profitable if SPY stayed flat
        if abs(spy_change_pct) <= 1.5:  # SPY within 1.5% = profit zone
            outcome['was_profitable'] = True
            outcome['exit_reason'] = 'target'
            outcome['estimated_pnl'] = 150  # Max credit estimate
        else:
            outcome['was_profitable'] = False
            outcome['exit_reason'] = 'breached'
            outcome['estimated_pnl'] = -350  # Max loss estimate
    
    return outcome


def update_signal_outcomes(
    hours_back: int = 24,
    dry_run: bool = False
):
    """
    Check signals from the last N hours and record outcomes.
    
    This is the key feedback loop for RL improvement.
    """
    print("=" * 60)
    print("TRADE OUTCOME TRACKER")
    print("=" * 60)
    
    db = TradingDatabase()
    db.connect()
    
    # Get recent signals without trade outcomes
    with db.conn.cursor() as cur:
        cur.execute("""
            SELECT s.* 
            FROM signals s
            LEFT JOIN trades t ON t.signal_id = s.id
            WHERE s.timestamp > NOW() - INTERVAL '%s hours'
            AND t.id IS NULL
            ORDER BY s.timestamp DESC
        """, (hours_back,))
        
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
    
    signals = [dict(zip(columns, row)) for row in rows]
    
    print(f"\nFound {len(signals)} signals to evaluate")
    
    if not signals:
        print("No pending signals to evaluate.")
        db.close()
        return
    
    # Get current SPY price
    current_price = get_current_spy_price()
    print(f"Current SPY: ${current_price:.2f}")
    
    # Evaluate each signal
    outcomes = []
    
    for signal in signals:
        print(f"\n--- Signal #{signal['id']} ---")
        print(f"  Time: {signal['timestamp']}")
        print(f"  Action: {signal['action_name']}")
        print(f"  SPY at signal: ${signal.get('spot_price', 0):.2f}")
        
        # Default entry price estimate
        entry_price = signal.get('entry_price') or 5.0
        
        outcome = evaluate_signal_outcome(signal, current_price, entry_price)
        outcomes.append(outcome)
        
        print(f"  SPY change: {outcome['spy_change_pct']:+.2f}%")
        print(f"  Outcome: {'PROFITABLE' if outcome['was_profitable'] else 'LOSS'}")
        print(f"  Exit reason: {outcome['exit_reason']}")
        print(f"  Est. P&L: ${outcome['estimated_pnl']:.2f}")
        
        # Store trade record
        if not dry_run and outcome['exit_reason'] != 'no_trade':
            try:
                trade_id = db.store_trade(
                    signal_id=signal['id'],
                    entry_spot=outcome['spot_at_signal'],
                    entry_price=entry_price,
                    contracts=1,
                    exit_spot=current_price,
                    exit_price=entry_price * (1 + (0.5 if outcome['was_profitable'] else -0.3)),
                    exit_reason=outcome['exit_reason'],
                    pnl=outcome['estimated_pnl']
                )
                print(f"  Trade recorded (id={trade_id})")
            except Exception as e:
                print(f"  Error storing trade: {e}")
    
    # Summary
    if outcomes:
        profitable = sum(1 for o in outcomes if o['was_profitable'])
        total_pnl = sum(o['estimated_pnl'] for o in outcomes)
        
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"  Signals evaluated: {len(outcomes)}")
        print(f"  Profitable: {profitable}/{len(outcomes)} ({profitable/len(outcomes)*100:.1f}%)")
        print(f"  Total Est. P&L: ${total_pnl:.2f}")
    
    db.close()
    print("\n✓ Outcome tracking complete!")


def run_continuous_tracker(interval_minutes: int = 60):
    """Run continuous outcome tracking."""
    import time
    
    print(f"Starting continuous tracker (every {interval_minutes} min)")
    
    while True:
        try:
            update_signal_outcomes(hours_back=24)
        except Exception as e:
            print(f"Error: {e}")
        
        print(f"\nNext check in {interval_minutes} minutes...")
        time.sleep(interval_minutes * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track trade outcomes")
    parser.add_argument("--hours", type=int, default=24, help="Hours back to check")
    parser.add_argument("--dry-run", action="store_true", help="Don't save to DB")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=60, help="Check interval (minutes)")
    
    args = parser.parse_args()
    
    if args.continuous:
        run_continuous_tracker(args.interval)
    else:
        update_signal_outcomes(hours_back=args.hours, dry_run=args.dry_run)
