"""
PostgreSQL Database Module for RL Trading Bot
Stores signals, trades, and market data for model improvement.
Supports Neon DB (cloud PostgreSQL) via DATABASE_URL.
"""

import os
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, Json
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False
    print("Warning: psycopg2 not installed. Run: uv add psycopg2-binary")

load_dotenv()


def get_database_url() -> str:
    """Get database connection string from environment."""
    # Neon DB uses DATABASE_URL
    db_url = os.getenv('DATABASE_URL')
    
    if db_url:
        return db_url
    
    # Fallback to individual params (local PostgreSQL)
    host = os.getenv('POSTGRES_HOST', 'localhost')
    port = os.getenv('POSTGRES_PORT', '5432')
    db = os.getenv('POSTGRES_DB', 'rltrade')
    user = os.getenv('POSTGRES_USER', 'postgres')
    password = os.getenv('POSTGRES_PASSWORD', 'postgres')
    
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


class TradingDatabase:
    """
    PostgreSQL database for storing trading data.
    Supports both local PostgreSQL and Neon DB (cloud).
    
    Tables:
    - signals: AI-generated trading signals
    - trades: Executed trades with outcomes
    - market_data: Historical OHLCV + features
    - model_versions: Track model performance
    """
    
    def __init__(self, database_url: Optional[str] = None):
        if not HAS_PSYCOPG2:
            raise ImportError("psycopg2 required. Install with: uv add psycopg2-binary")
        
        self.database_url = database_url or get_database_url()
        self.conn = None
    
    def connect(self):
        """Connect to PostgreSQL (local or Neon DB)."""
        try:
            # Neon requires sslmode=require
            self.conn = psycopg2.connect(self.database_url)
            
            # Extract database name for logging
            db_name = self.database_url.split('/')[-1].split('?')[0]
            is_neon = 'neon.tech' in self.database_url
            
            print(f"Connected to {'Neon DB' if is_neon else 'PostgreSQL'}: {db_name}")
        except Exception as e:
            print(f"Database connection failed: {e}")
            raise
    
    def close(self):
        """Close connection."""
        if self.conn:
            self.conn.close()
    
    def create_tables(self):
        """Create all required tables."""
        with self.conn.cursor() as cur:
            # Signals table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    model_version VARCHAR(100),
                    
                    -- Market state
                    spot_price DECIMAL(10, 2),
                    iv DECIMAL(6, 4),
                    iv_rank DECIMAL(5, 2),
                    rsi DECIMAL(5, 2),
                    macd DECIMAL(10, 4),
                    put_call_ratio DECIMAL(5, 2),
                    volume_surge DECIMAL(5, 2),
                    
                    -- Signal
                    action INTEGER,
                    action_name VARCHAR(50),
                    confidence DECIMAL(5, 2),
                    probabilities JSONB,
                    
                    -- Trade setup
                    trade_type VARCHAR(50),
                    strike DECIMAL(10, 2),
                    expiry DATE,
                    entry_price DECIMAL(10, 2),
                    stop_loss DECIMAL(10, 2),
                    target DECIMAL(10, 2),
                    
                    -- State vector (for retraining)
                    state_vector JSONB,
                    
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_signals_timestamp 
                ON signals(timestamp);
                
                CREATE INDEX IF NOT EXISTS idx_signals_action 
                ON signals(action);
            """)
            
            # Trades table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id SERIAL PRIMARY KEY,
                    signal_id INTEGER REFERENCES signals(id),
                    
                    -- Entry
                    entry_timestamp TIMESTAMPTZ,
                    entry_spot DECIMAL(10, 2),
                    entry_price DECIMAL(10, 2),
                    contracts INTEGER,
                    
                    -- Exit
                    exit_timestamp TIMESTAMPTZ,
                    exit_spot DECIMAL(10, 2),
                    exit_price DECIMAL(10, 2),
                    exit_reason VARCHAR(50),  -- target, stop_loss, expiry, manual
                    
                    -- Results
                    pnl DECIMAL(12, 2),
                    pnl_pct DECIMAL(8, 4),
                    holding_minutes INTEGER,
                    
                    -- For RL: was the signal correct?
                    was_profitable BOOLEAN,
                    
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_trades_entry 
                ON trades(entry_timestamp);
                
                CREATE INDEX IF NOT EXISTS idx_trades_profitable 
                ON trades(was_profitable);
            """)
            
            # Market data table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    symbol VARCHAR(10) DEFAULT 'SPY',
                    
                    -- OHLCV
                    open DECIMAL(10, 2),
                    high DECIMAL(10, 2),
                    low DECIMAL(10, 2),
                    close DECIMAL(10, 2),
                    volume BIGINT,
                    
                    -- VIX/IV
                    vix DECIMAL(6, 2),
                    iv DECIMAL(6, 4),
                    
                    -- Features
                    rsi DECIMAL(5, 2),
                    macd DECIMAL(10, 4),
                    ema20 DECIMAL(10, 2),
                    volatility DECIMAL(10, 6),
                    iv_rank DECIMAL(5, 2),
                    pcr DECIMAL(5, 2),
                    volume_surge DECIMAL(5, 2),
                    
                    UNIQUE(timestamp, symbol)
                );
                
                CREATE INDEX IF NOT EXISTS idx_market_timestamp 
                ON market_data(timestamp);
            """)
            
            # Model versions table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS model_versions (
                    id SERIAL PRIMARY KEY,
                    version VARCHAR(100) UNIQUE,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    
                    -- Training info
                    timesteps INTEGER,
                    training_days INTEGER,
                    train_start DATE,
                    train_end DATE,
                    
                    -- Performance
                    backtest_return DECIMAL(8, 4),
                    backtest_sharpe DECIMAL(6, 3),
                    win_rate DECIMAL(5, 2),
                    num_trades INTEGER,
                    
                    -- File path
                    model_path TEXT,
                    
                    is_active BOOLEAN DEFAULT FALSE
                );
            """)
            
            self.conn.commit()
            print("✓ Database tables created")
    
    def store_signal(
        self,
        signal: Dict[str, Any],
        state_vector: Optional[np.ndarray] = None,
        model_version: str = "v1"
    ) -> int:
        """
        Store a trading signal.
        
        Returns:
            Signal ID
        """
        def to_python(val):
            """Convert numpy types to Python native types."""
            if val is None:
                return None
            if isinstance(val, (np.integer, np.floating)):
                return float(val)
            if isinstance(val, np.ndarray):
                return val.tolist()
            return val
        
        with self.conn.cursor() as cur:
            trade_setup = signal.get('trade_setup', {}) or {}
            
            # Convert all numeric values
            spot = to_python(signal.get('spot'))
            iv = to_python(signal.get('iv', 0))
            if iv and iv > 1:
                iv = iv / 100
            
            cur.execute("""
                INSERT INTO signals (
                    timestamp, model_version,
                    spot_price, iv, iv_rank, rsi, macd, put_call_ratio, volume_surge,
                    action, action_name, confidence, probabilities,
                    trade_type, strike, expiry, entry_price, stop_loss, target,
                    state_vector
                ) VALUES (
                    %s, %s,
                    %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s,
                    %s
                ) RETURNING id
            """, (
                signal.get('timestamp', datetime.now().isoformat()),
                model_version,
                spot,
                iv,
                to_python(signal.get('iv_rank')),
                to_python(signal.get('rsi')),
                to_python(signal.get('macd')),
                to_python(signal.get('put_call_ratio')),
                to_python(signal.get('volume_surge', 1)),
                to_python(signal.get('action')),
                signal.get('action_name'),
                to_python(signal.get('confidence')),
                Json({k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                      for k, v in signal.get('probabilities', {}).items()}),
                trade_setup.get('type'),
                to_python(trade_setup.get('strike')),
                trade_setup.get('expiry'),
                to_python(trade_setup.get('entry')),
                to_python(trade_setup.get('stop_loss')),
                to_python(trade_setup.get('target')),
                Json(state_vector.tolist() if state_vector is not None else [])
            ))
            
            signal_id = cur.fetchone()[0]
            self.conn.commit()
            
            return signal_id
    
    def store_trade(
        self,
        signal_id: int,
        entry_spot: float,
        entry_price: float,
        contracts: int,
        exit_spot: Optional[float] = None,
        exit_price: Optional[float] = None,
        exit_reason: Optional[str] = None,
        pnl: Optional[float] = None
    ) -> int:
        """Store a trade execution."""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO trades (
                    signal_id, entry_timestamp, entry_spot, entry_price, contracts,
                    exit_timestamp, exit_spot, exit_price, exit_reason, pnl, pnl_pct,
                    was_profitable
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s,
                    %s
                ) RETURNING id
            """, (
                signal_id, datetime.now(), entry_spot, entry_price, contracts,
                datetime.now() if exit_spot else None, exit_spot, exit_price, exit_reason, pnl,
                pnl / (entry_price * 100 * contracts) if pnl and entry_price else None,
                pnl > 0 if pnl else None
            ))
            
            trade_id = cur.fetchone()[0]
            self.conn.commit()
            
            return trade_id
    
    def update_trade_exit(
        self,
        trade_id: int,
        exit_spot: float,
        exit_price: float,
        exit_reason: str,
        pnl: float,
        holding_minutes: int
    ):
        """Update trade with exit info."""
        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE trades SET
                    exit_timestamp = %s,
                    exit_spot = %s,
                    exit_price = %s,
                    exit_reason = %s,
                    pnl = %s,
                    pnl_pct = pnl / (entry_price * 100 * contracts),
                    holding_minutes = %s,
                    was_profitable = %s
                WHERE id = %s
            """, (
                datetime.now(), exit_spot, exit_price, exit_reason, pnl,
                holding_minutes, pnl > 0, trade_id
            ))
            self.conn.commit()
    
    def store_market_data(self, df: pd.DataFrame, symbol: str = "SPY"):
        """Store market data with features."""
        with self.conn.cursor() as cur:
            for idx, row in df.iterrows():
                try:
                    cur.execute("""
                        INSERT INTO market_data (
                            timestamp, symbol, open, high, low, close, volume,
                            vix, iv, rsi, macd, ema20, volatility, iv_rank, pcr, volume_surge
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s, %s
                        ) ON CONFLICT (timestamp, symbol) DO UPDATE SET
                            close = EXCLUDED.close,
                            rsi = EXCLUDED.rsi,
                            macd = EXCLUDED.macd
                    """, (
                        idx if isinstance(idx, datetime) else datetime.now(),
                        symbol,
                        row.get('Open'), row.get('High'), row.get('Low'), row.get('Close'),
                        row.get('Volume'),
                        row.get('VIX'), row.get('IV'),
                        row.get('RSI'), row.get('MACD'), row.get('EMA20'),
                        row.get('Volatility'), row.get('IV_Rank'),
                        row.get('PCR'), row.get('Volume_Surge')
                    ))
                except Exception as e:
                    print(f"Error storing row {idx}: {e}")
            
            self.conn.commit()
    
    def get_training_data(
        self,
        days: int = 365,
        include_signals: bool = True
    ) -> pd.DataFrame:
        """
        Get market data with signal outcomes for retraining.
        
        This is the key function for RL improvement:
        - Gets historical market data
        - Joins with actual trade outcomes
        - Provides reward signals based on real results
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            if include_signals:
                cur.execute("""
                    SELECT 
                        m.*,
                        s.action as signal_action,
                        s.confidence as signal_confidence,
                        t.pnl as trade_pnl,
                        t.was_profitable
                    FROM market_data m
                    LEFT JOIN signals s ON DATE(s.timestamp) = DATE(m.timestamp)
                    LEFT JOIN trades t ON t.signal_id = s.id
                    WHERE m.timestamp > NOW() - INTERVAL '%s days'
                    ORDER BY m.timestamp
                """, (days,))
            else:
                cur.execute("""
                    SELECT * FROM market_data
                    WHERE timestamp > NOW() - INTERVAL '%s days'
                    ORDER BY timestamp
                """, (days,))
            
            rows = cur.fetchall()
            
        return pd.DataFrame(rows)
    
    def get_signal_performance(self, days: int = 30) -> Dict[str, Any]:
        """Get signal performance metrics for model evaluation."""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    COUNT(*) as total_signals,
                    COUNT(CASE WHEN t.id IS NOT NULL THEN 1 END) as executed_trades,
                    COUNT(CASE WHEN t.was_profitable THEN 1 END) as profitable_trades,
                    AVG(t.pnl) as avg_pnl,
                    SUM(t.pnl) as total_pnl,
                    AVG(s.confidence) as avg_confidence,
                    
                    -- By action type
                    COUNT(CASE WHEN s.action = 0 THEN 1 END) as call_signals,
                    COUNT(CASE WHEN s.action = 1 THEN 1 END) as put_signals,
                    COUNT(CASE WHEN s.action >= 5 THEN 1 END) as hold_signals
                FROM signals s
                LEFT JOIN trades t ON t.signal_id = s.id
                WHERE s.timestamp > NOW() - INTERVAL '%s days'
            """, (days,))
            
            return dict(cur.fetchone())
    
    def get_recent_signals(self, limit: int = 50) -> List[Dict]:
        """Get recent signals for review."""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT 
                    s.*,
                    t.pnl,
                    t.was_profitable,
                    t.exit_reason
                FROM signals s
                LEFT JOIN trades t ON t.signal_id = s.id
                ORDER BY s.timestamp DESC
                LIMIT %s
            """, (limit,))
            
            return [dict(row) for row in cur.fetchall()]


def init_database():
    """Initialize the database (run once)."""
    db = TradingDatabase()
    db.connect()
    db.create_tables()
    db.close()
    print("Database initialized!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true", help="Initialize database")
    parser.add_argument("--stats", action="store_true", help="Show performance stats")
    parser.add_argument("--view", action="store_true", help="View all table contents")
    parser.add_argument("--signals", action="store_true", help="View signals only")
    parser.add_argument("--trades", action="store_true", help="View trades only")
    
    args = parser.parse_args()
    
    if args.init:
        init_database()
    elif args.stats:
        db = TradingDatabase()
        db.connect()
        stats = db.get_signal_performance(30)
        print("\nSignal Performance (30 days):")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        db.close()
    elif args.view or args.signals:
        db = TradingDatabase()
        db.connect()
        
        print("\n" + "=" * 80)
        print("SIGNALS TABLE")
        print("=" * 80)
        
        with db.conn.cursor() as cur:
            cur.execute("""
                SELECT id, timestamp, action_name, confidence, spot_price, 
                       trade_type, strike, entry_price
                FROM signals ORDER BY timestamp DESC LIMIT 20
            """)
            rows = cur.fetchall()
            
            if rows:
                print(f"{'ID':>4} | {'Time':20} | {'Action':15} | {'Conf':>6} | {'Spot':>8} | {'Type':15} | {'Strike':>7} | {'Entry':>7}")
                print("-" * 100)
                for row in rows:
                    print(f"{row[0]:>4} | {str(row[1])[:20]:20} | {str(row[2])[:15]:15} | {row[3] or 0:>5.1f}% | ${row[4] or 0:>7.2f} | {str(row[5] or '-')[:15]:15} | ${row[6] or 0:>6.0f} | ${row[7] or 0:>6.2f}")
            else:
                print("No signals found.")
        
        if args.view or args.trades:
            print("\n" + "=" * 80)
            print("TRADES TABLE")
            print("=" * 80)
            
            with db.conn.cursor() as cur:
                cur.execute("""
                    SELECT id, signal_id, entry_timestamp, entry_spot, pnl, 
                           was_profitable, exit_reason
                    FROM trades ORDER BY entry_timestamp DESC LIMIT 20
                """)
                rows = cur.fetchall()
                
                if rows:
                    print(f"{'ID':>4} | {'Signal':>6} | {'Time':20} | {'Entry':>8} | {'P&L':>10} | {'Profit':>7} | {'Reason':10}")
                    print("-" * 80)
                    for row in rows:
                        print(f"{row[0]:>4} | {row[1]:>6} | {str(row[2])[:20]:20} | ${row[3] or 0:>7.2f} | ${row[4] or 0:>9.2f} | {'YES' if row[5] else 'NO':>7} | {str(row[6] or '-')[:10]:10}")
                else:
                    print("No trades found.")
        
        db.close()
    elif args.trades:
        db = TradingDatabase()
        db.connect()
        
        print("\n" + "=" * 80)
        print("TRADES TABLE")
        print("=" * 80)
        
        with db.conn.cursor() as cur:
            cur.execute("""
                SELECT t.id, t.signal_id, t.entry_timestamp, t.entry_spot, t.pnl, 
                       t.was_profitable, t.exit_reason, s.action_name
                FROM trades t
                LEFT JOIN signals s ON s.id = t.signal_id
                ORDER BY t.entry_timestamp DESC LIMIT 20
            """)
            rows = cur.fetchall()
            
            if rows:
                print(f"{'ID':>4} | {'Action':15} | {'Enter':20} | {'Spot':>8} | {'P&L':>10} | {'Profit':>7} | {'Reason':10}")
                print("-" * 90)
                for row in rows:
                    print(f"{row[0]:>4} | {str(row[7] or '-')[:15]:15} | {str(row[2])[:20]:20} | ${row[3] or 0:>7.2f} | ${row[4] or 0:>9.2f} | {'YES' if row[5] else 'NO':>7} | {str(row[6] or '-')[:10]:10}")
            else:
                print("No trades found.")
        
        db.close()
    else:
        print("Commands:")
        print("  --init     Create database tables")
        print("  --stats    Show performance stats")
        print("  --view     View all signals and trades")
        print("  --signals  View signals only")
        print("  --trades   View trades only")
