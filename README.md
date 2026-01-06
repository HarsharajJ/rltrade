# SPY Options Trading Bot - Reinforcement Learning System

An AI-powered options trading signal generator using **Proximal Policy Optimization (PPO)** reinforcement learning, with cloud database storage for continuous model improvement.

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Reinforcement Learning Explained](#reinforcement-learning-explained)
3. [Project Architecture](#project-architecture)
4. [Setup & Installation](#setup--installation)
5. [Running the System](#running-the-system)
6. [The Learning Loop](#the-learning-loop)
7. [File Reference](#file-reference)
8. [Technical Details](#technical-details)

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRADING SIGNAL PIPELINE                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   [Market Data]  →  [Feature Engineering]  →  [PPO Model]  →  [Signal]
│        ↓                    ↓                      ↓              ↓
│   SPY + VIX          15 Features            7 Actions        Trade Setup
│   prices            IV Rank, RSI,          Call, Put,       Entry, Stop,
│                     MACD, etc.             Spreads          Target
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### The Process:

1. **Fetch Data**: Get SPY price, VIX (volatility), and real options data
2. **Build Features**: Calculate 15 technical indicators
3. **AI Decision**: PPO model outputs action probabilities
4. **Generate Signal**: Create trade setup with entry/exit levels
5. **Store to DB**: Save signal for outcome tracking
6. **Track Outcome**: Later check if signal was profitable
7. **Retrain**: Use outcomes to improve the model

---

## Reinforcement Learning Explained

### What is PPO?

**Proximal Policy Optimization (PPO)** is a reinforcement learning algorithm that learns by:

1. **Observing** the market state (15-dimensional feature vector)
2. **Taking actions** (buy call, buy put, spread, hold, close)
3. **Receiving rewards** (profit = positive, loss = negative)
4. **Updating policy** to maximize future rewards

### The RL Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                    REINFORCEMENT LEARNING SETUP                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   AGENT (PPO Model)                                                   │
│   └── Neural Network with 2 hidden layers (64 neurons each)          │
│                                                                       │
│   ENVIRONMENT (ProductionOptionsEnv)                                  │
│   └── Simulates options trading with:                                 │
│       • Real market data (SPY + VIX)                                  │
│       • Transaction costs (commission, slippage, spread)              │
│       • Position management                                           │
│                                                                       │
│   STATE (15 dimensions)                                               │
│   └── [Price, RSI, MACD, IV, IV_Rank, PCR, Volume, Gamma,            │
│        Momentum, Balance, HasPosition, PositionType, PnL, Time]       │
│                                                                       │
│   ACTIONS (7 choices)                                                 │
│   └── 0: BUY_CALL      (bullish, unlimited profit)                    │
│       1: BUY_PUT       (bearish, unlimited profit)                    │
│       2: BULL_SPREAD   (bullish, defined risk)                        │
│       3: BEAR_SPREAD   (bearish, defined risk)                        │
│       4: IRON_CONDOR   (neutral, defined risk)                        │
│       5: HOLD          (no action)                                    │
│       6: CLOSE         (exit position)                                │
│                                                                       │
│   REWARD                                                              │
│   └── Realized P&L - Transaction Costs - Time Decay Penalty          │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### How Learning Works

```
Episode 1: Agent explores randomly, makes mistakes, learns basic patterns
Episode 100: Agent learns to avoid holding too long (time decay hurts)
Episode 1000: Agent learns bullish signals in uptrends
Episode 10000: Agent develops nuanced strategy based on IV and momentum
```

---

## Project Architecture

```
rltrade/
│
├── live_signals_production.py   # 🎯 Main entry point - generates signals
├── train_production.py          # 🏋️ Train the PPO model
├── track_outcomes.py            # 📊 Track if signals were profitable
├── retrain_from_db.py           # 🔄 Retrain using trade history
│
├── src/
│   ├── database.py              # ☁️ Neon DB - stores signals/trades
│   ├── env_production.py        # 🎮 RL Environment (7 actions)
│   ├── features.py              # 📈 Feature engineering (15 dims)
│   ├── options_data.py          # 📊 Real options chain data
│   ├── options_pricing.py       # 🧮 Black-Scholes + Greeks
│   ├── data_loader.py           # 📥 SPY + VIX data fetching
│   └── walk_forward.py          # ✅ Rolling validation
│
├── models/                      # 💾 Saved trained models
│   ├── ppo_spy_options.zip      # Basic 10-dim model
│   └── ppo_retrained.zip        # Production 15-dim model
│
└── .env                         # 🔐 API keys and DB credentials
```

---

## Setup & Installation

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Neon DB account (free): https://neon.tech

### Step 1: Install Dependencies

```bash
cd rltrade
uv sync
```

### Step 2: Configure Environment

Create `.env` file:

```bash
# Neon DB (required) - get from https://console.neon.tech
DATABASE_URL=postgresql://user:password@ep-xxx.neon.tech/neondb?sslmode=require

# Alpaca (optional - for future live trading)
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
```

### Step 3: Initialize Database

```bash
uv run python -m src.database --init
```

This creates 4 tables:
- `signals` - AI-generated trading signals
- `trades` - Executed trades with outcomes
- `market_data` - Historical OHLCV data
- `model_versions` - Model performance tracking

---

## Running the System

### 1. Generate Trading Signals

```bash
# Single signal
uv run python live_signals_production.py --once

# Continuous (every 5 minutes)
uv run python live_signals_production.py

# Custom interval (every 1 minute)
uv run python live_signals_production.py --interval 60
```

**Output:**
```
>>> SPY OPTIONS SIGNAL <<<
Time: 2026-01-06T17:27:15

MARKET DATA
   SPY Price:     $687.72
   IV:            20.0%
   RSI:           58.7

AI RECOMMENDATION
   Signal:        [v] BEAR SPREAD
   Confidence:    70.4%

>>> OPTION CONTRACT <<<
   SYMBOL:     SPY 260113 P 688/683
   Type:       BEAR PUT SPREAD

>>> DEFINED RISK <<<
   [MAX PROFIT] $300.00
   [MAX LOSS]   $200.00

Signal stored in DB (id=2)
```

### 2. View Database Contents

```bash
# View all signals and trades
uv run python -m src.database --view

# View statistics
uv run python -m src.database --stats

# View only signals
uv run python -m src.database --signals
```

### 3. Track Trade Outcomes

After time passes, check if signals were profitable:

```bash
# Check signals from last 24 hours
uv run python track_outcomes.py --hours 24

# Run continuously (check every hour)
uv run python track_outcomes.py --continuous --interval 60
```

**How it determines profit/loss:**

| Signal Type | SPY Moves | Result |
|-------------|-----------|--------|
| BULL SPREAD | UP +1%+ | ✅ Profitable |
| BULL SPREAD | DOWN -0.8%+ | ❌ Loss |
| BEAR SPREAD | DOWN -1%+ | ✅ Profitable |
| BEAR SPREAD | UP +0.8%+ | ❌ Loss |

### 4. Train the Model

```bash
# Quick training (test)
uv run python train_production.py --timesteps 50000 --seeds 1

# Full production training (2-3 hours)
uv run python train_production.py --timesteps 500000 --seeds 3

# Retrain using database feedback
uv run python retrain_from_db.py --timesteps 100000
```

### 5. Walk-Forward Validation

Test model on unseen data with rolling windows:

```bash
uv run python -m src.walk_forward --years 3 --timesteps 100000
```

---

## The Learning Loop

This is the key to continuous improvement:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     CONTINUOUS IMPROVEMENT LOOP                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   1. GENERATE SIGNALS                                                 │
│      └── live_signals_production.py --once                            │
│      └── Signal stored in Neon DB with state vector                   │
│                              ↓                                        │
│   2. WAIT FOR MARKET MOVEMENT                                         │
│      └── SPY price changes over hours/days                            │
│                              ↓                                        │
│   3. TRACK OUTCOMES                                                   │
│      └── track_outcomes.py --hours 24                                 │
│      └── Compares prediction vs reality                               │
│      └── Records was_profitable = True/False                          │
│                              ↓                                        │
│   4. RETRAIN MODEL                                                    │
│      └── retrain_from_db.py --timesteps 100000                        │
│      └── Uses profitable signals as positive reward                   │
│      └── Uses losing signals as negative reward                       │
│                              ↓                                        │
│   5. IMPROVED MODEL                                                   │
│      └── New model saved as ppo_retrained.zip                         │
│      └── Better predictions over time                                 │
│                              ↓                                        │
│      ────────────── REPEAT ──────────────                             │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Automated Daily Workflow

```bash
# Morning: Generate signal
uv run python live_signals_production.py --once

# Evening: Track today's outcome
uv run python track_outcomes.py --hours 12

# Weekly: Retrain with new data
uv run python retrain_from_db.py --timesteps 100000

# View performance
uv run python -m src.database --stats
```

---

## File Reference

### Main Scripts

| File | Purpose | Command |
|------|---------|---------|
| `live_signals_production.py` | Generate trading signals | `--once`, `--interval 60` |
| `train_production.py` | Train PPO model | `--timesteps 500000 --seeds 3` |
| `track_outcomes.py` | Track signal outcomes | `--hours 24`, `--continuous` |
| `retrain_from_db.py` | Retrain from history | `--timesteps 100000` |

### Source Modules

| File | Purpose |
|------|---------|
| `src/database.py` | Neon DB connection, CRUD operations |
| `src/env_production.py` | RL environment with 7 actions, 15-dim state |
| `src/features.py` | Technical indicators, feature normalization |
| `src/options_data.py` | Fetch real options chains from yfinance |
| `src/options_pricing.py` | Black-Scholes, Greeks calculation |
| `src/data_loader.py` | Download SPY + VIX historical data |
| `src/walk_forward.py` | Rolling window validation |

---

## Technical Details

### State Vector (15 Dimensions)

| Index | Feature | Range | Source |
|-------|---------|-------|--------|
| 0 | SPY Price (normalized) | 0-1 | yfinance |
| 1 | RSI (14-period) | 0-1 | Calculated |
| 2 | MACD Histogram | 0-1 | Calculated |
| 3 | Implied Volatility | 0-1 | VIX proxy |
| 4 | IV Rank (52-week) | 0-1 | Calculated |
| 5 | Put/Call Ratio | 0-1 | Volume-based |
| 6 | Volume Surge | 0-1 | vs 20-day avg |
| 7 | IV Skew | 0-1 | Put-Call IV diff |
| 8 | Gamma Proxy | 0-1 | Strike distance |
| 9 | Momentum (5-day) | 0-1 | Price change |
| 10 | Portfolio Balance | 0-1 | Normalized |
| 11 | Has Position | 0/1 | Binary |
| 12 | Position Type | 0-1 | Encoded |
| 13 | Unrealized P&L | 0-1 | Normalized |
| 14 | Holding Time | 0-1 | Days/max_days |

### Transaction Costs

Built into the environment for realistic training:

```python
COMMISSION = $0.65 per contract per leg
SLIPPAGE = 2% of option premium
BID_ASK_SPREAD = $0.05 per contract
```

### Reward Function

```python
reward = realized_pnl 
       - transaction_costs 
       - time_decay_penalty 
       - invalid_action_penalty
```

---

## Disclaimer

> ⚠️ **RISK WARNING**: Options trading involves substantial risk of loss. This system is for **educational purposes only**. Past performance does not guarantee future results. Always paper trade extensively before using real money.

---

## License

MIT License - Use at your own risk.
#   r l t r a d e  
 