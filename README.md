# 🤖 SPY Options RL Trading Bot

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An AI-powered options trading signal generator using **Proximal Policy Optimization (PPO)** reinforcement learning with cloud database storage for continuous model improvement.

## ✨ Features

- 🎯 **7 Trading Actions**: Calls, Puts, Bull/Bear Spreads, Iron Condors
- 📊 **15-Dimensional State**: IV Rank, Put/Call Ratio, Gamma, Momentum
- ☁️ **Cloud Database**: Neon DB for signal/trade history
- 🔄 **Learning Loop**: Track outcomes → Retrain → Improve
- 💰 **Realistic Costs**: Commission, slippage, bid-ask spread

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/HarsharajJ/rltrade.git
cd rltrade

# Install dependencies
uv sync

# Set up environment
cp .env.example .env
# Edit .env with your Neon DB connection string

# Initialize database
uv run python -m src.database --init

# Generate signals
uv run python live_signals_production.py --once
```

## 📁 Project Structure

```
rltrade/
├── live_signals_production.py   # 🎯 Main signal generator
├── train_production.py          # 🏋️ Train PPO model
├── track_outcomes.py            # 📊 Track trade outcomes
├── retrain_from_db.py           # 🔄 Retrain from history
├── src/
│   ├── database.py              # ☁️ Neon DB integration
│   ├── env_production.py        # 🎮 RL Environment
│   ├── features.py              # 📈 Feature engineering
│   ├── options_data.py          # 📊 Options data
│   ├── options_pricing.py       # 🧮 Black-Scholes
│   ├── data_loader.py           # 📥 Data fetching
│   └── walk_forward.py          # ✅ Validation
└── models/                      # 💾 Trained models
```

## 🧠 How the RL Works

```
┌─────────────────────────────────────────────────────────────┐
│                     REINFORCEMENT LEARNING                   │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│   STATE (15 dims)    →    PPO AGENT    →    ACTION (7)       │
│   [Price, RSI,            Neural Net        [Buy Call,       │
│    MACD, IV...]           (64x64)            Spread...]      │
│                               ↑                               │
│                           REWARD                              │
│                    (P&L - Costs - Penalties)                  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Actions

| Action | Type | Risk Profile |
|--------|------|--------------|
| `BUY_CALL` | Long call | Unlimited profit, premium risk |
| `BUY_PUT` | Long put | Unlimited profit, premium risk |
| `BULL_SPREAD` | Call debit spread | Defined risk/reward |
| `BEAR_SPREAD` | Put debit spread | Defined risk/reward |
| `IRON_CONDOR` | Sell wings | Defined risk, neutral |
| `HOLD` | No action | - |
| `CLOSE` | Exit position | - |

## 📈 Sample Output

```
>>> SPY OPTIONS SIGNAL <<<
Time: 2026-01-06T17:42:43

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

>>> SPREAD PRICING <<<
   [ENTRY]      $2.00 (debit)
   [BREAKEVEN]  $686.00

>>> SPY PRICE TARGETS <<<
   Current:    $687.72
   [TARGET]    $683.00  ← Max profit here
   [STOP]      $690.00  ← Exit if breached

>>> POSITION SIZE <<<
   Max Risk:    $200.00
   Max Profit:  $300.00
```

## 🔄 The Learning Loop

```
1. Generate Signal  →  Stored in Neon DB
         ↓
2. Market Moves     →  SPY price changes
         ↓
3. Track Outcome    →  Was it profitable?
         ↓
4. Retrain Model    →  Learn from results
         ↓
5. Better Signals   →  Improved predictions
```

## 🛠️ Commands

| Command | Description |
|---------|-------------|
| `uv run python live_signals_production.py --once` | Generate one signal |
| `uv run python live_signals_production.py` | Continuous signals |
| `uv run python track_outcomes.py --hours 24` | Track outcomes |
| `uv run python retrain_from_db.py --timesteps 100000` | Retrain model |
| `uv run python train_production.py --timesteps 500000` | Full training |
| `uv run python -m src.database --view` | View DB contents |
| `uv run python -m src.database --stats` | Performance stats |

## ⚙️ Configuration

Create a `.env` file:

```bash
# Neon DB (required)
DATABASE_URL=postgresql://user:pass@ep-xxx.neon.tech/neondb?sslmode=require

# Alpaca (optional - for live trading)
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
```

## 📊 Technical Details

### State Vector (15 Dimensions)

| Feature | Description |
|---------|-------------|
| SPY Price | Normalized current price |
| RSI | 14-period relative strength |
| MACD | Momentum indicator |
| IV | Implied volatility from VIX |
| IV Rank | 52-week IV percentile |
| Put/Call Ratio | Volume-based sentiment |
| Volume Surge | vs 20-day average |
| Gamma Proxy | Strike distance |
| Momentum | 5-day price change |
| + 6 more | Position & portfolio state |

### Transaction Costs

```python
COMMISSION = $0.65/contract
SLIPPAGE = 2% of premium
BID_ASK = $0.05/contract
```

## ⚠️ Disclaimer

> **RISK WARNING**: Options trading involves substantial risk of loss. This is for **educational purposes only**. Past performance does not guarantee future results. Always paper trade extensively before using real money.

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

Made with 🤖 by [HarsharajJ](https://github.com/HarsharajJ)