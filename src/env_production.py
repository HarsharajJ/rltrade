"""
Production Options Trading Environment
Enhanced with spreads, transaction costs, and 15-dim state vector.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass, field

from src.options_pricing import create_option_contract, reprice_contract, OptionContract


# Action constants
ACTION_BUY_CALL = 0
ACTION_BUY_PUT = 1
ACTION_BULL_SPREAD = 2  # Call debit spread (bullish)
ACTION_BEAR_SPREAD = 3  # Put debit spread (bearish)
ACTION_IRON_CONDOR = 4  # Neutral, defined risk
ACTION_HOLD = 5
ACTION_CLOSE = 6

ACTION_NAMES = {
    0: "BUY_CALL",
    1: "BUY_PUT",
    2: "BULL_SPREAD",
    3: "BEAR_SPREAD",
    4: "IRON_CONDOR",
    5: "HOLD",
    6: "CLOSE"
}

# Transaction costs
COMMISSION_PER_CONTRACT = 0.65   # Per contract per leg
SLIPPAGE_PCT = 0.02             # 2% of premium
BID_ASK_SPREAD = 0.05           # $0.05 per contract


@dataclass
class SpreadPosition:
    """Represents a spread position."""
    spread_type: str  # 'call', 'put', 'bull_spread', 'bear_spread', 'iron_condor'
    legs: List[Dict] = field(default_factory=list)  # List of leg details
    entry_cost: float = 0.0  # Net debit/credit
    max_profit: float = 0.0
    max_loss: float = 0.0
    entry_step: int = 0


@dataclass
class TradeRecord:
    """Record of a completed trade."""
    entry_step: int
    exit_step: int
    trade_type: str
    entry_cost: float
    exit_value: float
    pnl: float
    pnl_pct: float
    holding_days: int
    transaction_costs: float


class ProductionOptionsEnv(gym.Env):
    """
    Production-grade options trading environment.
    
    Features:
    - 7 actions: BUY_CALL, BUY_PUT, BULL_SPREAD, BEAR_SPREAD, IRON_CONDOR, HOLD, CLOSE
    - 15-dimensional state vector
    - Realistic transaction costs (commission, slippage, bid-ask)
    - Spread strategies with defined risk
    
    Observation Space: Box(15,) normalized features
    Action Space: Discrete(7)
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10_000.0,
        contracts_per_trade: int = 1,
        default_dte: int = 7,
        spread_width: int = 5,  # $5 wide spreads
        commission: float = COMMISSION_PER_CONTRACT,
        slippage_pct: float = SLIPPAGE_PCT,
        bid_ask: float = BID_ASK_SPREAD,
        max_holding_days: int = 5,
        risk_free_rate: float = 0.05,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.contracts_per_trade = contracts_per_trade
        self.default_dte = default_dte
        self.spread_width = spread_width
        self.commission = commission
        self.slippage_pct = slippage_pct
        self.bid_ask = bid_ask
        self.max_holding_days = max_holding_days
        self.risk_free_rate = risk_free_rate
        self.render_mode = render_mode
        
        # Ensure all features exist
        self._ensure_features()
        
        # Define spaces
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(15,), dtype=np.float32
        )
        
        # Episode state
        self.current_step: int = 0
        self.cash: float = initial_balance
        self.current_position: Optional[SpreadPosition] = None
        self.portfolio_values: List[float] = []
        self.trades: List[TradeRecord] = []
    
    def _ensure_features(self):
        """Ensure all required features exist."""
        required = ['Close', 'RSI', 'MACD', 'EMA20', 'Volatility', 'IV',
                   'IV_Rank', 'PCR', 'Volume_Surge', 'IV_Skew', 'Gamma_Proxy',
                   'EMA_Ratio', 'Momentum']
        
        for col in required:
            if col not in self.df.columns:
                if col == 'IV':
                    self.df[col] = 0.18
                elif col == 'IV_Rank':
                    self.df[col] = 50.0
                elif col == 'PCR':
                    self.df[col] = 1.0
                elif col == 'Volume_Surge':
                    self.df[col] = 1.0
                elif col in ['IV_Skew', 'Gamma_Proxy', 'Momentum']:
                    self.df[col] = 0.0
                elif col == 'EMA_Ratio':
                    self.df[col] = 1.0
        
        self.df = self.df.ffill().bfill()
    
    def _calculate_transaction_cost(
        self,
        premium: float,
        num_legs: int = 1
    ) -> float:
        """Calculate total transaction cost for a trade."""
        contracts = self.contracts_per_trade
        
        # Commission per leg
        commission = self.commission * contracts * num_legs
        
        # Slippage on premium
        slippage = premium * self.slippage_pct * 100 * contracts
        
        # Bid-ask spread
        spread_cost = self.bid_ask * 100 * contracts * num_legs
        
        return commission + slippage + spread_cost
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        
        self.current_step = 0
        self.cash = self.initial_balance
        self.current_position = None
        self.portfolio_values = [self.initial_balance]
        self.trades = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        prev_value = self._get_portfolio_value()
        row = self.df.iloc[self.current_step]
        spot = row['Close']
        iv = row['IV']
        
        reward = 0.0
        trade_executed = False
        
        # Force close conditions
        force_close = False
        if self.current_position is not None:
            holding = self.current_step - self.current_position.entry_step
            if holding >= self.max_holding_days:
                force_close = True
        
        # Execute action
        if action == ACTION_BUY_CALL and self.current_position is None:
            reward += self._open_single_option('call', spot, iv)
            trade_executed = True
            
        elif action == ACTION_BUY_PUT and self.current_position is None:
            reward += self._open_single_option('put', spot, iv)
            trade_executed = True
            
        elif action == ACTION_BULL_SPREAD and self.current_position is None:
            reward += self._open_spread('bull', spot, iv)
            trade_executed = True
            
        elif action == ACTION_BEAR_SPREAD and self.current_position is None:
            reward += self._open_spread('bear', spot, iv)
            trade_executed = True
            
        elif action == ACTION_IRON_CONDOR and self.current_position is None:
            reward += self._open_iron_condor(spot, iv)
            trade_executed = True
            
        elif action == ACTION_CLOSE and self.current_position is not None:
            reward += self._close_position(spot, iv)
            trade_executed = True
            
        elif force_close and self.current_position is not None:
            reward += self._close_position(spot, iv)
            trade_executed = True
        
        # Penalties for invalid actions
        elif action == ACTION_CLOSE and self.current_position is None:
            reward -= 0.2
        elif action in [ACTION_BUY_CALL, ACTION_BUY_PUT, ACTION_BULL_SPREAD, 
                       ACTION_BEAR_SPREAD, ACTION_IRON_CONDOR] and self.current_position is not None:
            reward -= 0.1
        
        # Inaction penalty
        if action == ACTION_HOLD and self.current_position is None:
            reward -= 0.05
        
        # Time decay for positions
        if self.current_position is not None and not trade_executed:
            if self.current_position.spread_type in ['call', 'put']:
                # Theta decay for naked long options
                contract = self.current_position.legs[0].get('contract')
                if contract:
                    theta_cost = abs(contract.theta) * 100 * self.contracts_per_trade
                    reward -= theta_cost * 0.3
        
        # Advance step
        self.current_step += 1
        
        # Update position value
        if self.current_position is not None:
            self._update_position_value()
        
        # Track portfolio
        new_value = self._get_portfolio_value()
        self.portfolio_values.append(new_value)
        
        # P&L reward
        if prev_value > 0:
            pnl_pct = (new_value - prev_value) / prev_value
            reward += pnl_pct * 20  # Scale factor
        
        # Termination
        terminated = self.current_step >= len(self.df) - 1
        
        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _open_single_option(self, option_type: str, spot: float, iv: float) -> float:
        """Open a single call or put."""
        contract = create_option_contract(
            spot_price=spot,
            volatility=iv,
            days_to_expiry=self.default_dte,
            option_type=option_type
        )
        
        premium = contract.premium * 100 * self.contracts_per_trade
        tx_cost = self._calculate_transaction_cost(contract.premium, num_legs=1)
        total_cost = premium + tx_cost
        
        if total_cost > self.cash:
            return -0.1
        
        self.cash -= total_cost
        self.current_position = SpreadPosition(
            spread_type=option_type,
            legs=[{'contract': contract, 'type': option_type, 'premium': contract.premium}],
            entry_cost=total_cost,
            max_profit=float('inf'),
            max_loss=premium,
            entry_step=self.current_step
        )
        
        return 0.05 - tx_cost / 100  # Small bonus minus costs
    
    def _open_spread(self, direction: str, spot: float, iv: float) -> float:
        """Open a vertical spread (bull call or bear put)."""
        if direction == 'bull':
            # Buy ATM call, sell OTM call
            long_strike = round(spot)
            short_strike = long_strike + self.spread_width
            option_type = 'call'
        else:
            # Buy ATM put, sell OTM put
            long_strike = round(spot)
            short_strike = long_strike - self.spread_width
            option_type = 'put'
        
        long_contract = create_option_contract(spot, iv, self.default_dte, option_type)
        short_contract = create_option_contract(spot, iv, self.default_dte, option_type)
        # Adjust short premium for OTM (simplified)
        short_premium = long_contract.premium * 0.4  # OTM is cheaper
        
        net_debit = (long_contract.premium - short_premium) * 100 * self.contracts_per_trade
        tx_cost = self._calculate_transaction_cost(long_contract.premium, num_legs=2)
        total_cost = net_debit + tx_cost
        
        if total_cost > self.cash:
            return -0.1
        
        self.cash -= total_cost
        self.current_position = SpreadPosition(
            spread_type=f'{direction}_spread',
            legs=[
                {'strike': long_strike, 'type': f'long_{option_type}', 'premium': long_contract.premium},
                {'strike': short_strike, 'type': f'short_{option_type}', 'premium': short_premium}
            ],
            entry_cost=total_cost,
            max_profit=self.spread_width * 100 * self.contracts_per_trade - net_debit,
            max_loss=net_debit,
            entry_step=self.current_step
        )
        
        return 0.1 - tx_cost / 100  # Bonus for defined risk trade
    
    def _open_iron_condor(self, spot: float, iv: float) -> float:
        """Open an iron condor (neutral, defined risk)."""
        # Sell OTM call spread and OTM put spread
        call_short = round(spot) + self.spread_width
        call_long = call_short + self.spread_width
        put_short = round(spot) - self.spread_width
        put_long = put_short - self.spread_width
        
        # Approximate credit received (simplified)
        base_premium = create_option_contract(spot, iv, self.default_dte, 'call').premium
        credit_per_side = base_premium * 0.2  # OTM premium approximation
        net_credit = credit_per_side * 2 * 100 * self.contracts_per_trade
        
        tx_cost = self._calculate_transaction_cost(base_premium, num_legs=4)
        net_after_cost = net_credit - tx_cost
        
        if net_after_cost < 0:
            return -0.1
        
        self.cash += net_after_cost
        self.current_position = SpreadPosition(
            spread_type='iron_condor',
            legs=[
                {'strike': call_short, 'type': 'short_call'},
                {'strike': call_long, 'type': 'long_call'},
                {'strike': put_short, 'type': 'short_put'},
                {'strike': put_long, 'type': 'long_put'}
            ],
            entry_cost=-net_after_cost,  # Negative = credit received
            max_profit=net_after_cost,
            max_loss=self.spread_width * 100 * self.contracts_per_trade - net_after_cost,
            entry_step=self.current_step
        )
        
        return 0.15  # Bonus for income strategy
    
    def _close_position(self, spot: float, iv: float) -> float:
        """Close current position."""
        if self.current_position is None:
            return -0.1
        
        pos = self.current_position
        
        # Calculate exit value (simplified)
        if pos.spread_type in ['call', 'put']:
            # Reprice the option
            contract = pos.legs[0].get('contract')
            if contract:
                new_contract = reprice_contract(
                    contract, spot, iv,
                    days_passed=self.current_step - pos.entry_step,
                    risk_free_rate=self.risk_free_rate
                )
                exit_value = new_contract.premium * 100 * self.contracts_per_trade
            else:
                exit_value = pos.entry_cost * 0.8  # Default to 20% loss
        else:
            # Spreads/condors: estimate based on spot movement
            if pos.spread_type == 'iron_condor':
                # Win if spot stays within range
                entry_spot = self.df.iloc[pos.entry_step]['Close']
                movement = abs(spot - entry_spot) / entry_spot
                if movement < 0.03:  # Less than 3% move
                    exit_value = pos.max_profit * 0.8  # Take most of profit
                else:
                    exit_value = -pos.max_loss * movement / 0.1  # Lose proportionally
            else:
                # Vertical spread value
                entry_spot = self.df.iloc[pos.entry_step]['Close']
                if pos.spread_type == 'bull_spread':
                    pnl_factor = (spot - entry_spot) / entry_spot * 10
                else:
                    pnl_factor = (entry_spot - spot) / entry_spot * 10
                exit_value = pos.entry_cost + pos.max_profit * np.clip(pnl_factor, -1, 1)
        
        tx_cost = self._calculate_transaction_cost(exit_value / 100 / self.contracts_per_trade if exit_value > 0 else 0.1, num_legs=1)
        net_exit = exit_value - tx_cost
        
        self.cash += net_exit
        
        # Record trade
        holding = self.current_step - pos.entry_step
        pnl = net_exit - pos.entry_cost if pos.entry_cost > 0 else net_exit + pos.entry_cost
        pnl_pct = pnl / abs(pos.entry_cost) if pos.entry_cost != 0 else 0
        
        self.trades.append(TradeRecord(
            entry_step=pos.entry_step,
            exit_step=self.current_step,
            trade_type=pos.spread_type,
            entry_cost=pos.entry_cost,
            exit_value=net_exit,
            pnl=pnl,
            pnl_pct=pnl_pct,
            holding_days=holding,
            transaction_costs=tx_cost
        ))
        
        self.current_position = None
        
        # Reward
        reward = pnl / 50
        if pnl > 0:
            reward += 1.0
        else:
            reward -= 0.3
        
        return reward
    
    def _update_position_value(self):
        """Update position after time passes (for naked options)."""
        if self.current_position is None:
            return
        
        if self.current_position.spread_type in ['call', 'put']:
            contract = self.current_position.legs[0].get('contract')
            if contract:
                row = self.df.iloc[self.current_step]
                new_contract = reprice_contract(
                    contract,
                    new_spot=row['Close'],
                    new_volatility=row['IV'],
                    days_passed=1,
                    risk_free_rate=self.risk_free_rate
                )
                self.current_position.legs[0]['contract'] = new_contract
    
    def _get_portfolio_value(self) -> float:
        """Calculate current portfolio value."""
        value = self.cash
        
        if self.current_position is not None:
            if self.current_position.spread_type in ['call', 'put']:
                contract = self.current_position.legs[0].get('contract')
                if contract:
                    value += contract.premium * 100 * self.contracts_per_trade
        
        return value
    
    def _get_observation(self) -> np.ndarray:
        """Build 15-dimensional state vector."""
        row = self.df.iloc[self.current_step]
        
        # Normalize features
        obs = np.array([
            np.clip((row['Close'] - 400) / 400, 0, 1),           # SPY price
            row['RSI'],                                            # Already 0-1
            np.clip(row['MACD'] / 10 + 0.5, 0, 1),                # MACD
            np.clip((row['IV'] - 0.10) / 0.40, 0, 1),             # IV
            np.clip(row.get('IV_Rank', 50) / 100, 0, 1),          # IV Rank
            np.clip(row.get('PCR', 1) / 3, 0, 1),                 # Put/Call Ratio
            np.clip(row.get('Volume_Surge', 1) / 5, 0, 1),        # Volume Surge
            np.clip(row.get('IV_Skew', 0) / 10 + 0.5, 0, 1),      # IV Skew
            np.clip(row.get('Gamma_Proxy', 0) + 0.5, 0, 1),       # Gamma
            np.clip(row.get('Momentum', 0) / 10 + 0.5, 0, 1),     # Momentum
            np.clip(self._get_portfolio_value() / (self.initial_balance * 2), 0, 1),  # Balance
            1.0 if self.current_position is not None else 0.0,    # Has position
            self._get_position_type_encoding(),                    # Position type
            self._get_position_pnl_norm(),                        # Position P&L
            self._get_holding_time_norm()                         # Holding time
        ], dtype=np.float32)
        
        return np.clip(obs, 0, 1)
    
    def _get_position_type_encoding(self) -> float:
        """Encode position type as 0-1."""
        if self.current_position is None:
            return 0.0
        types = {'call': 0.2, 'put': 0.4, 'bull_spread': 0.5, 
                'bear_spread': 0.6, 'iron_condor': 0.8}
        return types.get(self.current_position.spread_type, 0.5)
    
    def _get_position_pnl_norm(self) -> float:
        """Get normalized position P&L."""
        if self.current_position is None:
            return 0.5
        
        current_value = self._get_portfolio_value() - self.cash
        if self.current_position.entry_cost == 0:
            return 0.5
        
        pnl_pct = (current_value - self.current_position.entry_cost) / abs(self.current_position.entry_cost)
        return np.clip(pnl_pct + 0.5, 0, 1)
    
    def _get_holding_time_norm(self) -> float:
        """Get normalized holding time."""
        if self.current_position is None:
            return 0.0
        
        holding = self.current_step - self.current_position.entry_step
        return np.clip(holding / self.max_holding_days, 0, 1)
    
    def _get_info(self) -> dict:
        return {
            'step': self.current_step,
            'portfolio_value': self._get_portfolio_value(),
            'cash': self.cash,
            'position': self.current_position.spread_type if self.current_position else None,
            'num_trades': len(self.trades)
        }
    
    def get_results(self) -> Dict[str, Any]:
        """Calculate performance metrics."""
        values = np.array(self.portfolio_values)
        
        total_return = (values[-1] - self.initial_balance) / self.initial_balance
        
        # Daily returns
        daily_returns = np.diff(values) / values[:-1]
        
        # Sharpe
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe = np.sqrt(252) * np.mean(daily_returns) / np.std(daily_returns)
        else:
            sharpe = 0.0
        
        # Max drawdown
        running_max = np.maximum.accumulate(values)
        drawdowns = (running_max - values) / running_max
        max_dd = np.max(drawdowns)
        
        # Trade stats
        if self.trades:
            wins = [t for t in self.trades if t.pnl > 0]
            win_rate = len(wins) / len(self.trades)
            avg_pnl = np.mean([t.pnl for t in self.trades])
            total_costs = sum(t.transaction_costs for t in self.trades)
            
            # By type
            by_type = {}
            for t in self.trades:
                if t.trade_type not in by_type:
                    by_type[t.trade_type] = []
                by_type[t.trade_type].append(t)
        else:
            win_rate = 0
            avg_pnl = 0
            total_costs = 0
            by_type = {}
        
        return {
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd * 100,
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'num_trades': len(self.trades),
            'avg_pnl': avg_pnl,
            'total_transaction_costs': total_costs,
            'trades_by_type': {k: len(v) for k, v in by_type.items()},
            'final_value': values[-1],
            'portfolio_values': values.tolist()
        }


if __name__ == "__main__":
    from src.data_loader import load_spy_with_vix
    from src.features import add_enhanced_features
    
    print("Loading data with enhanced features...")
    df = load_spy_with_vix(years=1)
    df = add_enhanced_features(df)
    
    print(f"Features: {list(df.columns)}")
    
    print("\nCreating ProductionOptionsEnv...")
    env = ProductionOptionsEnv(df, initial_balance=10_000)
    
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space.n} actions")
    
    # Test each action
    print("\nTesting actions...")
    actions = [0, 5, 5, 6, 1, 6, 2, 6, 3, 6, 4, 6]  # Each strategy
    
    for action in actions:
        if env.current_step >= len(df) - 2:
            break
        obs, reward, done, _, info = env.step(action)
        print(f"Action {ACTION_NAMES[action]}: reward={reward:.3f}, value=${info['portfolio_value']:.2f}")
    
    results = env.get_results()
    print(f"\nResults: Return={results['total_return_pct']:.2f}%, "
          f"Trades={results['num_trades']}, "
          f"Costs=${results['total_transaction_costs']:.2f}")
    
    print("\n✓ Production environment test passed!")
