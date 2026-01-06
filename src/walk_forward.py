"""
Walk-Forward Validation
Proper out-of-sample testing with rolling windows.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


def walk_forward_split(
    df: pd.DataFrame,
    train_years: int = 2,
    test_months: int = 3,
    step_months: int = 3
) -> List[Dict[str, pd.DataFrame]]:
    """
    Generate walk-forward train/test splits.
    
    Example with train_years=2, test_months=3:
    - Window 1: Train 2020-2021, Test 2022-Q1
    - Window 2: Train 2020-2022 Q1, Test 2022-Q2
    - Window 3: Train 2020-2022 Q2, Test 2022-Q3
    
    Args:
        df: Full DataFrame with date index.
        train_years: Years of training data.
        test_months: Months of test data.
        step_months: How many months to advance each window.
    
    Returns:
        List of dicts with 'train' and 'test' DataFrames.
    """
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    splits = []
    start_date = df.index.min()
    end_date = df.index.max()
    
    train_days = train_years * 252  # Trading days per year
    test_days = test_months * 21    # Trading days per month
    step_days = step_months * 21
    
    current_start = 0
    
    while True:
        train_end = current_start + train_days
        test_end = train_end + test_days
        
        if test_end > len(df):
            break
        
        train_df = df.iloc[current_start:train_end].copy()
        test_df = df.iloc[train_end:test_end].copy()
        
        if len(train_df) >= train_days * 0.9 and len(test_df) >= test_days * 0.5:
            splits.append({
                'train': train_df,
                'test': test_df,
                'train_start': train_df.index[0].strftime('%Y-%m-%d'),
                'train_end': train_df.index[-1].strftime('%Y-%m-%d'),
                'test_start': test_df.index[0].strftime('%Y-%m-%d'),
                'test_end': test_df.index[-1].strftime('%Y-%m-%d')
            })
        
        current_start += step_days
    
    return splits


def train_and_evaluate_window(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    window_id: int,
    timesteps: int = 100_000,
    n_envs: int = 4,
    save_dir: str = "models/walk_forward"
) -> Dict[str, Any]:
    """
    Train model on train data and evaluate on test data.
    
    Returns:
        Dict with training info and test results.
    """
    from src.env_production import ProductionOptionsEnv
    from src.features import add_enhanced_features
    
    # Ensure features
    train_df = add_enhanced_features(train_df)
    test_df = add_enhanced_features(test_df)
    
    # Create environments
    def make_train_env():
        return ProductionOptionsEnv(
            train_df,
            initial_balance=10_000,
            contracts_per_trade=1
        )
    
    vec_env = make_vec_env(make_train_env, n_envs=n_envs)
    
    # Train
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        ent_coef=0.05,
        clip_range=0.2,
        verbose=0
    )
    
    model.learn(total_timesteps=timesteps, progress_bar=True)
    
    # Save model
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    model_path = f"{save_dir}/window_{window_id}.zip"
    model.save(model_path)
    
    # Evaluate on test
    test_env = ProductionOptionsEnv(test_df, initial_balance=10_000)
    obs, _ = test_env.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=False)
        obs, _, terminated, truncated, _ = test_env.step(action)
        done = terminated or truncated
    
    results = test_env.get_results()
    results['window_id'] = window_id
    results['model_path'] = model_path
    
    return results


def run_walk_forward(
    years_data: int = 3,
    train_years: int = 2,
    test_months: int = 3,
    timesteps: int = 100_000,
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Run complete walk-forward validation.
    
    Args:
        years_data: Total years of data to use.
        train_years: Years of training per window.
        test_months: Months of testing per window.
        timesteps: Training timesteps per window.
        save_results: Whether to save results to JSON.
    
    Returns:
        Aggregated results across all windows.
    """
    from src.data_loader import load_spy_with_vix
    
    print("=" * 60)
    print("WALK-FORWARD VALIDATION")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading {years_data} years of SPY + VIX data...")
    df = load_spy_with_vix(years=years_data)
    
    # Generate splits
    splits = walk_forward_split(df, train_years, test_months)
    print(f"Generated {len(splits)} walk-forward windows")
    
    for i, split in enumerate(splits):
        print(f"  Window {i+1}: Train {split['train_start']} to {split['train_end']}, "
              f"Test {split['test_start']} to {split['test_end']}")
    
    # Train and evaluate each window
    all_results = []
    
    for i, split in enumerate(splits):
        print(f"\n{'='*60}")
        print(f"WINDOW {i+1}/{len(splits)}")
        print(f"{'='*60}")
        print(f"Train: {split['train_start']} to {split['train_end']}")
        print(f"Test: {split['test_start']} to {split['test_end']}")
        
        result = train_and_evaluate_window(
            split['train'],
            split['test'],
            window_id=i+1,
            timesteps=timesteps
        )
        
        result['train_period'] = f"{split['train_start']} to {split['train_end']}"
        result['test_period'] = f"{split['test_start']} to {split['test_end']}"
        all_results.append(result)
        
        print(f"\nWindow {i+1} Results:")
        print(f"  Return: {result['total_return_pct']:.2f}%")
        print(f"  Sharpe: {result['sharpe_ratio']:.3f}")
        print(f"  Max DD: {result['max_drawdown_pct']:.2f}%")
        print(f"  Trades: {result['num_trades']}")
    
    # Aggregate results
    print(f"\n{'='*60}")
    print("AGGREGATED RESULTS")
    print(f"{'='*60}")
    
    returns = [r['total_return_pct'] for r in all_results]
    sharpes = [r['sharpe_ratio'] for r in all_results]
    drawdowns = [r['max_drawdown_pct'] for r in all_results]
    trades = [r['num_trades'] for r in all_results]
    
    aggregated = {
        'num_windows': len(all_results),
        'avg_return': np.mean(returns),
        'std_return': np.std(returns),
        'min_return': np.min(returns),
        'max_return': np.max(returns),
        'avg_sharpe': np.mean(sharpes),
        'avg_max_dd': np.mean(drawdowns),
        'worst_dd': np.max(drawdowns),
        'avg_trades': np.mean(trades),
        'total_trades': sum(trades),
        'windows': all_results
    }
    
    print(f"\nAcross {len(all_results)} windows:")
    print(f"  Avg Return: {aggregated['avg_return']:.2f}% (+/- {aggregated['std_return']:.2f}%)")
    print(f"  Return Range: {aggregated['min_return']:.2f}% to {aggregated['max_return']:.2f}%")
    print(f"  Avg Sharpe: {aggregated['avg_sharpe']:.3f}")
    print(f"  Avg Max DD: {aggregated['avg_max_dd']:.2f}%")
    print(f"  Worst DD: {aggregated['worst_dd']:.2f}%")
    print(f"  Avg Trades/Window: {aggregated['avg_trades']:.1f}")
    
    # Save results
    if save_results:
        results_path = "models/walk_forward/results.json"
        with open(results_path, 'w') as f:
            # Convert numpy types for JSON serialization
            def convert(obj):
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            clean_results = json.loads(json.dumps(aggregated, default=convert))
            json.dump(clean_results, f, indent=2)
        
        print(f"\nResults saved to: {results_path}")
    
    return aggregated


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Walk-Forward Validation")
    parser.add_argument("--years", type=int, default=3, help="Years of data")
    parser.add_argument("--train-years", type=int, default=2, help="Training years per window")
    parser.add_argument("--test-months", type=int, default=3, help="Test months per window")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Training timesteps")
    
    args = parser.parse_args()
    
    run_walk_forward(
        years_data=args.years,
        train_years=args.train_years,
        test_months=args.test_months,
        timesteps=args.timesteps
    )
