"""
Production Training Script
Multi-seed training with enhanced environment and proper validation.
"""

import argparse
import os
from pathlib import Path
from datetime import datetime

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback


def train_production(
    timesteps: int = 500_000,
    n_envs: int = 4,
    n_seeds: int = 3,
    learning_rate: float = 3e-4,
    save_dir: str = "models/production",
    years_data: int = 3
):
    """
    Production-grade training with multiple seeds.
    
    Args:
        timesteps: Total timesteps per seed
        n_envs: Number of parallel environments
        n_seeds: Number of random seeds to train
        learning_rate: PPO learning rate
        save_dir: Directory to save models
        years_data: Years of training data
    """
    from src.data_loader import load_spy_with_vix
    from src.features import add_enhanced_features
    from src.env_production import ProductionOptionsEnv
    
    print("=" * 60)
    print("PRODUCTION TRAINING")
    print("=" * 60)
    print(f"Timesteps: {timesteps:,}")
    print(f"Seeds: {n_seeds}")
    print(f"Envs: {n_envs}")
    
    # Load and prepare data
    print(f"\nLoading {years_data} years of data...")
    df = load_spy_with_vix(years=years_data)
    df = add_enhanced_features(df)
    
    # Split: 80% train, 20% validation
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()
    
    print(f"Train: {len(train_df)} days")
    print(f"Validation: {len(val_df)} days")
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Train multiple seeds
    all_results = []
    
    for seed in range(n_seeds):
        print(f"\n{'='*60}")
        print(f"SEED {seed+1}/{n_seeds}")
        print(f"{'='*60}")
        
        # Create environments
        def make_train_env():
            return ProductionOptionsEnv(train_df, initial_balance=10_000)
        
        vec_env = make_vec_env(make_train_env, n_envs=n_envs)
        eval_env = ProductionOptionsEnv(val_df, initial_balance=10_000)
        
        # Callbacks
        checkpoint_cb = CheckpointCallback(
            save_freq=max(50_000 // n_envs, 1000),
            save_path=f"{save_dir}/checkpoints_seed{seed}",
            name_prefix=f"ppo_seed{seed}"
        )
        
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=f"{save_dir}/best_seed{seed}",
            log_path=f"{save_dir}/logs_seed{seed}",
            eval_freq=max(25_000 // n_envs, 500),
            deterministic=False,
            n_eval_episodes=1
        )
        
        # Create model
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=learning_rate,
            n_steps=1024,
            batch_size=64,
            n_epochs=10,
            ent_coef=0.05,
            clip_range=0.2,
            seed=seed * 42,
            verbose=0,
            tensorboard_log=f"{save_dir}/tensorboard_seed{seed}"
        )
        
        # Train
        print(f"Training for {timesteps:,} timesteps...")
        model.learn(
            total_timesteps=timesteps,
            callback=[checkpoint_cb, eval_cb],
            progress_bar=True
        )
        
        # Save final model
        model_path = f"{save_dir}/ppo_production_seed{seed}.zip"
        model.save(model_path)
        print(f"Model saved: {model_path}")
        
        # Evaluate
        print("Evaluating on validation set...")
        obs, _ = eval_env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, _, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
        
        results = eval_env.get_results()
        results['seed'] = seed
        results['model_path'] = model_path
        all_results.append(results)
        
        print(f"Seed {seed+1} Results:")
        print(f"  Return: {results['total_return_pct']:.2f}%")
        print(f"  Sharpe: {results['sharpe_ratio']:.3f}")
        print(f"  Max DD: {results['max_drawdown_pct']:.2f}%")
        print(f"  Trades: {results['num_trades']}")
        print(f"  TX Costs: ${results['total_transaction_costs']:.2f}")
    
    # Summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    
    returns = [r['total_return_pct'] for r in all_results]
    sharpes = [r['sharpe_ratio'] for r in all_results]
    
    print(f"\nAcross {n_seeds} seeds:")
    print(f"  Avg Return: {np.mean(returns):.2f}%")
    print(f"  Return Std: {np.std(returns):.2f}%")
    print(f"  Best Return: {np.max(returns):.2f}% (seed {np.argmax(returns)})")
    print(f"  Avg Sharpe: {np.mean(sharpes):.3f}")
    
    # Select best model
    best_idx = np.argmax(returns)
    best_model_path = all_results[best_idx]['model_path']
    
    # Copy to main model file
    import shutil
    final_path = f"{save_dir}/ppo_production_best.zip"
    shutil.copy(best_model_path, final_path)
    print(f"\nBest model copied to: {final_path}")
    
    print("\n✓ Production training complete!")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Production Training")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--years", type=int, default=3)
    
    args = parser.parse_args()
    
    train_production(
        timesteps=args.timesteps,
        n_seeds=args.seeds,
        n_envs=args.envs,
        years_data=args.years
    )
