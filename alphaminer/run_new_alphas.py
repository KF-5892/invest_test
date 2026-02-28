#!/usr/bin/env python3
"""
Script to run just the 20 new alpha strategies.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import sys
import os

# Add alpha module to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from alpha import (
    AlphaEngine,
    GapReversal,
    SimpleMA,
    VolatilityRank,
    PriceVelocity,
    RollingBeta,
    TrendConsistency,
    ReturnSkew,
    DrawdownRecovery,
    VolumePrice,
    SeasonalTrend,
    ReturnDensity,
    MomentumDecay,
    VolBreakout,
    MeanReversionSpeed,
    TailRisk,
    ReturnStability,
    SimpleMomentum,
    NoiseRatio
)

def create_new_alphas():
    """Create only the 20 new alpha strategies."""
    alphas = [
        GapReversal(0.02),
        SimpleMA(10),
        VolatilityRank(20, 60),
        PriceVelocity(5),
        RollingBeta(30),
        TrendConsistency(10),
        ReturnSkew(30),
        DrawdownRecovery(60),
        VolumePrice(20),
        SeasonalTrend(252),
        ReturnDensity(30, 0.005),
        MomentumDecay(5, 20),
        VolBreakout(20, 1.5),
        MeanReversionSpeed(15),
        TailRisk(60, 0.05),
        ReturnStability(30),
        SimpleMomentum(7),
        NoiseRatio(20),
        SimpleMA(20),
        VolatilityRank(15, 45),
    ]
    return alphas

def run_new_alphas():
    """Run the 20 new alpha strategies and append to existing results."""
    
    # Initialize engine
    engine = AlphaEngine('data/48_Industry_Portfolios_Daily.csv')
    engine.load_data()
    
    # Get recent 3 years of data for evaluation
    end_date = engine.data.index[-1]
    start_date = end_date - timedelta(days=3*365)
    
    print(f"Running new alphas from {start_date.date()} to {end_date.date()}")
    
    # Load existing results
    try:
        with open('alpha_results.json', 'r') as f:
            existing_results = json.load(f)
    except FileNotFoundError:
        existing_results = {}
    
    # Create new alphas
    alphas = create_new_alphas()
    
    print(f"Running {len(alphas)} new alpha strategies...")
    
    # Run each alpha
    new_results = {}
    
    for i, alpha in enumerate(alphas):
        print(f"Running alpha {i+1}/{len(alphas)}: {alpha.name}")
        
        try:
            engine.register_alpha(alpha)
            engine.run_alpha(alpha.name, start_date=start_date.strftime('%Y-%m-%d'))
            
            # Calculate performance metrics
            metrics = engine.calculate_performance_metrics(alpha.name)
            
            new_results[alpha.name] = {
                'description': alpha.description,
                'metrics': metrics,
                'returns_data': engine.results[alpha.name]['returns'].to_list(),
                'dates': [d.strftime('%Y-%m-%d') for d in engine.results[alpha.name]['returns'].index]
            }
            
            print(f"  - Sharpe: {metrics['sharpe_ratio']:.3f}, Annual Return: {metrics['annualized_return']:.3f}")
            
        except Exception as e:
            print(f"  - Error: {e}")
            new_results[alpha.name] = {'error': str(e)}
    
    # Merge with existing results
    all_results = {**existing_results, **new_results}
    
    # Save updated results to JSON
    with open('alpha_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nUpdated results saved to alpha_results.json")
    print(f"Total alphas: {len(all_results)}")
    
    # Create updated leaderboard
    create_leaderboard(all_results)
    
    return all_results

def create_leaderboard(results):
    """Create a leaderboard of alpha performance."""
    
    leaderboard_data = []
    
    for alpha_name, result in results.items():
        if 'metrics' in result:
            metrics = result['metrics']
            leaderboard_data.append({
                'name': alpha_name,
                'description': result.get('description', ''),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'annualized_return': metrics.get('annualized_return', 0),
                'calmar_ratio': metrics.get('calmar_ratio', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'volatility': metrics.get('volatility', 0),
                'num_trades': metrics.get('num_trades', 0)
            })
    
    # Sort by Sharpe ratio
    leaderboard_data.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
    
    # Save leaderboard
    with open('alpha_leaderboard.json', 'w') as f:
        json.dump(leaderboard_data, f, indent=2)
    
    # Print top performers
    print("\n" + "="*80)
    print("UPDATED ALPHA LEADERBOARD - TOP 15 BY SHARPE RATIO")
    print("="*80)
    print(f"{'Rank':<4} {'Alpha Name':<35} {'Sharpe':<8} {'Ann.Ret':<8} {'Calmar':<8} {'MaxDD':<8}")
    print("-"*80)
    
    for i, alpha in enumerate(leaderboard_data[:15]):
        print(f"{i+1:<4} {alpha['name'][:34]:<35} "
              f"{alpha['sharpe_ratio']:<8.3f} "
              f"{alpha['annualized_return']:<8.3f} "
              f"{alpha['calmar_ratio']:<8.3f} "
              f"{alpha['max_drawdown']:<8.3f}")
    
    print("="*80)
    
    # Show new alphas specifically
    print("\nNEW ALPHAS PERFORMANCE:")
    print("-"*60)
    new_alpha_names = [
        'gap_reversal_0.02', 'simple_ma_10d', 'vol_rank_20d_60d', 'price_velocity_5d',
        'rolling_beta_30d', 'trend_consistency_10d', 'return_skew_30d', 'drawdown_recovery_60d',
        'volume_price_20d', 'seasonal_trend_252d', 'return_density_30d_0.005', 'momentum_decay_5d_20d',
        'vol_breakout_20d_1.5x', 'mean_reversion_speed_15d', 'tail_risk_60d_0.05', 'return_stability_30d',
        'simple_momentum_7d', 'noise_ratio_20d', 'simple_ma_20d', 'vol_rank_15d_45d'
    ]
    
    new_alphas = [alpha for alpha in leaderboard_data if alpha['name'] in new_alpha_names]
    new_alphas.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
    
    for i, alpha in enumerate(new_alphas):
        rank = next((j+1 for j, a in enumerate(leaderboard_data) if a['name'] == alpha['name']), 0)
        print(f"#{rank:<3} {alpha['name']:<35} Sharpe: {alpha['sharpe_ratio']:<7.3f} Return: {alpha['annualized_return']:<7.3f}")

if __name__ == "__main__":
    results = run_new_alphas()