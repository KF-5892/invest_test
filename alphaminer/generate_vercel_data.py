#!/usr/bin/env python3
"""
Generate static data files for the Vercel frontend deployment.
This script copies and processes data from the main project to the frontend's public directory.
"""

import json
import shutil
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta

def generate_cumulative_returns(alphas, alpha_results_path):
    """
    Generate cumulative returns data for visualization using real returns data.
    
    Args:
        alphas: List of alpha strategy data
        alpha_results_path: Path to alpha_results.json containing actual returns data
    
    Returns:
        Dictionary with cumulative returns data for top strategies
    """
    # Load actual returns data
    try:
        with open(alpha_results_path, 'r') as f:
            alpha_results = json.load(f)
    except FileNotFoundError:
        print(f"❌ Warning: {alpha_results_path} not found, using simulated data")
        return generate_cumulative_returns_simulated(alphas)
    
    # Sort alphas by Sharpe ratio and take top 10
    top_alphas = sorted(alphas, key=lambda x: x['sharpe_ratio'], reverse=True)[:10]
    
    cumulative_data = {
        "dates": [],
        "strategies": {}
    }
    
    # Find the maximum length of returns data to create consistent date range
    max_days = 0
    valid_strategies = []
    
    for alpha in top_alphas:
        if alpha['name'] in alpha_results and 'returns_data' in alpha_results[alpha['name']]:
            returns_length = len(alpha_results[alpha['name']]['returns_data'])
            max_days = max(max_days, returns_length)
            valid_strategies.append(alpha)
    
    if max_days == 0:
        print("❌ Warning: No valid returns data found, using simulated data")
        return generate_cumulative_returns_simulated(alphas)
    
    # Use actual dates from the first valid strategy
    if valid_strategies and 'dates' in alpha_results[valid_strategies[0]['name']]:
        dates = alpha_results[valid_strategies[0]['name']]['dates'][:max_days]
    else:
        # Fallback to generating dates if not available
        start_date = datetime(2024, 1, 1)
        dates = []
        current_date = start_date
        for i in range(max_days):
            # Skip weekends (simplified)
            while current_date.weekday() >= 5:
                current_date += timedelta(days=1)
            dates.append(current_date.strftime("%Y-%m-%d"))
            current_date += timedelta(days=1)
    
    cumulative_data["dates"] = dates
    
    # Generate cumulative returns for each strategy using real data
    for alpha in valid_strategies:
        returns_data = alpha_results[alpha['name']]['returns_data']
        
        # Calculate cumulative returns (compound)
        cumulative_returns = [1.0]  # Starting value
        for daily_ret in returns_data:
            cumulative_returns.append(cumulative_returns[-1] * (1 + daily_ret))
        
        # Convert to percentage above initial value
        cumulative_percentages = [(val - 1) * 100 for val in cumulative_returns[1:]]
        
        # Pad with the final value if this strategy has fewer days than max_days
        while len(cumulative_percentages) < max_days:
            cumulative_percentages.append(cumulative_percentages[-1] if cumulative_percentages else 0)
        
        cumulative_data["strategies"][alpha['name']] = {
            "name": alpha['name'],
            "sharpe_ratio": alpha['sharpe_ratio'],
            "annualized_return": alpha['annualized_return'],
            "cumulative_returns": cumulative_percentages,
            "final_return": cumulative_percentages[-1] if cumulative_percentages else 0,
            "actual_data": True
        }
    
    print(f"✅ Using real returns data for {len(valid_strategies)} strategies over {max_days} days")
    return cumulative_data

def generate_cumulative_returns_simulated(alphas, num_days=252):
    """
    Fallback function to generate simulated cumulative returns data.
    """
    # Sort alphas by Sharpe ratio and take top 10
    top_alphas = sorted(alphas, key=lambda x: x['sharpe_ratio'], reverse=True)[:10]
    
    cumulative_data = {
        "dates": [],
        "strategies": {}
    }
    
    # Generate date range (trading days only)
    start_date = datetime(2024, 1, 1)
    dates = []
    current_date = start_date
    for i in range(num_days):
        # Skip weekends (simplified)
        while current_date.weekday() >= 5:
            current_date += timedelta(days=1)
        dates.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)
    
    cumulative_data["dates"] = dates
    
    # Generate cumulative returns for each strategy
    for alpha in top_alphas:
        daily_return = alpha['annualized_return'] / num_days
        volatility = alpha['volatility']
        
        # Generate random daily returns with given mean and volatility
        np.random.seed(hash(alpha['name']) % 2**32)  # Consistent seed for reproducibility
        daily_returns = np.random.normal(daily_return, volatility / np.sqrt(num_days), num_days)
        
        # Calculate cumulative returns (compound)
        cumulative_returns = [1.0]  # Starting value
        for daily_ret in daily_returns:
            cumulative_returns.append(cumulative_returns[-1] * (1 + daily_ret))
        
        # Convert to percentage above initial value
        cumulative_percentages = [(val - 1) * 100 for val in cumulative_returns[1:]]
        
        cumulative_data["strategies"][alpha['name']] = {
            "name": alpha['name'],
            "sharpe_ratio": alpha['sharpe_ratio'],
            "annualized_return": alpha['annualized_return'],
            "cumulative_returns": cumulative_percentages,
            "final_return": cumulative_percentages[-1] if cumulative_percentages else 0,
            "actual_data": False
        }
    
    print(f"✅ Using simulated returns data for {len(top_alphas)} strategies over {num_days} days")
    return cumulative_data

def generate_source_code():
    """Generate source code JSON using the dedicated script."""
    try:
        print("🔧 Generating source code data...")
        result = subprocess.run([sys.executable, 'generate_source_code.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Source code generation completed successfully")
            return True
        else:
            print(f"❌ Source code generation failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error running source code generation: {e}")
        return False

def main():
    # Paths
    project_root = Path(__file__).parent
    frontend_root = project_root / "vercel-frontend"
    frontend_data = frontend_root / "public" / "data"
    
    # Create data directory if it doesn't exist
    frontend_data.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Generating static data for Vercel frontend...")
    
    # Generate source code first
    generate_source_code()
    
    # Copy alpha leaderboard data
    leaderboard_source = project_root / "alpha_leaderboard.json"
    leaderboard_dest = frontend_data / "alpha_leaderboard.json"
    
    if leaderboard_source.exists():
        shutil.copy2(leaderboard_source, leaderboard_dest)
        print(f"✅ Copied alpha leaderboard: {leaderboard_dest}")
    else:
        print(f"❌ Warning: {leaderboard_source} not found")
    
    # Source code is now generated directly in the target location
    source_code_dest = frontend_data / "source_code.json"
    if source_code_dest.exists():
        print(f"✅ Source code data ready: {source_code_dest}")
    else:
        print(f"❌ Warning: source_code.json generation may have failed")
    
    # Generate enhanced metadata
    metadata = {
        "generated_at": "2024-01-01T00:00:00Z",
        "total_strategies": 0,
        "data_sources": [
            "alpha_leaderboard.json",
            "source_code.json"
        ],
        "performance_metrics": [
            "sharpe_ratio",
            "annualized_return", 
            "calmar_ratio",
            "max_drawdown",
            "volatility",
            "num_trades"
        ]
    }
    
    # Update metadata with actual counts if data exists
    if leaderboard_dest.exists():
        with open(leaderboard_dest, 'r') as f:
            data = json.load(f)
            metadata["total_strategies"] = len(data)
    
    # Write metadata
    metadata_dest = frontend_data / "metadata.json"
    with open(metadata_dest, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Generated metadata: {metadata_dest}")
    
    # Generate summary statistics
    if leaderboard_dest.exists():
        with open(leaderboard_dest, 'r') as f:
            alphas = json.load(f)
        
        if alphas:
            # Calculate summary stats
            sharpe_ratios = [a['sharpe_ratio'] for a in alphas]
            returns = [a['annualized_return'] for a in alphas]
            
            summary = {
                "total_strategies": len(alphas),
                "positive_strategies": len([r for r in returns if r > 0]),
                "avg_sharpe_ratio": sum(sharpe_ratios) / len(sharpe_ratios),
                "best_sharpe": max(sharpe_ratios),
                "worst_sharpe": min(sharpe_ratios),
                "avg_return": sum(returns) / len(returns),
                "best_return": max(returns),
                "worst_return": min(returns),
                "top_performers": [
                    {
                        "name": alpha["name"],
                        "sharpe_ratio": alpha["sharpe_ratio"],
                        "annualized_return": alpha["annualized_return"]
                    }
                    for alpha in alphas[:10]
                ]
            }
            
            summary_dest = frontend_data / "summary.json"
            with open(summary_dest, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"✅ Generated summary statistics: {summary_dest}")
            
            # Generate cumulative returns data using real returns
            alpha_results_path = project_root / "alpha_results.json"
            cumulative_returns = generate_cumulative_returns(alphas, alpha_results_path)
            cumulative_dest = frontend_data / "cumulative_returns.json"
            with open(cumulative_dest, 'w') as f:
                json.dump(cumulative_returns, f, indent=2)
            print(f"✅ Generated cumulative returns data: {cumulative_dest}")
    
    # Create a simple API-like structure for the frontend
    api_data = {}
    
    if leaderboard_dest.exists():
        with open(leaderboard_dest, 'r') as f:
            api_data['alphas'] = json.load(f)
    
    if source_code_dest.exists():
        with open(source_code_dest, 'r') as f:
            api_data['source_code'] = json.load(f)
    
    # Write combined API data
    api_dest = frontend_data / "api.json"
    with open(api_dest, 'w') as f:
        json.dump(api_data, f, indent=2)
    print(f"✅ Generated combined API data: {api_dest}")
    
    print(f"\n🎉 Frontend data generation complete!")
    print(f"📁 Data location: {frontend_data}")
    print(f"📊 Files generated:")
    for file in frontend_data.glob("*.json"):
        size_kb = file.stat().st_size / 1024
        print(f"   • {file.name} ({size_kb:.1f} KB)")
    
    print(f"\n🚀 Ready to deploy to Vercel!")
    print(f"💡 Run the following commands to deploy:")
    print(f"   cd {frontend_root}")
    print(f"   npm install")
    print(f"   npm run build")
    print(f"   vercel --prod")

if __name__ == "__main__":
    main()