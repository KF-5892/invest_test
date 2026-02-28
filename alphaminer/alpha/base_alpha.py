import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class BaseAlpha(ABC):
    """Base class for alpha strategies."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.results = {}
        
    @abstractmethod
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        """
        Calculate alpha scores for each stock at current time.
        
        Args:
            data: DataFrame with date index and stock columns containing returns
            lookback_window: Number of days to look back for calculations
            
        Returns:
            Series with stock scores for current date
        """
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data format."""
        if data.empty:
            return False
        if not isinstance(data.index[0], (int, str)):
            return False
        return True
    
    def get_info(self) -> Dict[str, Any]:
        """Get alpha information."""
        return {
            'name': self.name,
            'description': self.description,
            'results': self.results
        }

class AlphaEngine:
    """Engine to run and evaluate alpha strategies."""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = None
        self.alphas = {}
        self.results = {}
        
    def load_data(self):
        """Load the Fama-French 48 industry portfolio data."""
        df = pd.read_csv(self.data_path)
        
        # Convert date column to datetime
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0].astype(str), format='%Y%m%d')
        df.set_index(df.columns[0], inplace=True)
        
        # Convert percentage returns to decimal
        df = df.astype(float) / 100
        
        # Remove any completely empty columns
        df = df.dropna(axis=1, how='all')
        
        self.data = df
        print(f"Loaded data: {df.shape[0]} days, {df.shape[1]} industries")
        return df
    
    def register_alpha(self, alpha: BaseAlpha):
        """Register an alpha strategy."""
        self.alphas[alpha.name] = alpha
        
    def run_alpha(self, alpha_name: str, start_date: str = None, end_date: str = None):
        """Run a specific alpha strategy."""
        if alpha_name not in self.alphas:
            raise ValueError(f"Alpha {alpha_name} not registered")
            
        if self.data is None:
            self.load_data()
            
        alpha = self.alphas[alpha_name]
        
        # Filter data by date range if specified
        data = self.data.copy()
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
            
        # Run alpha on historical data
        signals = []
        returns = []
        
        # Minimum lookback period
        min_lookback = 60
        
        for i in range(min_lookback, len(data)):
            current_date = data.index[i]
            historical_data = data.iloc[:i+1]
            
            try:
                # Get alpha signal
                signal = alpha.calculate(historical_data)
                
                # Get next day returns
                if i < len(data) - 1:
                    next_returns = data.iloc[i+1]
                    
                    # Calculate portfolio return (equal weighted)
                    valid_signals = signal.dropna()
                    if len(valid_signals) > 0:
                        # Normalize signals to weights
                        weights = valid_signals / valid_signals.abs().sum()
                        portfolio_return = (weights * next_returns[weights.index]).sum()
                        
                        signals.append({
                            'date': current_date,
                            'signal': signal,
                            'return': portfolio_return
                        })
                        returns.append(portfolio_return)
                        
            except Exception as e:
                print(f"Error calculating alpha on {current_date}: {e}")
                continue
        
        # Store results
        self.results[alpha_name] = {
            'signals': signals,
            'returns': pd.Series(returns, index=[s['date'] for s in signals]),
            'cumulative_returns': np.cumprod(1 + np.array(returns)) - 1
        }
        
        return self.results[alpha_name]
    
    def calculate_performance_metrics(self, alpha_name: str) -> Dict[str, float]:
        """Calculate performance metrics for an alpha."""
        if alpha_name not in self.results:
            raise ValueError(f"No results for alpha {alpha_name}")
            
        returns = self.results[alpha_name]['returns']
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown for Calmar ratio
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'num_trades': len(returns)
        }