import pandas as pd
import numpy as np
from .base_alpha import BaseAlpha

class ZScoreAlpha(BaseAlpha):
    """Z-score based mean reversion alpha."""
    
    def __init__(self, lookback_days: int = 60, zscore_window: int = 20):
        super().__init__(
            name=f"zscore_{lookback_days}d_window{zscore_window}d",
            description=f"Z-score mean reversion with {lookback_days}d lookback"
        )
        self.lookback_days = lookback_days
        self.zscore_window = zscore_window
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.lookback_days:
            return pd.Series(dtype=float)
        
        # Get recent data
        recent_data = data.iloc[-self.lookback_days:]
        
        # Calculate rolling z-score for each stock
        rolling_mean = recent_data.rolling(self.zscore_window).mean()
        rolling_std = recent_data.rolling(self.zscore_window).std()
        
        # Current z-score (last value)
        current_price = recent_data.iloc[-1]
        current_mean = rolling_mean.iloc[-1]
        current_std = rolling_std.iloc[-1]
        
        # Avoid division by zero
        current_std = current_std.replace(0, np.nan)
        
        zscore = (current_price - current_mean) / current_std
        
        # Return negative z-score for mean reversion
        return -zscore

class RankAlpha(BaseAlpha):
    """Rank-based alpha strategy."""
    
    def __init__(self, lookback_days: int = 30):
        super().__init__(
            name=f"rank_alpha_{lookback_days}d",
            description=f"Rank-based strategy over {lookback_days} days"
        )
        self.lookback_days = lookback_days
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.lookback_days:
            return pd.Series(dtype=float)
        
        # Calculate returns
        returns = data.iloc[-self.lookback_days:].sum()
        
        # Convert to ranks (0 to 1)
        ranks = returns.rank(pct=True)
        
        # Center around 0
        return ranks - 0.5

class CorrelationAlpha(BaseAlpha):
    """Alpha based on correlation with market."""
    
    def __init__(self, lookback_days: int = 60):
        super().__init__(
            name=f"correlation_alpha_{lookback_days}d",
            description=f"Low correlation with market alpha over {lookback_days} days"
        )
        self.lookback_days = lookback_days
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.lookback_days:
            return pd.Series(dtype=float)
        
        # Get recent data
        recent_data = data.iloc[-self.lookback_days:]
        
        # Calculate market proxy (equal-weighted average)
        market_returns = recent_data.mean(axis=1)
        
        # Calculate correlation with market for each stock
        correlations = recent_data.corrwith(market_returns)
        
        # Return negative correlation (favor low correlation stocks)
        return -correlations

class VolatilityAlpha(BaseAlpha):
    """Volatility-based alpha strategy."""
    
    def __init__(self, lookback_days: int = 30, target_vol: float = 0.02):
        super().__init__(
            name=f"volatility_alpha_{lookback_days}d_target{target_vol:.3f}",
            description=f"Target volatility strategy over {lookback_days} days"
        )
        self.lookback_days = lookback_days
        self.target_vol = target_vol
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.lookback_days:
            return pd.Series(dtype=float)
        
        # Calculate volatility
        volatility = data.iloc[-self.lookback_days:].std()
        
        # Favor stocks with volatility close to target
        vol_distance = abs(volatility - self.target_vol)
        
        # Return inverse of distance (closer to target = higher score)
        return 1 / (1 + vol_distance)

class RSIAlpha(BaseAlpha):
    """RSI-based mean reversion alpha."""
    
    def __init__(self, rsi_period: int = 14):
        super().__init__(
            name=f"rsi_alpha_{rsi_period}d",
            description=f"RSI mean reversion with {rsi_period}-day period"
        )
        self.rsi_period = rsi_period
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.rsi_period + 1:
            return pd.Series(dtype=float)
        
        # Get recent data
        recent_data = data.iloc[-(self.rsi_period + 10):]  # Extra buffer
        
        # Vectorized calculation for all columns at once
        delta = recent_data.diff()
        
        # Separate gains and losses using vectorized operations
        gains = delta.where(delta > 0, 0)
        losses = (-delta).where(delta < 0, 0)
        
        # Calculate average gains and losses using vectorized rolling
        avg_gains = gains.rolling(window=self.rsi_period, min_periods=self.rsi_period).mean()
        avg_losses = losses.rolling(window=self.rsi_period, min_periods=self.rsi_period).mean()
        
        # Calculate RSI vectorized
        rs = avg_gains / avg_losses.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        
        # Get current RSI (last row)
        current_rsi = rsi.iloc[-1]
        
        # Mean reversion: buy oversold (RSI < 30), sell overbought (RSI > 70)
        signals = pd.Series(0.0, index=current_rsi.index)
        signals[current_rsi < 30] = 1.0    # Oversold - buy signal
        signals[current_rsi > 70] = -1.0   # Overbought - sell signal
        
        return signals