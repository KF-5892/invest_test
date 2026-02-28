import pandas as pd
import numpy as np
from .base_alpha import BaseAlpha

class ShortTermReversal(BaseAlpha):
    """Short-term mean reversion alpha."""
    
    def __init__(self, lookback_days: int = 5):
        super().__init__(
            name=f"short_term_reversal_{lookback_days}d",
            description=f"Short-term reversal based on {lookback_days}-day returns"
        )
        self.lookback_days = lookback_days
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.lookback_days:
            return pd.Series(dtype=float)
        
        # Calculate short-term returns
        recent_returns = data.iloc[-self.lookback_days:].sum()
        
        # Return negative of recent returns (reversal)
        return -recent_returns

class MomentumAlpha(BaseAlpha):
    """Medium-term momentum alpha."""
    
    def __init__(self, lookback_days: int = 60, skip_days: int = 5):
        super().__init__(
            name=f"momentum_{lookback_days}d_skip{skip_days}d",
            description=f"Momentum based on {lookback_days}-day returns, skipping last {skip_days} days"
        )
        self.lookback_days = lookback_days
        self.skip_days = skip_days
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.lookback_days + self.skip_days:
            return pd.Series(dtype=float)
        
        # Calculate momentum excluding recent days
        end_idx = len(data) - self.skip_days
        start_idx = end_idx - self.lookback_days
        
        momentum_returns = data.iloc[start_idx:end_idx].sum()
        return momentum_returns

class VolatilityAdjustedMomentum(BaseAlpha):
    """Momentum adjusted by volatility."""
    
    def __init__(self, lookback_days: int = 60, vol_window: int = 30):
        super().__init__(
            name=f"vol_adj_momentum_{lookback_days}d_vol{vol_window}d",
            description=f"Volatility-adjusted momentum over {lookback_days} days"
        )
        self.lookback_days = lookback_days
        self.vol_window = vol_window
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < max(self.lookback_days, self.vol_window):
            return pd.Series(dtype=float)
        
        # Calculate returns and volatility
        returns = data.iloc[-self.lookback_days:].sum()
        volatility = data.iloc[-self.vol_window:].std()
        
        # Avoid division by zero
        volatility = volatility.replace(0, np.nan)
        
        return returns / volatility

class CrossSectionalMomentum(BaseAlpha):
    """Cross-sectional momentum (relative to universe)."""
    
    def __init__(self, lookback_days: int = 30):
        super().__init__(
            name=f"cross_sectional_momentum_{lookback_days}d",
            description=f"Cross-sectional momentum over {lookback_days} days"
        )
        self.lookback_days = lookback_days
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.lookback_days:
            return pd.Series(dtype=float)
        
        # Calculate returns
        returns = data.iloc[-self.lookback_days:].sum()
        
        # Cross-sectional: subtract median
        median_return = returns.median()
        return returns - median_return