import pandas as pd
import numpy as np
from .base_alpha import BaseAlpha

class BollingerBandsAlpha(BaseAlpha):
    """Bollinger Bands mean reversion alpha."""
    
    def __init__(self, lookback_days: int = 20, std_multiplier: float = 2.0):
        super().__init__(
            name=f"bollinger_bands_{lookback_days}d_std{std_multiplier}",
            description=f"Bollinger Bands mean reversion with {lookback_days}-day period"
        )
        self.lookback_days = lookback_days
        self.std_multiplier = std_multiplier
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.lookback_days:
            return pd.Series(dtype=float)
        
        recent_data = data.iloc[-self.lookback_days:]
        
        # Calculate Bollinger Bands for each stock
        signals = {}
        for col in recent_data.columns:
            prices = recent_data[col].cumsum()  # Convert returns to price levels
            
            # Calculate bands
            mean = prices.rolling(self.lookback_days).mean().iloc[-1]
            std = prices.rolling(self.lookback_days).std().iloc[-1]
            upper_band = mean + (self.std_multiplier * std)
            lower_band = mean - (self.std_multiplier * std)
            current_price = prices.iloc[-1]
            
            # Generate signal based on band position
            if current_price < lower_band:
                signals[col] = 1.0  # Oversold - buy
            elif current_price > upper_band:
                signals[col] = -1.0  # Overbought - sell
            else:
                signals[col] = 0.0
        
        return pd.Series(signals)

class AdaptiveVolatilityAlpha(BaseAlpha):
    """Adaptive volatility targeting with momentum filter."""
    
    def __init__(self, vol_window: int = 20, momentum_window: int = 5):
        super().__init__(
            name=f"adaptive_vol_{vol_window}d_mom{momentum_window}d",
            description=f"Adaptive volatility with momentum filter"
        )
        self.vol_window = vol_window
        self.momentum_window = momentum_window
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < max(self.vol_window, self.momentum_window):
            return pd.Series(dtype=float)
        
        # Calculate current volatility
        vol = data.iloc[-self.vol_window:].std()
        
        # Calculate short-term momentum
        momentum = data.iloc[-self.momentum_window:].sum()
        
        # Adaptive target based on market regime
        market_vol = vol.median()
        if market_vol > 0.02:  # High vol regime
            target_vol = 0.015
        else:  # Low vol regime
            target_vol = 0.025
        
        # Score based on volatility distance and momentum direction
        vol_score = 1 / (1 + abs(vol - target_vol))
        momentum_score = np.sign(momentum) * 0.5
        
        return vol_score + momentum_score

class TrendStrengthAlpha(BaseAlpha):
    """Trend strength based on moving average convergence."""
    
    def __init__(self, fast_ma: int = 5, slow_ma: int = 20):
        super().__init__(
            name=f"trend_strength_{fast_ma}d_{slow_ma}d",
            description=f"Trend strength using {fast_ma}/{slow_ma} MA convergence"
        )
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.slow_ma:
            return pd.Series(dtype=float)
        
        # Calculate moving averages of cumulative returns
        prices = data.cumsum()
        fast_ma = prices.rolling(self.fast_ma).mean()
        slow_ma = prices.rolling(self.slow_ma).mean()
        
        # Current MA difference (trend strength)
        ma_diff = (fast_ma.iloc[-1] - slow_ma.iloc[-1]) / slow_ma.iloc[-1]
        
        # Add volatility adjustment
        vol = data.iloc[-self.slow_ma:].std()
        vol_adj = 1 / (1 + vol)
        
        return ma_diff * vol_adj

class MeanReversionScore(BaseAlpha):
    """Combined mean reversion score using multiple indicators."""
    
    def __init__(self, lookback_days: int = 15):
        super().__init__(
            name=f"mean_reversion_score_{lookback_days}d",
            description=f"Combined mean reversion score over {lookback_days} days"
        )
        self.lookback_days = lookback_days
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.lookback_days:
            return pd.Series(dtype=float)
        
        recent_data = data.iloc[-self.lookback_days:]
        
        # Component 1: Short-term reversal
        short_returns = recent_data.iloc[-3:].sum()
        reversal_score = -short_returns
        
        # Component 2: Distance from moving average
        ma = recent_data.mean()
        current = recent_data.iloc[-1]
        ma_distance = -(current - ma)
        
        # Component 3: Volatility-adjusted score
        vol = recent_data.std()
        vol_adj = 1 / (1 + vol)
        
        # Combine scores
        combined_score = (reversal_score + ma_distance) * vol_adj
        return combined_score

class VolatilityBreakoutAlpha(BaseAlpha):
    """Volatility breakout strategy."""
    
    def __init__(self, vol_lookback: int = 20, breakout_threshold: float = 1.5):
        super().__init__(
            name=f"vol_breakout_{vol_lookback}d_thresh{breakout_threshold}",
            description=f"Volatility breakout with {vol_lookback}-day lookback"
        )
        self.vol_lookback = vol_lookback
        self.breakout_threshold = breakout_threshold
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.vol_lookback + 1:
            return pd.Series(dtype=float)
        
        # Calculate rolling volatility
        vol = data.rolling(self.vol_lookback).std()
        
        # Current vs average volatility
        current_vol = vol.iloc[-1]
        avg_vol = vol.iloc[-self.vol_lookback:-1].mean()
        
        # Detect breakouts
        vol_ratio = current_vol / avg_vol
        
        # Recent returns direction
        recent_returns = data.iloc[-3:].sum()
        
        # Signal: volatility breakout in direction of recent momentum
        breakout_signal = np.where(vol_ratio > self.breakout_threshold, 1.0, 0.0)
        direction = np.sign(recent_returns)
        
        return breakout_signal * direction

class RelativeStrengthAlpha(BaseAlpha):
    """Relative strength vs sector performance."""
    
    def __init__(self, lookback_days: int = 30):
        super().__init__(
            name=f"relative_strength_{lookback_days}d",
            description=f"Relative strength vs universe over {lookback_days} days"
        )
        self.lookback_days = lookback_days
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.lookback_days:
            return pd.Series(dtype=float)
        
        # Calculate returns for each stock
        returns = data.iloc[-self.lookback_days:].sum()
        
        # Calculate percentile ranks
        ranks = returns.rank(pct=True)
        
        # Favor extreme performers but with mean reversion bias
        # Strong performers get negative signal (expect reversal)
        # Weak performers get positive signal (expect recovery)
        relative_strength = 0.5 - ranks
        
        # Add volatility filter
        vol = data.iloc[-self.lookback_days:].std()
        vol_filter = np.where(vol < vol.quantile(0.7), 1.0, 0.5)
        
        return relative_strength * vol_filter

class MultiTimeframeRSI(BaseAlpha):
    """Multi-timeframe RSI convergence."""
    
    def __init__(self, short_period: int = 7, long_period: int = 21):
        super().__init__(
            name=f"multi_rsi_{short_period}d_{long_period}d",
            description=f"Multi-timeframe RSI using {short_period}/{long_period} periods"
        )
        self.short_period = short_period
        self.long_period = long_period
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.long_period + 10:
            return pd.Series(dtype=float)
        
        # Use vectorized operations for all columns at once
        recent_data = data.iloc[-(self.long_period + 15):]
        
        # Calculate deltas for all columns
        delta = recent_data.diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Vectorized RSI calculation for both periods
        short_avg_gains = gains.rolling(window=self.short_period).mean().iloc[-1]
        short_avg_losses = losses.rolling(window=self.short_period).mean().iloc[-1]
        long_avg_gains = gains.rolling(window=self.long_period).mean().iloc[-1]
        long_avg_losses = losses.rolling(window=self.long_period).mean().iloc[-1]
        
        # Calculate RSI values for all columns
        short_rs = short_avg_gains / short_avg_losses.replace(0, np.nan)
        short_rsi = 100 - (100 / (1 + short_rs))
        long_rs = long_avg_gains / long_avg_losses.replace(0, np.nan)
        long_rsi = 100 - (100 / (1 + long_rs))
        
        # Vectorized signal generation
        signals = pd.Series(0.0, index=recent_data.columns)
        
        # Strong oversold: both RSI conditions met
        strong_oversold = (short_rsi < 30) & (long_rsi < 40)
        signals[strong_oversold] = 1.0
        
        # Strong overbought: both RSI conditions met  
        strong_overbought = (short_rsi > 70) & (long_rsi > 60)
        signals[strong_overbought] = -1.0
        
        # Mild oversold: less restrictive conditions
        mild_oversold = (short_rsi < 40) & (long_rsi < 50) & ~strong_oversold
        signals[mild_oversold] = 0.5
        
        # Mild overbought: less restrictive conditions
        mild_overbought = (short_rsi > 60) & (long_rsi > 50) & ~strong_overbought
        signals[mild_overbought] = -0.5
        
        return signals

class VolatilityRegimeAlpha(BaseAlpha):
    """Volatility regime-dependent strategy."""
    
    def __init__(self, vol_window: int = 30, regime_threshold: float = 0.02):
        super().__init__(
            name=f"vol_regime_{vol_window}d_thresh{regime_threshold}",
            description=f"Volatility regime strategy with {vol_window}-day window"
        )
        self.vol_window = vol_window
        self.regime_threshold = regime_threshold
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.vol_window:
            return pd.Series(dtype=float)
        
        # Calculate current volatility regime
        vol = data.iloc[-self.vol_window:].std()
        market_vol = vol.median()
        
        # Determine regime
        high_vol_regime = market_vol > self.regime_threshold
        
        if high_vol_regime:
            # High vol: favor mean reversion
            recent_returns = data.iloc[-5:].sum()
            signals = -recent_returns  # Reversal
        else:
            # Low vol: favor momentum
            recent_returns = data.iloc[-10:].sum()
            signals = recent_returns  # Momentum
        
        # Apply volatility filter
        vol_score = 1 / (1 + vol)
        return signals * vol_score

class PairwiseMomentumAlpha(BaseAlpha):
    """Pairwise momentum strategy."""
    
    def __init__(self, lookback_days: int = 20, top_n: int = 5):
        super().__init__(
            name=f"pairwise_momentum_{lookback_days}d_top{top_n}",
            description=f"Pairwise momentum over {lookback_days} days"
        )
        self.lookback_days = lookback_days
        self.top_n = top_n
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.lookback_days:
            return pd.Series(dtype=float)
        
        # Calculate returns
        returns = data.iloc[-self.lookback_days:].sum()
        
        # Get top and bottom performers
        top_performers = returns.nlargest(self.top_n)
        bottom_performers = returns.nsmallest(self.top_n)
        
        # Create signals
        signals = pd.Series(0.0, index=returns.index)
        
        # Short top performers (expect reversal)
        signals[top_performers.index] = -0.5
        
        # Long bottom performers (expect recovery)
        signals[bottom_performers.index] = 1.0
        
        return signals

class VolatilityMeanReversionAlpha(BaseAlpha):
    """Pure volatility mean reversion strategy."""
    
    def __init__(self, vol_window: int = 15, target_vol: float = 0.018):
        super().__init__(
            name=f"vol_mean_reversion_{vol_window}d_target{target_vol}",
            description=f"Volatility mean reversion with {vol_window}-day window"
        )
        self.vol_window = vol_window
        self.target_vol = target_vol
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.vol_window:
            return pd.Series(dtype=float)
        
        # Calculate current volatility
        current_vol = data.iloc[-self.vol_window:].std()
        
        # Calculate long-term average volatility
        if len(data) >= 60:
            long_term_vol = data.iloc[-60:].std()
        else:
            long_term_vol = current_vol
        
        # Mean reversion signal: favor stocks with vol closer to long-term average
        vol_distance_from_target = abs(current_vol - self.target_vol)
        vol_distance_from_mean = abs(current_vol - long_term_vol)
        
        # Combine signals
        target_score = 1 / (1 + vol_distance_from_target)
        mean_reversion_score = 1 / (1 + vol_distance_from_mean)
        
        # Weight the combination
        combined_score = 0.6 * target_score + 0.4 * mean_reversion_score
        
        return combined_score