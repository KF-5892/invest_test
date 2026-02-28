import pandas as pd
import numpy as np
from .base_alpha import BaseAlpha

class VolumeWeightedMomentum(BaseAlpha):
    """Volume-weighted momentum strategy using return volatility as volume proxy."""
    
    def __init__(self, momentum_period: int = 30, volume_period: int = 20):
        super().__init__(
            name=f"volume_weighted_momentum_{momentum_period}d_{volume_period}d",
            description=f"Volume-weighted momentum over {momentum_period}d with {volume_period}d volume"
        )
        self.momentum_period = momentum_period
        self.volume_period = volume_period
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < max(self.momentum_period, self.volume_period):
            return pd.Series(dtype=float)
        
        # Calculate momentum
        momentum = data.iloc[-self.momentum_period:].sum()
        
        # Use absolute returns as volume proxy
        volume_proxy = data.iloc[-self.volume_period:].abs().mean()
        
        # Weight momentum by volume proxy
        volume_weight = volume_proxy / volume_proxy.sum()
        
        return momentum * volume_weight

class CrossSectionalVolatilityRank(BaseAlpha):
    """Cross-sectional volatility ranking with mean reversion."""
    
    def __init__(self, vol_period: int = 20, rank_period: int = 60):
        super().__init__(
            name=f"cross_vol_rank_{vol_period}d_{rank_period}d",
            description=f"Cross-sectional volatility rank {vol_period}d over {rank_period}d"
        )
        self.vol_period = vol_period
        self.rank_period = rank_period
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.rank_period:
            return pd.Series(dtype=float)
        
        # Calculate current volatility
        current_vol = data.iloc[-self.vol_period:].std()
        
        # Calculate historical volatility distribution
        hist_data = data.iloc[-self.rank_period:]
        vol_series = hist_data.rolling(self.vol_period).std().dropna()
        
        # Cross-sectional rank of current volatility
        current_rank = pd.Series(index=current_vol.index, dtype=float)
        for asset in current_vol.index:
            asset_vols = vol_series[asset].dropna()
            if len(asset_vols) > 0:
                current_rank[asset] = (asset_vols <= current_vol[asset]).mean()
        
        # Mean revert high volatility assets
        return 1 - current_rank

class AdaptiveTimeDecay(BaseAlpha):
    """Adaptive time decay momentum with exponentially decaying weights."""
    
    def __init__(self, lookback_period: int = 30, decay_factor: float = 0.95):
        super().__init__(
            name=f"adaptive_time_decay_{lookback_period}d_{decay_factor}",
            description=f"Adaptive time decay over {lookback_period}d with factor {decay_factor}"
        )
        self.lookback_period = lookback_period
        self.decay_factor = decay_factor
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.lookback_period:
            return pd.Series(dtype=float)
        
        recent_data = data.iloc[-self.lookback_period:]
        
        # Create exponentially decaying weights (more weight on recent data)
        weights = np.array([self.decay_factor ** i for i in range(self.lookback_period)])
        weights = weights[::-1]  # Reverse so recent data gets higher weights
        weights = weights / weights.sum()
        
        # Calculate weighted momentum
        weighted_returns = recent_data.multiply(weights, axis=0)
        
        return weighted_returns.sum()

class MomentumBreakdownDetection(BaseAlpha):
    """Detect momentum breakdown using trend consistency."""
    
    def __init__(self, short_period: int = 10, long_period: int = 30, consistency_threshold: float = 0.6):
        super().__init__(
            name=f"momentum_breakdown_{short_period}d_{long_period}d_{consistency_threshold}",
            description=f"Momentum breakdown detection {short_period}d vs {long_period}d"
        )
        self.short_period = short_period
        self.long_period = long_period
        self.consistency_threshold = consistency_threshold
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.long_period:
            return pd.Series(dtype=float)
        
        # Calculate short and long term momentum
        short_momentum = data.iloc[-self.short_period:].sum()
        long_momentum = data.iloc[-self.long_period:].sum()
        
        # Calculate trend consistency for short period
        short_data = data.iloc[-self.short_period:]
        positive_days = (short_data > 0).sum()
        consistency = positive_days / self.short_period
        
        # Detect breakdown: long momentum positive but short momentum weak
        breakdown_signal = pd.Series(0.0, index=data.columns)
        
        # Strong long momentum but weak short consistency suggests breakdown
        strong_long = long_momentum > long_momentum.median()
        weak_consistency = consistency < self.consistency_threshold
        
        breakdown_signal[strong_long & weak_consistency] = -1.0
        breakdown_signal[~strong_long & (consistency > (1 - self.consistency_threshold))] = 1.0
        
        return breakdown_signal

class VolatilityAdjustedCarry(BaseAlpha):
    """Carry strategy adjusted for volatility risk."""
    
    def __init__(self, carry_period: int = 60, vol_period: int = 30, target_vol: float = 0.15):
        super().__init__(
            name=f"vol_adj_carry_{carry_period}d_{vol_period}d_{target_vol}",
            description=f"Volatility adjusted carry {carry_period}d target vol {target_vol}"
        )
        self.carry_period = carry_period
        self.vol_period = vol_period
        self.target_vol = target_vol
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < max(self.carry_period, self.vol_period):
            return pd.Series(dtype=float)
        
        # Calculate carry (average return)
        carry = data.iloc[-self.carry_period:].mean()
        
        # Calculate volatility
        volatility = data.iloc[-self.vol_period:].std() * np.sqrt(252)
        
        # Volatility adjustment factor
        vol_adj = self.target_vol / volatility.replace(0, np.nan)
        
        # Adjust carry for volatility
        return carry * vol_adj

class InterTemporalArbitrage(BaseAlpha):
    """Exploit timing differences between short and long-term signals."""
    
    def __init__(self, fast_period: int = 5, slow_period: int = 25, signal_period: int = 10):
        super().__init__(
            name=f"intertemporal_arb_{fast_period}d_{slow_period}d_{signal_period}d",
            description=f"Intertemporal arbitrage {fast_period}d vs {slow_period}d"
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.slow_period + self.signal_period:
            return pd.Series(dtype=float)
        
        # Calculate fast and slow signals
        fast_signal = data.iloc[-self.fast_period:].sum()
        slow_signal = data.iloc[-self.slow_period:-self.signal_period].sum()
        
        # Normalize by period length
        fast_signal = fast_signal / self.fast_period
        slow_signal = slow_signal / (self.slow_period - self.signal_period)
        
        # Arbitrage signal: when fast and slow disagree
        signal_diff = fast_signal - slow_signal
        
        # Cross-sectional ranking
        signal_rank = signal_diff.rank(pct=True)
        
        return 2 * signal_rank - 1  # Convert to -1 to 1 range

class MicroTrendReversal(BaseAlpha):
    """Short-term trend reversal with micro-level signals."""
    
    def __init__(self, micro_period: int = 3, trend_period: int = 15, threshold: float = 0.01):
        super().__init__(
            name=f"micro_trend_reversal_{micro_period}d_{trend_period}d_{threshold}",
            description=f"Micro trend reversal {micro_period}d in {trend_period}d"
        )
        self.micro_period = micro_period
        self.trend_period = trend_period
        self.threshold = threshold
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.trend_period:
            return pd.Series(dtype=float)
        
        # Calculate micro-trend (very recent)
        micro_trend = data.iloc[-self.micro_period:].sum()
        
        # Calculate medium-term trend
        medium_trend = data.iloc[-self.trend_period:-self.micro_period].sum()
        medium_trend = medium_trend / (self.trend_period - self.micro_period) * self.micro_period
        
        # Reversal signal when micro trend opposes medium trend significantly
        reversal_signal = pd.Series(0.0, index=data.columns)
        
        # Strong opposing signals
        strong_micro_up = micro_trend > self.threshold
        strong_micro_down = micro_trend < -self.threshold
        strong_medium_up = medium_trend > self.threshold
        strong_medium_down = medium_trend < -self.threshold
        
        # Reversal conditions
        reversal_signal[strong_micro_down & strong_medium_up] = 1.0  # Buy on micro dip in uptrend
        reversal_signal[strong_micro_up & strong_medium_down] = -1.0  # Sell on micro rally in downtrend
        
        return reversal_signal

class CorrelationMomentum(BaseAlpha):
    """Momentum strategy based on correlation changes."""
    
    def __init__(self, corr_period: int = 30, momentum_period: int = 20):
        super().__init__(
            name=f"correlation_momentum_{corr_period}d_{momentum_period}d",
            description=f"Correlation momentum {corr_period}d correlation {momentum_period}d momentum"
        )
        self.corr_period = corr_period
        self.momentum_period = momentum_period
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < max(self.corr_period, self.momentum_period) * 2:
            return pd.Series(dtype=float)
        
        # Calculate current and past correlation with market
        market_return = data.mean(axis=1)
        
        current_data = data.iloc[-self.corr_period:]
        current_market = market_return.iloc[-self.corr_period:]
        
        past_data = data.iloc[-self.corr_period*2:-self.corr_period]
        past_market = market_return.iloc[-self.corr_period*2:-self.corr_period]
        
        current_corr = current_data.corrwith(current_market)
        past_corr = past_data.corrwith(past_market)
        
        # Correlation change
        corr_change = current_corr - past_corr
        
        # Calculate momentum
        momentum = data.iloc[-self.momentum_period:].sum()
        
        # Signal: momentum in direction of decreasing correlation
        return momentum * (-corr_change)

class RiskAdjustedMeanReversion(BaseAlpha):
    """Mean reversion with dynamic risk adjustment."""
    
    def __init__(self, reversion_period: int = 20, risk_period: int = 60, risk_factor: float = 2.0):
        super().__init__(
            name=f"risk_adj_mean_rev_{reversion_period}d_{risk_period}d_{risk_factor}",
            description=f"Risk adjusted mean reversion {reversion_period}d risk {risk_factor}x"
        )
        self.reversion_period = reversion_period
        self.risk_period = risk_period
        self.risk_factor = risk_factor
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < max(self.reversion_period, self.risk_period):
            return pd.Series(dtype=float)
        
        # Calculate recent performance vs historical mean
        recent_returns = data.iloc[-self.reversion_period:].sum()
        historical_mean = data.iloc[-self.risk_period:].mean() * self.reversion_period
        
        # Mean reversion signal
        reversion_signal = historical_mean - recent_returns
        
        # Risk adjustment using historical volatility
        risk_measure = data.iloc[-self.risk_period:].std() * np.sqrt(252)
        
        # Adjust signal strength by inverse of risk
        risk_adj_factor = 1 / (1 + risk_measure * self.risk_factor)
        
        return reversion_signal * risk_adj_factor

class VolatilitySignalFiltering(BaseAlpha):
    """Filter momentum signals based on volatility regimes."""
    
    def __init__(self, signal_period: int = 20, vol_period: int = 60, vol_threshold: float = 0.02):
        super().__init__(
            name=f"vol_signal_filter_{signal_period}d_{vol_period}d_{vol_threshold}",
            description=f"Volatility filtered signals {signal_period}d vol threshold {vol_threshold}"
        )
        self.signal_period = signal_period
        self.vol_period = vol_period
        self.vol_threshold = vol_threshold
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < max(self.signal_period, self.vol_period):
            return pd.Series(dtype=float)
        
        # Base momentum signal
        momentum_signal = data.iloc[-self.signal_period:].sum()
        
        # Current volatility regime
        current_vol = data.iloc[-self.vol_period:].std()
        historical_vol = data.iloc[-lookback_window:].std() if len(data) >= lookback_window else current_vol
        
        # Volatility regime indicator
        vol_regime = current_vol / historical_vol
        
        # Filter signals based on volatility regime
        filtered_signal = momentum_signal.copy()
        
        # In high volatility regimes, reduce signal strength
        high_vol_mask = current_vol > self.vol_threshold
        filtered_signal[high_vol_mask] = filtered_signal[high_vol_mask] * 0.5
        
        # In extremely high volatility, reverse signals (contrarian)
        extreme_vol_mask = vol_regime > 2.0
        filtered_signal[extreme_vol_mask] = -filtered_signal[extreme_vol_mask] * 0.3
        
        return filtered_signal