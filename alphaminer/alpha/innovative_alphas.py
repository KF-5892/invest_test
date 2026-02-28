import pandas as pd
import numpy as np
from .base_alpha import BaseAlpha

class AsymmetricVolatilityAlpha(BaseAlpha):
    """Captures downside vs upside volatility differences."""
    
    def __init__(self, period: int = 30):
        super().__init__(
            name=f"asymmetric_vol_{period}d",
            description=f"Asymmetric volatility alpha over {period} days"
        )
        self.period = period
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.period:
            return pd.Series(dtype=float)
        
        recent_data = data.iloc[-self.period:]
        
        # Separate upside and downside returns
        upside = recent_data.where(recent_data > 0, 0)
        downside = recent_data.where(recent_data < 0, 0)
        
        # Calculate volatilities
        upside_vol = upside.std()
        downside_vol = downside.abs().std()
        
        # Asymmetry ratio (higher downside vol = negative signal)
        asymmetry = (downside_vol - upside_vol) / (downside_vol + upside_vol + 1e-8)
        
        return -asymmetry  # Favor assets with lower downside volatility

class MomentumPersistenceAlpha(BaseAlpha):
    """Measures how long momentum trends persist."""
    
    def __init__(self, short_period: int = 5, long_period: int = 20):
        super().__init__(
            name=f"momentum_persistence_{short_period}d_{long_period}d",
            description=f"Momentum persistence {short_period}d vs {long_period}d"
        )
        self.short_period = short_period
        self.long_period = long_period
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.long_period:
            return pd.Series(dtype=float)
        
        recent_data = data.iloc[-self.long_period:]
        
        # Calculate rolling momentum signals
        rolling_momentum = recent_data.rolling(self.short_period).sum()
        
        # Count consecutive periods with same sign momentum
        momentum_signs = np.sign(rolling_momentum)
        
        # Calculate persistence (consecutive same-sign periods)
        persistence_scores = pd.Series(0.0, index=data.columns)
        
        for col in data.columns:
            signs = momentum_signs[col].dropna()
            if len(signs) > 0:
                # Count current streak length
                current_streak = 1
                for i in range(len(signs)-2, -1, -1):
                    if signs.iloc[i] == signs.iloc[-1] and signs.iloc[-1] != 0:
                        current_streak += 1
                    else:
                        break
                
                persistence_scores[col] = current_streak * signs.iloc[-1]
        
        return persistence_scores

class CrossSectionalVolumeMomentum(BaseAlpha):
    """Volume-weighted momentum ranking."""
    
    def __init__(self, momentum_period: int = 20, volume_period: int = 10):
        super().__init__(
            name=f"cs_volume_momentum_{momentum_period}d_{volume_period}d",
            description=f"Cross-sectional volume momentum {momentum_period}d/{volume_period}d"
        )
        self.momentum_period = momentum_period
        self.volume_period = volume_period
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < max(self.momentum_period, self.volume_period):
            return pd.Series(dtype=float)
        
        # Calculate momentum
        momentum = data.iloc[-self.momentum_period:].sum()
        
        # Proxy volume with absolute returns
        volume_proxy = data.iloc[-self.volume_period:].abs().mean()
        
        # Volume-weighted momentum
        vol_weighted_momentum = momentum * volume_proxy
        
        # Cross-sectional ranking
        ranks = vol_weighted_momentum.rank(pct=True)
        
        return ranks - 0.5

class RegimeChangeDetectionAlpha(BaseAlpha):
    """Detects structural breaks in return patterns."""
    
    def __init__(self, lookback: int = 60, window: int = 20):
        super().__init__(
            name=f"regime_change_{lookback}d_{window}d",
            description=f"Regime change detection {lookback}d lookback, {window}d window"
        )
        self.lookback = lookback
        self.window = window
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.lookback:
            return pd.Series(dtype=float)
        
        recent_data = data.iloc[-self.lookback:]
        
        # Calculate rolling statistics
        rolling_mean = recent_data.rolling(self.window).mean()
        rolling_vol = recent_data.rolling(self.window).std()
        
        if len(rolling_mean) < 2:
            return pd.Series(0.0, index=data.columns)
        
        # Detect regime changes by comparing recent vs historical stats
        recent_mean = rolling_mean.iloc[-self.window//2:].mean()
        historical_mean = rolling_mean.iloc[:-self.window//2].mean()
        
        recent_vol = rolling_vol.iloc[-self.window//2:].mean()
        historical_vol = rolling_vol.iloc[:-self.window//2].mean()
        
        # Change magnitude
        mean_change = (recent_mean - historical_mean) / (historical_vol + 1e-8)
        vol_change = (recent_vol - historical_vol) / (historical_vol + 1e-8)
        
        # Combined regime change signal
        regime_signal = mean_change + 0.5 * vol_change
        
        return regime_signal

class OptionsInspiredSkewAlpha(BaseAlpha):
    """Mimics options market skew patterns."""
    
    def __init__(self, period: int = 30, quantile_low: float = 0.1, quantile_high: float = 0.9):
        super().__init__(
            name=f"options_skew_{period}d_{quantile_low}_{quantile_high}",
            description=f"Options-inspired skew alpha {period}d"
        )
        self.period = period
        self.quantile_low = quantile_low
        self.quantile_high = quantile_high
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.period:
            return pd.Series(dtype=float)
        
        recent_data = data.iloc[-self.period:]
        
        # Calculate tail quantiles (mimicking option strikes)
        low_quantile = recent_data.quantile(self.quantile_low)
        high_quantile = recent_data.quantile(self.quantile_high)
        median = recent_data.quantile(0.5)
        
        # Options-style skew: put/call volatility ratio
        left_tail_vol = (median - low_quantile).abs()
        right_tail_vol = (high_quantile - median).abs()
        
        # Skew signal (high left tail vol = fear = potential reversal)
        skew_ratio = left_tail_vol / (right_tail_vol + 1e-8)
        
        return np.log(skew_ratio + 1e-8)

class LiquidityStressAlpha(BaseAlpha):
    """Captures liquidity-driven reversals."""
    
    def __init__(self, stress_period: int = 10, recovery_period: int = 20):
        super().__init__(
            name=f"liquidity_stress_{stress_period}d_{recovery_period}d",
            description=f"Liquidity stress alpha {stress_period}d/{recovery_period}d"
        )
        self.stress_period = stress_period
        self.recovery_period = recovery_period
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.recovery_period:
            return pd.Series(dtype=float)
        
        # Proxy for liquidity stress: high volatility + negative returns
        recent_data = data.iloc[-self.recovery_period:]
        stress_data = recent_data.iloc[-self.stress_period:]
        
        # Stress indicators
        stress_vol = stress_data.std()
        stress_returns = stress_data.sum()
        
        # Combined stress score
        stress_score = stress_vol * np.maximum(-stress_returns, 0)  # Only negative returns
        
        # Liquidity recovery potential (mean reversion after stress)
        historical_vol = recent_data.iloc[:-self.stress_period].std()
        stress_excess = stress_vol - historical_vol
        
        # Recovery signal
        recovery_signal = stress_excess * np.sign(-stress_returns)
        
        return recovery_signal

class EarningsSurpriseMomentum(BaseAlpha):
    """Post-earnings drift patterns."""
    
    def __init__(self, surprise_period: int = 5, drift_period: int = 15):
        super().__init__(
            name=f"earnings_surprise_{surprise_period}d_{drift_period}d",
            description=f"Earnings surprise momentum {surprise_period}d/{drift_period}d"
        )
        self.surprise_period = surprise_period
        self.drift_period = drift_period
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.drift_period + self.surprise_period:
            return pd.Series(dtype=float)
        
        # Simulate earnings surprise with large single-day moves
        recent_data = data.iloc[-(self.drift_period + self.surprise_period):]
        
        # Identify "earnings days" (large absolute moves)
        daily_returns = recent_data
        vol_threshold = daily_returns.std() * 2  # 2 sigma moves
        
        # Find recent large moves (proxy for earnings)
        surprise_data = recent_data.iloc[-self.surprise_period:]
        large_moves = surprise_data.abs() > vol_threshold
        
        # Calculate surprise magnitude and direction
        surprise_magnitude = surprise_data.abs().max()
        surprise_direction = surprise_data.loc[surprise_data.abs().idxmax()]
        
        # Post-surprise drift expectation
        drift_signal = surprise_direction * np.sqrt(surprise_magnitude)
        
        return drift_signal

class MacroCycleAlpha(BaseAlpha):
    """Economic cycle-aware momentum."""
    
    def __init__(self, cycle_period: int = 63, momentum_period: int = 20):  # ~3 month cycle
        super().__init__(
            name=f"macro_cycle_{cycle_period}d_{momentum_period}d",
            description=f"Macro cycle alpha {cycle_period}d/{momentum_period}d"
        )
        self.cycle_period = cycle_period
        self.momentum_period = momentum_period
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.cycle_period:
            return pd.Series(dtype=float)
        
        # Proxy for macro cycle: cross-sectional dispersion
        recent_data = data.iloc[-self.cycle_period:]
        
        # Calculate rolling cross-sectional volatility (market stress proxy)
        cross_sec_vol = recent_data.std(axis=1).rolling(20).mean()
        
        # Macro regime: expansion (low dispersion) vs contraction (high dispersion)
        current_dispersion = cross_sec_vol.iloc[-1]
        historical_dispersion = cross_sec_vol.quantile(0.5)
        
        regime_signal = -1 if current_dispersion > historical_dispersion else 1
        
        # Momentum in different regimes
        momentum = data.iloc[-self.momentum_period:].sum()
        
        # Adjust momentum based on macro regime
        if regime_signal > 0:  # Expansion: momentum works
            return momentum
        else:  # Contraction: mean reversion works
            return -momentum

class CrossCorrelationBreakdownAlpha(BaseAlpha):
    """Inter-asset correlation shifts."""
    
    def __init__(self, short_window: int = 15, long_window: int = 60):
        super().__init__(
            name=f"corr_breakdown_{short_window}d_{long_window}d",
            description=f"Cross-correlation breakdown {short_window}d/{long_window}d"
        )
        self.short_window = short_window
        self.long_window = long_window
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.long_window:
            return pd.Series(dtype=float)
        
        # Calculate short and long-term correlation matrices
        recent_data = data.iloc[-self.long_window:]
        
        short_corr = recent_data.iloc[-self.short_window:].corr()
        long_corr = recent_data.corr()
        
        # Calculate correlation breakdown for each asset
        breakdown_scores = pd.Series(0.0, index=data.columns)
        
        for asset in data.columns:
            # Average correlation with other assets
            short_avg_corr = short_corr[asset].drop(asset).mean()
            long_avg_corr = long_corr[asset].drop(asset).mean()
            
            # Correlation breakdown (decorrelation)
            breakdown = long_avg_corr - short_avg_corr
            breakdown_scores[asset] = breakdown
        
        # Favor assets with correlation breakdown (potential for independent moves)
        return breakdown_scores

class VolatilitySurfaceTermStructure(BaseAlpha):
    """Term structure of implied volatility."""
    
    def __init__(self, short_term: int = 10, medium_term: int = 30, long_term: int = 60):
        super().__init__(
            name=f"vol_term_structure_{short_term}d_{medium_term}d_{long_term}d",
            description=f"Volatility term structure {short_term}d/{medium_term}d/{long_term}d"
        )
        self.short_term = short_term
        self.medium_term = medium_term
        self.long_term = long_term
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.long_term:
            return pd.Series(dtype=float)
        
        # Calculate volatilities at different horizons
        recent_data = data.iloc[-self.long_term:]
        
        short_vol = recent_data.iloc[-self.short_term:].std() * np.sqrt(252)
        medium_vol = recent_data.iloc[-self.medium_term:].std() * np.sqrt(252)
        long_vol = recent_data.std() * np.sqrt(252)
        
        # Term structure slopes
        short_medium_slope = (medium_vol - short_vol) / (self.medium_term - self.short_term)
        medium_long_slope = (long_vol - medium_vol) / (self.long_term - self.medium_term)
        
        # Term structure curvature
        curvature = medium_long_slope - short_medium_slope
        
        # Signal: negative curvature suggests volatility mean reversion
        return -curvature