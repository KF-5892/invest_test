import pandas as pd
import numpy as np
from .base_alpha import BaseAlpha

class VolatilitySurfaceAlpha(BaseAlpha):
    """Volatility surface arbitrage across different time horizons."""
    
    def __init__(self, short_vol: int = 10, long_vol: int = 30):
        super().__init__(
            name=f"vol_surface_{short_vol}d_{long_vol}d",
            description=f"Vol surface arbitrage {short_vol}d vs {long_vol}d"
        )
        self.short_vol = short_vol
        self.long_vol = long_vol
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.long_vol:
            return pd.Series(dtype=float)
        
        short_vol = data.iloc[-self.short_vol:].std()
        long_vol = data.iloc[-self.long_vol:].std()
        
        # Vol surface slope
        vol_slope = (short_vol - long_vol) / long_vol
        
        # Favor steep negative slopes (short vol < long vol)
        return -vol_slope

class FactorMomentumAlpha(BaseAlpha):
    """Multi-factor momentum with exponential decay."""
    
    def __init__(self, lookback: int = 60, decay_rate: float = 0.95):
        super().__init__(
            name=f"factor_momentum_{lookback}d_decay{decay_rate}",
            description=f"Factor momentum with decay rate {decay_rate}"
        )
        self.lookback = lookback
        self.decay_rate = decay_rate
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.lookback:
            return pd.Series(dtype=float)
        
        recent_data = data.iloc[-self.lookback:]
        
        # Calculate exponentially weighted momentum
        weights = np.array([self.decay_rate**i for i in range(self.lookback-1, -1, -1)])
        weights = weights / weights.sum()
        
        weighted_returns = (recent_data.values * weights.reshape(-1, 1)).sum(axis=0)
        return pd.Series(weighted_returns, index=data.columns)

class RegimeSwitchingAlpha(BaseAlpha):
    """Market regime detection with adaptive strategies."""
    
    def __init__(self, regime_window: int = 60, vol_threshold: float = 0.02):
        super().__init__(
            name=f"regime_switching_{regime_window}d_thresh{vol_threshold}",
            description=f"Regime switching with {regime_window}d window"
        )
        self.regime_window = regime_window
        self.vol_threshold = vol_threshold
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.regime_window:
            return pd.Series(dtype=float)
        
        recent_data = data.iloc[-self.regime_window:]
        market_vol = recent_data.std().median()
        
        if market_vol > self.vol_threshold:
            # High vol regime: mean reversion
            signal = -recent_data.iloc[-5:].sum()
        else:
            # Low vol regime: momentum
            signal = recent_data.iloc[-10:].sum()
        
        return signal

class InformationRatioAlpha(BaseAlpha):
    """Risk-adjusted momentum using information ratio."""
    
    def __init__(self, lookback: int = 30, benchmark_period: int = 60):
        super().__init__(
            name=f"info_ratio_{lookback}d_bench{benchmark_period}d",
            description=f"Information ratio alpha {lookback}d lookback"
        )
        self.lookback = lookback
        self.benchmark_period = benchmark_period
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.benchmark_period:
            return pd.Series(dtype=float)
        
        # Calculate excess returns vs benchmark
        returns = data.iloc[-self.lookback:].sum()
        benchmark_return = data.iloc[-self.benchmark_period:].mean(axis=1).sum()
        excess_returns = returns - benchmark_return
        
        # Information ratio = excess return / tracking error
        tracking_error = data.iloc[-self.lookback:].std()
        info_ratio = excess_returns / tracking_error.replace(0, np.nan)
        
        return info_ratio

class LeadLagAlpha(BaseAlpha):
    """Cross-asset lead-lag momentum relationships."""
    
    def __init__(self, lag_period: int = 3, momentum_period: int = 20):
        super().__init__(
            name=f"lead_lag_{lag_period}d_mom{momentum_period}d",
            description=f"Lead-lag alpha with {lag_period}d lag"
        )
        self.lag_period = lag_period
        self.momentum_period = momentum_period
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.momentum_period + self.lag_period:
            return pd.Series(dtype=float)
        
        # Calculate lagged correlations with market
        market = data.mean(axis=1)
        signals = {}
        
        for col in data.columns:
            # Correlation between current stock and lagged market
            stock_current = data[col].iloc[-self.momentum_period:]
            market_lagged = market.iloc[-self.momentum_period-self.lag_period:-self.lag_period]
            
            if len(stock_current) == len(market_lagged):
                corr = np.corrcoef(stock_current, market_lagged)[0,1]
                signals[col] = corr if not np.isnan(corr) else 0
            else:
                signals[col] = 0
        
        return pd.Series(signals)

class ConditionalVolatilityAlpha(BaseAlpha):
    """GARCH-style conditional volatility prediction."""
    
    def __init__(self, alpha: float = 0.1, beta: float = 0.8):
        super().__init__(
            name=f"conditional_vol_a{alpha}_b{beta}",
            description=f"Conditional volatility GARCH({alpha}, {beta})"
        )
        self.alpha_param = alpha
        self.beta = beta
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < 30:
            return pd.Series(dtype=float)
        
        signals = {}
        for col in data.columns:
            returns = data[col].iloc[-30:]
            
            # Simple GARCH(1,1) approximation
            var_long_run = returns.var()
            current_return_sq = returns.iloc[-1]**2
            prev_vol_sq = returns.iloc[-5:].var()
            
            # Conditional variance
            cond_var = (1 - self.alpha_param - self.beta) * var_long_run + \
                      self.alpha_param * current_return_sq + \
                      self.beta * prev_vol_sq
            
            # Signal: inverse of predicted volatility
            signals[col] = 1 / (1 + np.sqrt(cond_var))
        
        return pd.Series(signals)

class JumpDetectionAlpha(BaseAlpha):
    """Jump detection and trading strategy."""
    
    def __init__(self, jump_threshold: float = 3.0, lookback: int = 20):
        super().__init__(
            name=f"jump_detection_thresh{jump_threshold}_look{lookback}d",
            description=f"Jump detection with {jump_threshold} std threshold"
        )
        self.jump_threshold = jump_threshold
        self.lookback = lookback
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.lookback + 1:
            return pd.Series(dtype=float)
        
        # Calculate rolling statistics
        rolling_std = data.rolling(self.lookback).std()
        current_return = data.iloc[-1]
        current_std = rolling_std.iloc[-1]
        
        # Detect jumps
        jump_size = current_return / current_std
        
        # Mean revert after jumps
        jump_signal = np.where(abs(jump_size) > self.jump_threshold, 
                              -np.sign(jump_size), 0)
        
        return pd.Series(jump_signal, index=data.columns)

class MicrostructureAlpha(BaseAlpha):
    """Microstructure patterns using high-frequency concepts."""
    
    def __init__(self, window: int = 10):
        super().__init__(
            name=f"microstructure_{window}d",
            description=f"Microstructure patterns {window}d window"
        )
        self.window = window
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.window + 5:
            return pd.Series(dtype=float)
        
        recent_data = data.iloc[-self.window:]
        
        # Calculate bid-ask spread proxy (volatility)
        spread_proxy = recent_data.std()
        
        # Volume imbalance proxy (skewness)
        volume_imbalance = recent_data.skew()
        
        # Combine signals
        microstructure_signal = -spread_proxy + 0.5 * volume_imbalance
        
        return microstructure_signal

class TermStructureAlpha(BaseAlpha):
    """Volatility term structure arbitrage."""
    
    def __init__(self, short_term: int = 5, medium_term: int = 20, long_term: int = 60):
        super().__init__(
            name=f"term_structure_{short_term}d_{medium_term}d_{long_term}d",
            description=f"Vol term structure {short_term}/{medium_term}/{long_term}d"
        )
        self.short_term = short_term
        self.medium_term = medium_term
        self.long_term = long_term
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.long_term:
            return pd.Series(dtype=float)
        
        short_vol = data.iloc[-self.short_term:].std()
        medium_vol = data.iloc[-self.medium_term:].std()
        long_vol = data.iloc[-self.long_term:].std()
        
        # Term structure slope
        slope1 = (medium_vol - short_vol) / short_vol
        slope2 = (long_vol - medium_vol) / medium_vol
        
        # Favor inverted term structures
        term_signal = -(slope1 + slope2)
        
        return term_signal

class CarryAlpha(BaseAlpha):
    """Statistical carry trade opportunities."""
    
    def __init__(self, lookback: int = 60, carry_window: int = 10):
        super().__init__(
            name=f"carry_{lookback}d_window{carry_window}d",
            description=f"Carry alpha {lookback}d lookback"
        )
        self.lookback = lookback
        self.carry_window = carry_window
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.lookback:
            return pd.Series(dtype=float)
        
        # Calculate expected return (carry)
        expected_return = data.iloc[-self.lookback:].mean()
        
        # Calculate volatility (risk)
        volatility = data.iloc[-self.lookback:].std()
        
        # Recent momentum
        momentum = data.iloc[-self.carry_window:].sum()
        
        # Carry signal: expected return per unit of risk
        carry_ratio = expected_return / volatility.replace(0, np.nan)
        
        # Combine with momentum
        return carry_ratio + 0.3 * momentum

class OptionsFlowAlpha(BaseAlpha):
    """Implied options flow from return patterns."""
    
    def __init__(self, lookback: int = 20):
        super().__init__(
            name=f"options_flow_{lookback}d",
            description=f"Options flow alpha {lookback}d"
        )
        self.lookback = lookback
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.lookback + 5:
            return pd.Series(dtype=float)
        
        recent_data = data.iloc[-self.lookback:]
        
        # Gamma exposure proxy (convexity)
        returns_sq = recent_data**2
        convexity = returns_sq.rolling(5).mean().iloc[-1]
        
        # Delta exposure proxy (directional)
        directional = recent_data.iloc[-5:].sum()
        
        # Vega exposure proxy (vol sensitivity)
        vol_sensitivity = recent_data.std()
        
        # Combined options flow signal
        flow_signal = 0.4 * directional - 0.3 * convexity + 0.3 * vol_sensitivity
        
        return flow_signal

class CrossSectionalDispersionAlpha(BaseAlpha):
    """Cross-sectional dispersion trading strategy."""
    
    def __init__(self, lookback: int = 30):
        super().__init__(
            name=f"cs_dispersion_{lookback}d",
            description=f"Cross-sectional dispersion {lookback}d"
        )
        self.lookback = lookback
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.lookback:
            return pd.Series(dtype=float)
        
        recent_data = data.iloc[-self.lookback:]
        
        # Calculate cross-sectional dispersion
        daily_std = recent_data.std(axis=1)
        current_dispersion = daily_std.iloc[-1]
        avg_dispersion = daily_std.mean()
        
        # Dispersion mean reversion
        dispersion_ratio = current_dispersion / avg_dispersion
        
        # When dispersion is high, favor convergence (mean reversion)
        # When dispersion is low, favor divergence (momentum)
        if dispersion_ratio > 1.2:
            # High dispersion: mean revert
            returns = recent_data.iloc[-5:].sum()
            signal = -returns
        else:
            # Low dispersion: momentum
            returns = recent_data.iloc[-10:].sum()
            signal = returns
        
        return signal

class EconomicSurpriseAlpha(BaseAlpha):
    """Economic surprise reaction strategy."""
    
    def __init__(self, surprise_window: int = 5, reaction_window: int = 15):
        super().__init__(
            name=f"econ_surprise_{surprise_window}d_{reaction_window}d",
            description=f"Economic surprise reaction {surprise_window}/{reaction_window}d"
        )
        self.surprise_window = surprise_window
        self.reaction_window = reaction_window
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.reaction_window + self.surprise_window:
            return pd.Series(dtype=float)
        
        # Proxy for economic surprise: large market moves
        market_returns = data.mean(axis=1)
        surprise_threshold = market_returns.std() * 2
        
        recent_market = market_returns.iloc[-self.surprise_window:]
        large_moves = abs(recent_market) > surprise_threshold
        
        if large_moves.any():
            # Surprise detected: fade the move
            surprise_direction = np.sign(recent_market[large_moves].iloc[-1])
            individual_moves = data.iloc[-self.surprise_window:].sum()
            signal = -surprise_direction * individual_moves / abs(individual_moves).sum()
        else:
            # No surprise: neutral
            signal = pd.Series(0.0, index=data.columns)
        
        return signal

class SentimentReversalAlpha(BaseAlpha):
    """Contrarian sentiment strategy."""
    
    def __init__(self, sentiment_window: int = 20, extreme_threshold: float = 1.5):
        super().__init__(
            name=f"sentiment_reversal_{sentiment_window}d_thresh{extreme_threshold}",
            description=f"Sentiment reversal {sentiment_window}d window"
        )
        self.sentiment_window = sentiment_window
        self.extreme_threshold = extreme_threshold
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.sentiment_window + 10:
            return pd.Series(dtype=float)
        
        recent_data = data.iloc[-self.sentiment_window:]
        
        # Sentiment proxy: cumulative returns z-score
        cum_returns = recent_data.sum()
        long_term_mean = data.iloc[-60:].sum().mean() if len(data) >= 60 else cum_returns.mean()
        long_term_std = data.iloc[-60:].sum().std() if len(data) >= 60 else cum_returns.std()
        
        sentiment_zscore = (cum_returns - long_term_mean) / long_term_std
        
        # Contrarian signal for extreme sentiment
        signals = pd.Series(0.0, index=data.columns)
        signals[sentiment_zscore > self.extreme_threshold] = -1.0  # Fade euphoria
        signals[sentiment_zscore < -self.extreme_threshold] = 1.0  # Fade panic
        
        return signals

class MomentumCrashAlpha(BaseAlpha):
    """Momentum crash protection strategy."""
    
    def __init__(self, momentum_window: int = 60, crash_threshold: float = -0.1):
        super().__init__(
            name=f"momentum_crash_{momentum_window}d_thresh{crash_threshold}",
            description=f"Momentum crash protection {momentum_window}d"
        )
        self.momentum_window = momentum_window
        self.crash_threshold = crash_threshold
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.momentum_window + 10:
            return pd.Series(dtype=float)
        
        # Calculate momentum
        momentum = data.iloc[-self.momentum_window:].sum()
        
        # Detect crash conditions (large negative market move)
        market_return = data.mean(axis=1).iloc[-5:].sum()
        crash_condition = market_return < self.crash_threshold
        
        if crash_condition:
            # During crashes: reverse momentum signals
            signal = -momentum
        else:
            # Normal times: regular momentum with decay
            recent_vol = data.iloc[-20:].std()
            vol_adj = 1 / (1 + recent_vol)
            signal = momentum * vol_adj
        
        return signal

class VolatilityClusteringAlpha(BaseAlpha):
    """Volatility clustering exploitation."""
    
    def __init__(self, cluster_window: int = 10, vol_threshold: float = 2.0):
        super().__init__(
            name=f"vol_clustering_{cluster_window}d_thresh{vol_threshold}",
            description=f"Volatility clustering {cluster_window}d window"
        )
        self.cluster_window = cluster_window
        self.vol_threshold = vol_threshold
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.cluster_window + 20:
            return pd.Series(dtype=float)
        
        # Detect volatility clusters
        recent_vol = data.rolling(self.cluster_window).std()
        long_term_vol = data.rolling(60).std().iloc[-1] if len(data) >= 60 else recent_vol.iloc[-1]
        
        current_vol = recent_vol.iloc[-1]
        vol_ratio = current_vol / long_term_vol
        
        # High vol clustering: expect persistence
        # Low vol clustering: expect mean reversion
        high_vol_cluster = vol_ratio > self.vol_threshold
        
        recent_returns = data.iloc[-3:].sum()
        
        if high_vol_cluster.any():
            # High vol: trend following
            signal = np.sign(recent_returns)
        else:
            # Low vol: mean reversion
            signal = -recent_returns
        
        return signal

class AsymmetricRiskAlpha(BaseAlpha):
    """Asymmetric upside vs downside risk strategy."""
    
    def __init__(self, lookback: int = 30):
        super().__init__(
            name=f"asymmetric_risk_{lookback}d",
            description=f"Asymmetric risk alpha {lookback}d"
        )
        self.lookback = lookback
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.lookback:
            return pd.Series(dtype=float)
        
        recent_data = data.iloc[-self.lookback:]
        
        # Separate upside and downside moves
        upside_returns = recent_data[recent_data > 0]
        downside_returns = recent_data[recent_data < 0]
        
        # Calculate asymmetric volatilities
        upside_vol = upside_returns.std().fillna(0)
        downside_vol = downside_returns.abs().std().fillna(0)
        
        # Asymmetry ratio
        asymmetry = (upside_vol - downside_vol) / (upside_vol + downside_vol).replace(0, np.nan)
        
        # Favor assets with positive asymmetry (more upside vol)
        return asymmetry.fillna(0)

class LiquidityAlpha(BaseAlpha):
    """Liquidity-based mean reversion strategy."""
    
    def __init__(self, liquidity_window: int = 20):
        super().__init__(
            name=f"liquidity_{liquidity_window}d",
            description=f"Liquidity alpha {liquidity_window}d"
        )
        self.liquidity_window = liquidity_window
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.liquidity_window + 10:
            return pd.Series(dtype=float)
        
        recent_data = data.iloc[-self.liquidity_window:]
        
        # Liquidity proxy: inverse of return volatility
        liquidity = 1 / (recent_data.std() + 1e-6)
        
        # Recent price pressure
        price_pressure = recent_data.iloc[-5:].sum()
        
        # Liquidity-adjusted mean reversion
        # More liquid assets revert faster
        reversion_speed = liquidity / liquidity.median()
        signal = -price_pressure * reversion_speed
        
        return signal

class CrossFrequencyAlpha(BaseAlpha):
    """Multi-timeframe signal aggregation."""
    
    def __init__(self, short: int = 5, medium: int = 20, long: int = 60):
        super().__init__(
            name=f"cross_frequency_{short}d_{medium}d_{long}d",
            description=f"Cross-frequency {short}/{medium}/{long}d"
        )
        self.short = short
        self.medium = medium
        self.long = long
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.long:
            return pd.Series(dtype=float)
        
        # Signals from different timeframes
        short_signal = data.iloc[-self.short:].sum()
        medium_signal = data.iloc[-self.medium:].sum() / (self.medium / self.short)
        long_signal = data.iloc[-self.long:].sum() / (self.long / self.short)
        
        # Weighted combination
        combined_signal = (0.5 * short_signal + 
                          0.3 * medium_signal + 
                          0.2 * long_signal)
        
        return combined_signal

class AdaptiveSignalAlpha(BaseAlpha):
    """Adaptive signal using ensemble methods."""
    
    def __init__(self, adapt_window: int = 60):
        super().__init__(
            name=f"adaptive_signal_{adapt_window}d",
            description=f"Adaptive signal {adapt_window}d"
        )
        self.adapt_window = adapt_window
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.adapt_window + 20:
            return pd.Series(dtype=float)
        
        # Multiple base signals
        momentum = data.iloc[-20:].sum()
        mean_revert = -data.iloc[-5:].sum()
        vol_signal = 1 / (data.iloc[-15:].std() + 1e-6)
        
        # Recent market regime
        market_vol = data.std(axis=1).iloc[-self.adapt_window:].mean()
        
        # Adaptive weights based on market regime
        if market_vol > 0.02:
            # High vol: favor mean reversion
            weights = [0.2, 0.6, 0.2]
        else:
            # Low vol: favor momentum
            weights = [0.6, 0.2, 0.2]
        
        # Combine signals
        adaptive_signal = (weights[0] * momentum + 
                          weights[1] * mean_revert + 
                          weights[2] * vol_signal)
        
        return adaptive_signal