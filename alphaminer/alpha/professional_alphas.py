import pandas as pd
import numpy as np
from .base_alpha import BaseAlpha
from scipy import stats
from typing import List

class MarketStructureArbitrage(BaseAlpha):
    """Exploits temporary market structure inefficiencies."""
    
    def __init__(self, lookback: int = 20, threshold: float = 0.01):
        super().__init__(
            name=f"market_structure_arbitrage_{lookback}d_{threshold}",
            description=f"Market structure arbitrage over {lookback}d with {threshold} threshold"
        )
        self.lookback = lookback
        self.threshold = threshold
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.lookback:
            return pd.Series(dtype=float)
        
        recent_data = data.iloc[-self.lookback:]
        
        # Calculate market structure metrics
        # Intraday range proxy (high-low spread)
        daily_range = recent_data.abs()
        avg_range = daily_range.mean()
        current_range = daily_range.iloc[-1]
        
        # Detect compression (low range) followed by expansion
        range_ratio = current_range / avg_range
        compressed = range_ratio < (1 - self.threshold)
        
        # Volume proxy using absolute returns
        volume_proxy = recent_data.abs().rolling(5).mean().iloc[-1]
        avg_volume = recent_data.abs().mean()
        
        # Signal generation
        signals = pd.Series(0.0, index=data.columns)
        
        # Buy compressed assets with increasing volume
        buy_condition = compressed & (volume_proxy > avg_volume * 1.1)
        signals[buy_condition] = 1.0
        
        # Sell expanded assets
        expanded = range_ratio > (1 + self.threshold)
        signals[expanded] = -0.5
        
        return signals

class VolatilityClusteringPredictor(BaseAlpha):
    """Predicts future volatility based on clustering patterns."""
    
    def __init__(self, vol_window: int = 15, cluster_threshold: float = 1.5):
        super().__init__(
            name=f"volatility_clustering_predictor_{vol_window}d_{cluster_threshold}",
            description=f"Volatility clustering predictor {vol_window}d window"
        )
        self.vol_window = vol_window
        self.cluster_threshold = cluster_threshold
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.vol_window * 2:
            return pd.Series(dtype=float)
        
        # Calculate rolling volatility
        rolling_vol = data.rolling(self.vol_window).std()
        
        if len(rolling_vol) < self.vol_window:
            return pd.Series(dtype=float)
        
        recent_vol = rolling_vol.iloc[-self.vol_window:]
        
        # Detect volatility clustering
        vol_increases = (recent_vol.diff() > 0).sum()
        vol_decreases = (recent_vol.diff() < 0).sum()
        
        # Clustering score
        clustering_score = vol_increases / self.vol_window
        
        # Current volatility level
        current_vol = rolling_vol.iloc[-1]
        avg_vol = rolling_vol.iloc[-60:].mean() if len(rolling_vol) >= 60 else rolling_vol.mean()
        
        # Signal: fade high volatility clusters, buy low volatility
        vol_ratio = current_vol / avg_vol
        
        # Generate signals
        signals = pd.Series(0.0, index=data.columns)
        
        # High vol cluster -> fade
        high_vol_cluster = (vol_ratio > self.cluster_threshold) & (clustering_score > 0.6)
        signals[high_vol_cluster] = -1.0
        
        # Low vol after cluster -> buy
        low_vol_after_cluster = (vol_ratio < 0.8) & (clustering_score > 0.4)
        signals[low_vol_after_cluster] = 1.0
        
        return signals

class CrossSectionalMeanReversionSpeed(BaseAlpha):
    """Speed of mean reversion relative to peers."""
    
    def __init__(self, period: int = 30, decay_factor: float = 0.95):
        super().__init__(
            name=f"cross_sectional_mean_reversion_speed_{period}d_{decay_factor}",
            description=f"Cross-sectional mean reversion speed over {period}d"
        )
        self.period = period
        self.decay_factor = decay_factor
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.period * 2:
            return pd.Series(dtype=float)
        
        recent_data = data.iloc[-self.period:]
        
        # Calculate autocorrelation for mean reversion speed
        autocorr_series = recent_data.apply(lambda x: x.autocorr(lag=1) if len(x.dropna()) > 10 else 0)
        
        # Cross-sectional ranking of autocorrelation
        autocorr_rank = autocorr_series.rank(pct=True)
        
        # Distance from cross-sectional mean
        returns_sum = recent_data.sum()
        cross_sectional_mean = returns_sum.median()
        distance_from_mean = returns_sum - cross_sectional_mean
        
        # Mean reversion speed signal
        # Low autocorr = fast mean reversion
        speed_score = 1 - autocorr_series.abs()
        
        # Combine distance and speed
        signals = -distance_from_mean * speed_score
        
        # Normalize signals
        if signals.std() > 0:
            signals = (signals - signals.mean()) / signals.std()
        
        return signals.fillna(0)

class AdaptiveRiskParityAlpha(BaseAlpha):
    """Dynamic risk parity based on changing correlations."""
    
    def __init__(self, vol_window: int = 30, corr_window: int = 60):
        super().__init__(
            name=f"adaptive_risk_parity_{vol_window}d_{corr_window}d",
            description=f"Adaptive risk parity vol:{vol_window}d corr:{corr_window}d"
        )
        self.vol_window = vol_window
        self.corr_window = corr_window
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < max(self.vol_window, self.corr_window):
            return pd.Series(dtype=float)
        
        # Calculate recent volatilities
        recent_vol = data.iloc[-self.vol_window:].std()
        
        # Calculate correlation matrix
        corr_data = data.iloc[-self.corr_window:]
        corr_matrix = corr_data.corr()
        
        # Risk parity weights (inverse volatility)
        inv_vol = 1 / recent_vol.replace(0, np.nan)
        risk_parity_weights = inv_vol / inv_vol.sum()
        
        # Adjust for correlation (diversification benefit)
        # Higher correlation = lower weight
        avg_correlation = corr_matrix.mean().abs()
        diversification_factor = 1 - avg_correlation
        
        # Adaptive weights
        adaptive_weights = risk_parity_weights * diversification_factor
        adaptive_weights = adaptive_weights / adaptive_weights.sum()
        
        # Convert to signals (relative to equal weight)
        equal_weight = 1 / len(data.columns)
        signals = adaptive_weights - equal_weight
        
        return signals.fillna(0)

class MomentumQualityScore(BaseAlpha):
    """Quality-weighted momentum based on persistence."""
    
    def __init__(self, momentum_period: int = 60, quality_period: int = 20):
        super().__init__(
            name=f"momentum_quality_score_{momentum_period}d_{quality_period}d",
            description=f"Quality momentum over {momentum_period}d with {quality_period}d quality"
        )
        self.momentum_period = momentum_period
        self.quality_period = quality_period
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.momentum_period:
            return pd.Series(dtype=float)
        
        # Calculate momentum
        momentum = data.iloc[-self.momentum_period:].sum()
        
        # Calculate quality metrics
        recent_data = data.iloc[-self.quality_period:]
        
        # Quality 1: Consistency (% of positive days)
        positive_days = (recent_data > 0).sum()
        consistency = positive_days / self.quality_period
        
        # Quality 2: Trend stability (low volatility of daily returns)
        stability = 1 / (1 + recent_data.std())
        
        # Quality 3: No extreme outliers (robust momentum)
        q75 = recent_data.quantile(0.75)
        q25 = recent_data.quantile(0.25)
        iqr_momentum = recent_data[(recent_data >= q25) & (recent_data <= q75)].sum()
        
        # Combined quality score
        quality_score = (consistency + stability) / 2
        
        # Quality-weighted momentum
        quality_momentum = momentum * quality_score
        
        return quality_momentum.fillna(0)

class VolatilitySpilloverAlpha(BaseAlpha):
    """Cross-asset volatility spillover effects."""
    
    def __init__(self, vol_window: int = 20, spillover_lag: int = 3):
        super().__init__(
            name=f"volatility_spillover_{vol_window}d_lag{spillover_lag}",
            description=f"Volatility spillover {vol_window}d window with {spillover_lag}d lag"
        )
        self.vol_window = vol_window
        self.spillover_lag = spillover_lag
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.vol_window + self.spillover_lag:
            return pd.Series(dtype=float)
        
        # Calculate volatility
        vol_data = data.abs().rolling(self.vol_window).mean()
        
        if len(vol_data) < self.spillover_lag + 1:
            return pd.Series(dtype=float)
        
        # Current volatility
        current_vol = vol_data.iloc[-1]
        
        # Lagged market volatility (average of all assets)
        lagged_market_vol = vol_data.iloc[-(self.spillover_lag+1)].mean()
        
        # Spillover effect: assets with vol below market average may catch up
        vol_differential = current_vol - lagged_market_vol
        
        # Signal: buy assets with low current vol relative to lagged market vol
        signals = -vol_differential
        
        # Normalize
        if signals.std() > 0:
            signals = (signals - signals.mean()) / signals.std()
        
        return signals.fillna(0)

class RegimeAwareReversal(BaseAlpha):
    """Reversal strategy that adapts to market regimes."""
    
    def __init__(self, regime_window: int = 60, reversal_period: int = 5):
        super().__init__(
            name=f"regime_aware_reversal_{regime_window}d_{reversal_period}d",
            description=f"Regime-aware reversal with {regime_window}d regime, {reversal_period}d reversal"
        )
        self.regime_window = regime_window
        self.reversal_period = reversal_period
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.regime_window:
            return pd.Series(dtype=float)
        
        # Detect market regime
        regime_data = data.iloc[-self.regime_window:]
        market_returns = regime_data.mean(axis=1)
        
        # Regime classification: high/low volatility
        market_vol = market_returns.std()
        recent_vol = market_returns.iloc[-20:].std() if len(market_returns) >= 20 else market_vol
        
        high_vol_regime = recent_vol > market_vol * 1.2
        
        # Recent returns for reversal
        if len(data) < self.reversal_period:
            return pd.Series(dtype=float)
        
        recent_returns = data.iloc[-self.reversal_period:].sum()
        
        # Adaptive reversal based on regime
        if high_vol_regime:
            # Strong reversal in high vol regime
            reversal_strength = 1.5
        else:
            # Mild reversal in low vol regime
            reversal_strength = 0.8
        
        signals = -recent_returns * reversal_strength
        
        # Normalize
        if signals.std() > 0:
            signals = (signals - signals.mean()) / signals.std()
        
        return signals.fillna(0)

class LiquidityDrivenMomentum(BaseAlpha):
    """Momentum filtered by liquidity conditions."""
    
    def __init__(self, momentum_period: int = 30, liquidity_window: int = 15):
        super().__init__(
            name=f"liquidity_driven_momentum_{momentum_period}d_{liquidity_window}d",
            description=f"Liquidity-driven momentum {momentum_period}d with {liquidity_window}d liquidity"
        )
        self.momentum_period = momentum_period
        self.liquidity_window = liquidity_window
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < max(self.momentum_period, self.liquidity_window):
            return pd.Series(dtype=float)
        
        # Calculate momentum
        momentum = data.iloc[-self.momentum_period:].sum()
        
        # Liquidity proxy: inverse of volatility (high vol = low liquidity)
        recent_data = data.iloc[-self.liquidity_window:]
        liquidity_proxy = 1 / (recent_data.std() + 1e-8)
        
        # Relative liquidity
        liquidity_rank = liquidity_proxy.rank(pct=True)
        
        # Filter momentum by liquidity
        # Only trade momentum in liquid assets
        liquidity_threshold = 0.3  # Bottom 30% considered illiquid
        
        filtered_momentum = momentum.copy()
        filtered_momentum[liquidity_rank < liquidity_threshold] = 0
        
        # Boost momentum for highly liquid assets
        high_liquidity = liquidity_rank > 0.7
        filtered_momentum[high_liquidity] *= 1.2
        
        return filtered_momentum.fillna(0)

class MacroSentimentAlpha(BaseAlpha):
    """Economic sentiment-driven alpha."""
    
    def __init__(self, sentiment_window: int = 45, threshold: float = 0.02):
        super().__init__(
            name=f"macro_sentiment_{sentiment_window}d_{threshold}",
            description=f"Macro sentiment alpha over {sentiment_window}d"
        )
        self.sentiment_window = sentiment_window
        self.threshold = threshold
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.sentiment_window:
            return pd.Series(dtype=float)
        
        # Market sentiment proxy: average market performance
        market_sentiment = data.iloc[-self.sentiment_window:].mean(axis=1).mean()
        
        # Sector rotation based on sentiment
        recent_returns = data.iloc[-self.sentiment_window:].sum()
        
        # In positive sentiment: momentum
        # In negative sentiment: reversal
        if market_sentiment > self.threshold:
            # Positive sentiment - momentum strategy
            signals = recent_returns
        elif market_sentiment < -self.threshold:
            # Negative sentiment - reversal strategy
            signals = -recent_returns
        else:
            # Neutral sentiment - no signal
            signals = pd.Series(0.0, index=data.columns)
        
        # Normalize
        if signals.std() > 0:
            signals = (signals - signals.mean()) / signals.std()
        
        return signals.fillna(0)

class VolatilityTermStructureAlpha(BaseAlpha):
    """Exploits volatility term structure patterns."""
    
    def __init__(self, short_window: int = 10, medium_window: int = 30, long_window: int = 60):
        super().__init__(
            name=f"volatility_term_structure_{short_window}d_{medium_window}d_{long_window}d",
            description=f"Vol term structure {short_window}d/{medium_window}d/{long_window}d"
        )
        self.short_window = short_window
        self.medium_window = medium_window
        self.long_window = long_window
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.long_window:
            return pd.Series(dtype=float)
        
        # Calculate volatilities at different horizons
        short_vol = data.iloc[-self.short_window:].std()
        medium_vol = data.iloc[-self.medium_window:].std()
        long_vol = data.iloc[-self.long_window:].std()
        
        # Term structure slope
        short_medium_slope = (medium_vol - short_vol) / short_vol
        medium_long_slope = (long_vol - medium_vol) / medium_vol
        
        # Volatility term structure signals
        # Upward sloping: expect vol expansion -> fade momentum
        # Downward sloping: expect vol contraction -> follow momentum
        
        term_structure_slope = (short_medium_slope + medium_long_slope) / 2
        
        # Recent price momentum
        recent_momentum = data.iloc[-self.short_window:].sum()
        
        # Adjust momentum based on term structure
        # Steep upward slope -> fade momentum
        # Steep downward slope -> amplify momentum
        adjustment_factor = 1 - term_structure_slope
        
        signals = recent_momentum * adjustment_factor
        
        # Normalize
        if signals.std() > 0:
            signals = (signals - signals.mean()) / signals.std()
        
        return signals.fillna(0)