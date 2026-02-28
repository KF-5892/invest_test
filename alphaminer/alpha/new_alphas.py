import pandas as pd
import numpy as np
from .base_alpha import BaseAlpha

class QuantileRankReversal(BaseAlpha):
    """Quantile rank-based mean reversion with adaptive threshold."""
    
    def __init__(self, lookback: int = 30, quantile_threshold: float = 0.8):
        super().__init__(
            name=f"quantile_rank_reversal_{lookback}d_{quantile_threshold}",
            description=f"Quantile rank reversal {lookback}d threshold {quantile_threshold}"
        )
        self.lookback = lookback
        self.quantile_threshold = quantile_threshold
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.lookback:
            return pd.Series(dtype=float)
        
        recent_data = data.iloc[-self.lookback:]
        cum_returns = recent_data.sum()
        
        # Calculate rolling quantile ranks
        ranks = cum_returns.rank(pct=True)
        
        # Strong reversal signals for extreme quantiles
        signals = pd.Series(0.0, index=ranks.index)
        signals[ranks > self.quantile_threshold] = -2.0  # Top quantile -> strong sell
        signals[ranks < (1 - self.quantile_threshold)] = 2.0  # Bottom quantile -> strong buy
        signals[(ranks > 0.6) & (ranks <= self.quantile_threshold)] = -0.5  # Mild sell
        signals[(ranks < 0.4) & (ranks >= (1 - self.quantile_threshold))] = 0.5  # Mild buy
        
        return signals

class VolatilityAdjustedSkewness(BaseAlpha):
    """Skewness-based strategy adjusted for volatility regime."""
    
    def __init__(self, skew_window: int = 20, vol_window: int = 30):
        super().__init__(
            name=f"vol_adj_skewness_{skew_window}d_{vol_window}d",
            description=f"Volatility-adjusted skewness {skew_window}d/{vol_window}d"
        )
        self.skew_window = skew_window
        self.vol_window = vol_window
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < max(self.skew_window, self.vol_window):
            return pd.Series(dtype=float)
        
        # Calculate skewness
        skewness = data.iloc[-self.skew_window:].skew()
        
        # Calculate volatility regime
        current_vol = data.iloc[-self.vol_window:].std()
        long_vol = data.iloc[-60:].std() if len(data) >= 60 else current_vol
        vol_regime = current_vol / long_vol
        
        # Adjust skewness signal based on vol regime
        # In high vol: favor negative skew more (crash protection)
        # In low vol: use skew more directly
        vol_adjustment = np.where(vol_regime > 1.2, 1.5, 1.0)
        
        return -skewness * vol_adjustment

class MomentumVolatilityDecoupling(BaseAlpha):
    """Strategy based on momentum-volatility relationship breakdown."""
    
    def __init__(self, momentum_window: int = 15, vol_window: int = 20, lookback_corr: int = 60):
        super().__init__(
            name=f"momentum_vol_decoupling_{momentum_window}d_{vol_window}d_{lookback_corr}d",
            description=f"Momentum-volatility decoupling {momentum_window}/{vol_window}/{lookback_corr}d"
        )
        self.momentum_window = momentum_window
        self.vol_window = vol_window
        self.lookback_corr = lookback_corr
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.lookback_corr:
            return pd.Series(dtype=float)
        
        # Calculate momentum and volatility
        momentum = data.iloc[-self.momentum_window:].sum()
        volatility = data.iloc[-self.vol_window:].std()
        
        # Historical correlation between momentum and volatility
        hist_data = data.iloc[-self.lookback_corr:]
        hist_momentum = hist_data.rolling(self.momentum_window).sum()
        hist_volatility = hist_data.rolling(self.vol_window).std()
        
        correlations = {}
        for col in data.columns:
            mom_series = hist_momentum[col].dropna()
            vol_series = hist_volatility[col].dropna()
            
            # Align series to same length by taking common indices
            common_idx = mom_series.index.intersection(vol_series.index)
            if len(common_idx) > 1:
                mom_aligned = mom_series.loc[common_idx]
                vol_aligned = vol_series.loc[common_idx]
                corr = np.corrcoef(mom_aligned, vol_aligned)[0,1]
                correlations[col] = corr if not np.isnan(corr) else 0
            else:
                correlations[col] = 0
        
        corr_series = pd.Series(correlations)
        
        # When correlation breaks down (decoupling), trade the divergence
        expected_vol = corr_series * momentum
        vol_divergence = volatility - expected_vol.abs()
        
        # Trade against the divergence
        return -np.sign(vol_divergence) * momentum

class CrossAssetMomentumSpillover(BaseAlpha):
    """Cross-asset momentum spillover effects."""
    
    def __init__(self, spillover_window: int = 10, momentum_window: int = 20):
        super().__init__(
            name=f"cross_momentum_spillover_{spillover_window}d_{momentum_window}d",
            description=f"Cross-asset momentum spillover {spillover_window}/{momentum_window}d"
        )
        self.spillover_window = spillover_window
        self.momentum_window = momentum_window
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.momentum_window:
            return pd.Series(dtype=float)
        
        # Calculate recent momentum for all assets
        momentum = data.iloc[-self.momentum_window:].sum()
        
        # Calculate spillover effects (momentum of related assets)
        market_momentum = momentum.mean()
        
        # For each asset, calculate how much its momentum differs from market
        momentum_divergence = momentum - market_momentum
        
        # Spillover signal: when an asset's momentum is decoupled from market
        spillover_signal = np.where(abs(momentum_divergence) > momentum.std(), 
                                  -np.sign(momentum_divergence), 0)
        
        return pd.Series(spillover_signal, index=data.columns)

class AdaptiveMeanReversionSpeed(BaseAlpha):
    """Adaptive mean reversion based on historical reversion speed."""
    
    def __init__(self, reversion_window: int = 15, adaptation_window: int = 60):
        super().__init__(
            name=f"adaptive_mean_reversion_{reversion_window}d_{adaptation_window}d",
            description=f"Adaptive mean reversion {reversion_window}/{adaptation_window}d"
        )
        self.reversion_window = reversion_window
        self.adaptation_window = adaptation_window
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.adaptation_window:
            return pd.Series(dtype=float)
        
        signals = {}
        
        for col in data.columns:
            series = data[col].iloc[-self.adaptation_window:]
            
            # Calculate historical mean reversion speed
            # using autocorrelation at different lags
            autocorr_1 = series.autocorr(lag=1)
            autocorr_5 = series.autocorr(lag=5)
            autocorr_10 = series.autocorr(lag=10)
            
            # Mean reversion speed indicator
            reversion_speed = 1 - (abs(autocorr_1) + abs(autocorr_5) + abs(autocorr_10)) / 3
            
            # Current deviation from mean
            recent_returns = series.iloc[-self.reversion_window:].sum()
            mean_return = series.mean() * self.reversion_window
            deviation = recent_returns - mean_return
            
            # Signal strength adapted to reversion speed
            signals[col] = -deviation * reversion_speed
        
        return pd.Series(signals)

class VolatilityMeanReversionWithSkew(BaseAlpha):
    """Volatility mean reversion enhanced with skewness filter."""
    
    def __init__(self, vol_window: int = 20, skew_window: int = 30):
        super().__init__(
            name=f"vol_mean_revert_skew_{vol_window}d_{skew_window}d",
            description=f"Vol mean reversion with skew {vol_window}/{skew_window}d"
        )
        self.vol_window = vol_window
        self.skew_window = skew_window
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < max(self.vol_window, self.skew_window):
            return pd.Series(dtype=float)
        
        # Current vs long-term volatility
        current_vol = data.iloc[-self.vol_window:].std()
        long_vol = data.iloc[-60:].std() if len(data) >= 60 else current_vol
        vol_ratio = current_vol / long_vol
        
        # Skewness filter
        skewness = data.iloc[-self.skew_window:].skew()
        
        # Mean reversion signal for high vol, but only if skew is reasonable
        vol_signal = 1 / vol_ratio  # Favor lower vol
        skew_filter = np.where(skewness < -1.0, 0.5, 1.0)  # Reduce signal if very negative skew
        
        return vol_signal * skew_filter

class HighFrequencyReversal(BaseAlpha):
    """High-frequency style reversal using daily data patterns."""
    
    def __init__(self, lookback: int = 5, threshold: float = 0.01):
        super().__init__(
            name=f"hf_reversal_{lookback}d_{threshold}",
            description=f"High-frequency reversal {lookback}d threshold {threshold}"
        )
        self.lookback = lookback
        self.threshold = threshold
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.lookback + 1:
            return pd.Series(dtype=float)
        
        # Pattern: consecutive moves in same direction
        recent_data = data.iloc[-self.lookback:]
        
        # Count consecutive positive/negative days
        consecutive_pos = (recent_data > 0).astype(int)
        consecutive_neg = (recent_data < 0).astype(int)
        
        # Cumulative consecutive counts
        pos_streaks = consecutive_pos.sum()
        neg_streaks = consecutive_neg.sum()
        
        # Recent move magnitude
        recent_move = recent_data.sum()
        
        # Reversal signal for strong consecutive moves
        reversal_strength = np.where(
            (abs(recent_move) > self.threshold) & 
            ((pos_streaks >= self.lookback - 1) | (neg_streaks >= self.lookback - 1)),
            -np.sign(recent_move), 0
        )
        
        return pd.Series(reversal_strength, index=data.columns)

class CorrelationBreakdownAlpha(BaseAlpha):
    """Strategy based on correlation structure breakdown."""
    
    def __init__(self, corr_window: int = 30, lookback_window: int = 60):
        super().__init__(
            name=f"correlation_breakdown_{corr_window}d_{lookback_window}d",
            description=f"Correlation breakdown {corr_window}/{lookback_window}d"
        )
        self.corr_window = corr_window
        self.lookback_window = lookback_window
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.lookback_window:
            return pd.Series(dtype=float)
        
        # Current correlation matrix
        current_data = data.iloc[-self.corr_window:]
        current_corr = current_data.corr()
        
        # Historical correlation matrix
        hist_data = data.iloc[-self.lookback_window:-self.corr_window]
        hist_corr = hist_data.corr()
        
        # Correlation breakdown score
        signals = {}
        market_return = current_data.mean(axis=1).sum()
        
        for col in data.columns:
            # How much has this asset's correlation with others changed?
            current_avg_corr = current_corr[col].drop(col).mean()
            hist_avg_corr = hist_corr[col].drop(col).mean()
            corr_change = abs(current_avg_corr - hist_avg_corr)
            
            # When correlation breaks down, fade recent performance
            recent_perf = current_data[col].sum()
            signals[col] = -recent_perf * corr_change
        
        return pd.Series(signals)

class MacroRegimeMomentum(BaseAlpha):
    """Momentum strategy that adapts to macro regimes."""
    
    def __init__(self, regime_window: int = 60, momentum_window: int = 20):
        super().__init__(
            name=f"macro_regime_momentum_{regime_window}d_{momentum_window}d",
            description=f"Macro regime momentum {regime_window}/{momentum_window}d"
        )
        self.regime_window = regime_window
        self.momentum_window = momentum_window
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.regime_window:
            return pd.Series(dtype=float)
        
        # Market regime indicators
        market_returns = data.mean(axis=1)
        market_vol = market_returns.iloc[-self.regime_window:].std()
        market_trend = market_returns.iloc[-self.regime_window:].sum()
        
        # Cross-sectional dispersion
        daily_std = data.iloc[-self.regime_window:].std(axis=1)
        dispersion = daily_std.mean()
        
        # Determine regime
        high_vol_regime = market_vol > data.mean(axis=1).std()
        trending_regime = abs(market_trend) > market_vol * 2
        high_dispersion = dispersion > daily_std.std()
        
        # Momentum calculation
        momentum = data.iloc[-self.momentum_window:].sum()
        
        # Regime-based adjustments
        if high_vol_regime and trending_regime:
            # Strong trend + high vol: momentum with vol adjustment
            signal = momentum / data.iloc[-self.momentum_window:].std()
        elif high_dispersion:
            # High dispersion: cross-sectional momentum
            signal = momentum - momentum.median()
        else:
            # Normal regime: standard momentum
            signal = momentum
        
        return signal

class TailRiskParityAlpha(BaseAlpha):
    """Risk parity approach focused on tail risk."""
    
    def __init__(self, tail_window: int = 60, percentile: float = 0.05):
        super().__init__(
            name=f"tail_risk_parity_{tail_window}d_{percentile}",
            description=f"Tail risk parity {tail_window}d percentile {percentile}"
        )
        self.tail_window = tail_window
        self.percentile = percentile
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.tail_window:
            return pd.Series(dtype=float)
        
        recent_data = data.iloc[-self.tail_window:]
        
        # Calculate Value at Risk (VaR) and Expected Shortfall (ES)
        var_values = recent_data.quantile(self.percentile)
        
        # Expected Shortfall (average of returns below VaR)
        expected_shortfall = {}
        for col in recent_data.columns:
            tail_returns = recent_data[col][recent_data[col] <= var_values[col]]
            if len(tail_returns) > 0:
                expected_shortfall[col] = tail_returns.mean()
            else:
                expected_shortfall[col] = var_values[col]
        
        es_series = pd.Series(expected_shortfall)
        
        # Risk parity weights (inverse of tail risk)
        risk_weights = 1 / abs(es_series)
        risk_weights = risk_weights / risk_weights.sum()
        
        # Recent performance
        recent_returns = recent_data.iloc[-5:].sum()
        
        # Combine risk parity with performance
        return risk_weights * np.sign(recent_returns)

class FractionalDifferencing(BaseAlpha):
    """Fractional differencing for memory-aware signals."""
    
    def __init__(self, d_param: float = 0.4, lookback: int = 30):
        super().__init__(
            name=f"fractional_diff_d{d_param}_{lookback}d",
            description=f"Fractional differencing d={d_param} {lookback}d"
        )
        self.d_param = d_param
        self.lookback = lookback
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.lookback + 10:
            return pd.Series(dtype=float)
        
        signals = {}
        
        for col in data.columns:
            series = data[col].iloc[-self.lookback-10:]
            
            # Simple fractional differencing approximation
            # Using binomial coefficients for fractional differencing
            weights = [1]
            for k in range(1, min(len(series), 20)):
                weight = -weights[-1] * (self.d_param - k + 1) / k
                weights.append(weight)
            
            # Apply fractional differencing
            frac_diff = 0
            for i, w in enumerate(weights):
                if i < len(series):
                    frac_diff += w * series.iloc[-(i+1)]
            
            signals[col] = frac_diff
        
        return pd.Series(signals)

class VolatilityRiskPremium(BaseAlpha):
    """Volatility risk premium strategy."""
    
    def __init__(self, short_vol: int = 10, long_vol: int = 30, premium_window: int = 60):
        super().__init__(
            name=f"vol_risk_premium_{short_vol}d_{long_vol}d_{premium_window}d",
            description=f"Volatility risk premium {short_vol}/{long_vol}/{premium_window}d"
        )
        self.short_vol = short_vol
        self.long_vol = long_vol
        self.premium_window = premium_window
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.premium_window:
            return pd.Series(dtype=float)
        
        # Realized volatilities
        short_realized_vol = data.iloc[-self.short_vol:].std()
        long_realized_vol = data.iloc[-self.long_vol:].std()
        
        # Historical volatility premium
        hist_data = data.iloc[-self.premium_window:]
        hist_short_vol = hist_data.rolling(self.short_vol).std()
        hist_long_vol = hist_data.rolling(self.long_vol).std()
        
        # Average volatility premium
        vol_premium = (hist_long_vol - hist_short_vol).mean()
        
        # Current vs expected premium
        current_premium = long_realized_vol - short_realized_vol
        premium_surprise = current_premium - vol_premium
        
        # Trade against premium surprises
        return -premium_surprise

class DynamicBetaAlpha(BaseAlpha):
    """Dynamic beta strategy with regime awareness."""
    
    def __init__(self, beta_window: int = 30, regime_window: int = 60):
        super().__init__(
            name=f"dynamic_beta_{beta_window}d_{regime_window}d",
            description=f"Dynamic beta {beta_window}/{regime_window}d"
        )
        self.beta_window = beta_window
        self.regime_window = regime_window
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.regime_window:
            return pd.Series(dtype=float)
        
        # Market proxy
        market = data.mean(axis=1)
        
        # Calculate rolling betas
        betas = {}
        for col in data.columns:
            recent_stock = data[col].iloc[-self.beta_window:]
            recent_market = market.iloc[-self.beta_window:]
            
            # Beta calculation
            covariance = np.cov(recent_stock, recent_market)[0,1]
            market_var = np.var(recent_market)
            beta = covariance / market_var if market_var > 0 else 1
            betas[col] = beta
        
        beta_series = pd.Series(betas)
        
        # Market regime
        market_trend = market.iloc[-self.regime_window:].sum()
        market_vol = market.iloc[-self.regime_window:].std()
        
        # Beta strategy based on regime
        if market_trend > 0 and market_vol < market.std():
            # Bull market low vol: favor high beta
            signal = beta_series - 1
        else:
            # Bear/high vol market: favor low beta
            signal = 1 - beta_series
        
        return signal

class MomentumQualityFilter(BaseAlpha):
    """Momentum with quality filters."""
    
    def __init__(self, momentum_window: int = 30, quality_window: int = 60):
        super().__init__(
            name=f"momentum_quality_{momentum_window}d_{quality_window}d",
            description=f"Momentum with quality filter {momentum_window}/{quality_window}d"
        )
        self.momentum_window = momentum_window
        self.quality_window = quality_window
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.quality_window:
            return pd.Series(dtype=float)
        
        # Basic momentum
        momentum = data.iloc[-self.momentum_window:].sum()
        
        # Quality metrics
        stability = 1 / (data.iloc[-self.quality_window:].std() + 1e-6)
        consistency = (data.iloc[-self.quality_window:] > 0).sum() / self.quality_window
        
        # Downside volatility
        downside_returns = data.iloc[-self.quality_window:].copy()
        downside_returns[downside_returns > 0] = 0
        downside_vol = downside_returns.std()
        downside_protection = 1 / (downside_vol + 1e-6)
        
        # Combined quality score
        quality_score = (stability + consistency + downside_protection) / 3
        
        # Quality-adjusted momentum
        return momentum * quality_score

class MultiHorizonVolatility(BaseAlpha):
    """Multi-horizon volatility term structure strategy."""
    
    def __init__(self, horizons: list = [5, 10, 20, 40]):
        super().__init__(
            name=f"multi_horizon_vol_{'_'.join(map(str, horizons))}d",
            description=f"Multi-horizon volatility {horizons}"
        )
        self.horizons = horizons
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < max(self.horizons) + 10:
            return pd.Series(dtype=float)
        
        # Calculate volatilities at different horizons
        vols = {}
        for horizon in self.horizons:
            vols[horizon] = data.iloc[-horizon:].std()
        
        # Volatility term structure slope
        vol_slopes = []
        for i in range(len(self.horizons) - 1):
            short_vol = vols[self.horizons[i]]
            long_vol = vols[self.horizons[i+1]]
            slope = (long_vol - short_vol) / short_vol
            vol_slopes.append(slope)
        
        # Average slope
        avg_slope = sum(vol_slopes) / len(vol_slopes)
        
        # Favor negative slopes (backwardation)
        return -avg_slope

class EventDrivenReversal(BaseAlpha):
    """Event-driven reversal strategy."""
    
    def __init__(self, event_threshold: float = 2.5, lookback: int = 20):
        super().__init__(
            name=f"event_driven_reversal_{event_threshold}_{lookback}d",
            description=f"Event-driven reversal threshold {event_threshold} {lookback}d"
        )
        self.event_threshold = event_threshold
        self.lookback = lookback
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.lookback + 5:
            return pd.Series(dtype=float)
        
        # Detect events (large moves)
        recent_vol = data.rolling(self.lookback).std()
        current_return = data.iloc[-1]
        normalized_return = current_return / recent_vol.iloc[-1]
        
        # Event detection
        event_detected = abs(normalized_return) > self.event_threshold
        
        # Reversal signals after events
        reversal_signal = np.where(event_detected, -np.sign(current_return), 0)
        
        # Magnitude based on event size
        event_magnitude = abs(normalized_return) / self.event_threshold
        scaled_signal = reversal_signal * np.minimum(event_magnitude, 2.0)
        
        return pd.Series(scaled_signal, index=data.columns)

class AdaptiveVolatilityTargeting(BaseAlpha):
    """Adaptive volatility targeting with regime detection."""
    
    def __init__(self, target_vol: float = 0.015, adapt_window: int = 60):
        super().__init__(
            name=f"adaptive_vol_target_{target_vol}_{adapt_window}d",
            description=f"Adaptive vol targeting {target_vol} {adapt_window}d"
        )
        self.target_vol = target_vol
        self.adapt_window = adapt_window
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.adapt_window:
            return pd.Series(dtype=float)
        
        # Current volatility
        current_vol = data.iloc[-20:].std()
        
        # Regime-adjusted target
        market_vol = data.mean(axis=1).iloc[-self.adapt_window:].std()
        vol_regime_multiplier = np.clip(market_vol / 0.02, 0.5, 2.0)
        adjusted_target = self.target_vol * vol_regime_multiplier
        
        # Distance from target
        vol_distance = abs(current_vol - adjusted_target)
        
        # Momentum factor
        momentum = data.iloc[-10:].sum()
        
        # Signal: inverse vol distance weighted by momentum
        vol_score = 1 / (1 + vol_distance)
        
        return vol_score * np.sign(momentum)

class CrossSectionalMomentumDecay(BaseAlpha):
    """Cross-sectional momentum with decay analysis."""
    
    def __init__(self, momentum_window: int = 20, decay_window: int = 5):
        super().__init__(
            name=f"cs_momentum_decay_{momentum_window}d_{decay_window}d",
            description=f"Cross-sectional momentum decay {momentum_window}/{decay_window}d"
        )
        self.momentum_window = momentum_window
        self.decay_window = decay_window
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.momentum_window + self.decay_window:
            return pd.Series(dtype=float)
        
        # Historical momentum
        hist_momentum = data.iloc[-self.momentum_window-self.decay_window:-self.decay_window].sum()
        
        # Recent momentum
        recent_momentum = data.iloc[-self.decay_window:].sum()
        
        # Momentum decay rate
        momentum_decay = (hist_momentum - recent_momentum * (self.momentum_window / self.decay_window))
        
        # Cross-sectional ranks
        hist_ranks = hist_momentum.rank(pct=True)
        recent_ranks = recent_momentum.rank(pct=True)
        
        # Rank decay
        rank_decay = hist_ranks - recent_ranks
        
        # Combined signal: fade decaying momentum
        return rank_decay * np.sign(momentum_decay)

class HighMomentVolatilityTrap(BaseAlpha):
    """High momentum volatility trap strategy."""
    
    def __init__(self, momentum_threshold: float = 0.02, vol_threshold: float = 0.03):
        super().__init__(
            name=f"high_mom_vol_trap_{momentum_threshold}_{vol_threshold}",
            description=f"High momentum vol trap {momentum_threshold}/{vol_threshold}"
        )
        self.momentum_threshold = momentum_threshold
        self.vol_threshold = vol_threshold
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < 30:
            return pd.Series(dtype=float)
        
        # Recent momentum and volatility
        momentum = data.iloc[-10:].sum()
        volatility = data.iloc[-20:].std()
        
        # Identify high momentum + high volatility situations
        high_momentum = abs(momentum) > self.momentum_threshold
        high_volatility = volatility > self.vol_threshold
        
        # This combination often leads to reversals
        trap_condition = high_momentum & high_volatility
        
        # Reversal signal
        reversal_signal = np.where(trap_condition, -np.sign(momentum), 0)
        
        # Scale by volatility (higher vol = stronger signal)
        scaled_signal = reversal_signal * (volatility / self.vol_threshold)
        
        return pd.Series(scaled_signal, index=data.columns)

class MeanReversionWithMomentumFilter(BaseAlpha):
    """Mean reversion with momentum regime filter."""
    
    def __init__(self, reversion_window: int = 5, momentum_filter: int = 20):
        super().__init__(
            name=f"mean_revert_mom_filter_{reversion_window}d_{momentum_filter}d",
            description=f"Mean reversion with momentum filter {reversion_window}/{momentum_filter}d"
        )
        self.reversion_window = reversion_window
        self.momentum_filter = momentum_filter
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.momentum_filter:
            return pd.Series(dtype=float)
        
        # Mean reversion signal
        recent_returns = data.iloc[-self.reversion_window:].sum()
        reversion_signal = -recent_returns
        
        # Momentum filter
        momentum = data.iloc[-self.momentum_filter:].sum()
        momentum_strength = abs(momentum)
        
        # Apply reversion signal only in low momentum environments
        momentum_threshold = momentum.std()
        filter_strength = np.where(momentum_strength < momentum_threshold, 1.0, 0.3)
        
        return reversion_signal * filter_strength