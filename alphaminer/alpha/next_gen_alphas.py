import pandas as pd
import numpy as np
from .base_alpha import BaseAlpha

class VolatilityRiskPremiumDecay(BaseAlpha):
    """Captures the time decay of volatility risk premium."""
    
    def __init__(self, short_window: int = 5, long_window: int = 20, decay_rate: float = 0.95):
        super().__init__(
            name=f"vol_risk_premium_decay_{short_window}d_{long_window}d_{decay_rate}",
            description=f"Volatility risk premium decay {short_window}/{long_window}d decay {decay_rate}"
        )
        self.short_window = short_window
        self.long_window = long_window
        self.decay_rate = decay_rate
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.long_window + 10:
            return pd.Series(dtype=float)
        
        # Short-term vs long-term implied volatility proxy
        short_realized_vol = data.iloc[-self.short_window:].std() * np.sqrt(252)
        long_realized_vol = data.iloc[-self.long_window:].std() * np.sqrt(252)
        
        # Historical volatility risk premium
        hist_data = data.iloc[-60:]
        hist_short_vol = hist_data.rolling(self.short_window).std() * np.sqrt(252)
        hist_long_vol = hist_data.rolling(self.long_window).std() * np.sqrt(252)
        vol_premium_series = hist_long_vol - hist_short_vol
        
        # Time decay weights
        time_weights = np.array([self.decay_rate**i for i in range(len(vol_premium_series))])
        time_weights = time_weights / time_weights.sum()
        
        # Weighted average of historical premiums
        expected_premium = (vol_premium_series * time_weights).sum()
        current_premium = long_realized_vol - short_realized_vol
        
        # Premium decay signal
        premium_decay = expected_premium - current_premium
        
        return premium_decay

class CrossAssetContagionAlpha(BaseAlpha):
    """Detects and trades contagion effects across assets."""
    
    def __init__(self, contagion_window: int = 10, threshold: float = 2.0):
        super().__init__(
            name=f"cross_asset_contagion_{contagion_window}d_{threshold}",
            description=f"Cross-asset contagion detection {contagion_window}d threshold {threshold}"
        )
        self.contagion_window = contagion_window
        self.threshold = threshold
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.contagion_window + 20:
            return pd.Series(dtype=float)
        
        recent_data = data.iloc[-self.contagion_window:]
        
        # Calculate z-scores for each asset
        long_term_mean = data.iloc[-60:].mean() if len(data) >= 60 else recent_data.mean()
        long_term_std = data.iloc[-60:].std() if len(data) >= 60 else recent_data.std()
        
        current_returns = recent_data.sum()
        z_scores = (current_returns - long_term_mean * self.contagion_window) / (long_term_std * np.sqrt(self.contagion_window))
        
        # Identify contagion source (extreme negative performer)
        contagion_source = z_scores[z_scores < -self.threshold]
        
        if len(contagion_source) > 0:
            # Calculate correlations with contagion source
            source_asset = contagion_source.idxmin()  # Most extreme
            source_returns = recent_data[source_asset]
            
            signals = {}
            for asset in data.columns:
                if asset != source_asset:
                    correlation = np.corrcoef(recent_data[asset], source_returns)[0,1]
                    # High correlation with contagion source suggests oversold opportunity
                    if correlation > 0.5 and z_scores[asset] < -1.0:
                        signals[asset] = 1.0  # Buy oversold correlated assets
                    elif correlation < -0.3:
                        signals[asset] = -0.5  # Short negatively correlated (defensive)
                    else:
                        signals[asset] = 0.0
                else:
                    signals[asset] = 0.5  # Contrarian on source
            
            return pd.Series(signals)
        else:
            return pd.Series(0.0, index=data.columns)

class MomentumReversalTiming(BaseAlpha):
    """Times momentum reversal points using multiple technical signals."""
    
    def __init__(self, momentum_window: int = 20, rsi_period: int = 14, bb_period: int = 20):
        super().__init__(
            name=f"momentum_reversal_timing_{momentum_window}d_{rsi_period}d_{bb_period}d",
            description=f"Momentum reversal timing {momentum_window}/{rsi_period}/{bb_period}d"
        )
        self.momentum_window = momentum_window
        self.rsi_period = rsi_period
        self.bb_period = bb_period
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < max(self.momentum_window, self.rsi_period + 10, self.bb_period):
            return pd.Series(dtype=float)
        
        signals = {}
        
        for col in data.columns:
            # Momentum signal
            momentum = data[col].iloc[-self.momentum_window:].sum()
            
            # RSI calculation
            deltas = data[col].iloc[-(self.rsi_period + 10):].diff()
            gains = deltas.where(deltas > 0, 0).rolling(self.rsi_period).mean()
            losses = (-deltas).where(deltas < 0, 0).rolling(self.rsi_period).mean()
            rs = gains / losses.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            # Bollinger Bands
            prices = data[col].iloc[-self.bb_period:].cumsum()
            bb_mean = prices.mean()
            bb_std = prices.std()
            current_price = prices.iloc[-1]
            bb_position = (current_price - bb_mean) / bb_std
            
            # Reversal timing signals
            momentum_reversal = 0
            if momentum > 0:  # Positive momentum
                if current_rsi > 70 and bb_position > 1.5:
                    momentum_reversal = -1.0  # Strong sell signal
                elif current_rsi > 60 and bb_position > 1.0:
                    momentum_reversal = -0.5  # Mild sell signal
            else:  # Negative momentum
                if current_rsi < 30 and bb_position < -1.5:
                    momentum_reversal = 1.0  # Strong buy signal
                elif current_rsi < 40 and bb_position < -1.0:
                    momentum_reversal = 0.5  # Mild buy signal
            
            signals[col] = momentum_reversal
        
        return pd.Series(signals)

class OptionsGreeksSimulation(BaseAlpha):
    """Simulates options greeks behavior for equity strategies."""
    
    def __init__(self, time_to_expiry: int = 30, strike_distance: float = 0.02):
        super().__init__(
            name=f"options_greeks_sim_{time_to_expiry}d_{strike_distance}",
            description=f"Options greeks simulation {time_to_expiry}d expiry {strike_distance} strike"
        )
        self.time_to_expiry = time_to_expiry
        self.strike_distance = strike_distance
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < 30:
            return pd.Series(dtype=float)
        
        # Simulate basic options greeks
        current_vol = data.iloc[-20:].std() * np.sqrt(252)
        recent_returns = data.iloc[-5:].sum()
        
        # Delta proxy (directional exposure)
        delta_proxy = np.tanh(recent_returns / current_vol)  # Bounded between -1 and 1
        
        # Gamma proxy (convexity)
        price_acceleration = data.iloc[-3:].sum() - data.iloc[-6:-3].sum()
        gamma_proxy = price_acceleration / (current_vol + 1e-6)
        
        # Theta proxy (time decay) - favor assets that might gain from time decay
        theta_proxy = current_vol / np.sqrt(self.time_to_expiry / 252)
        
        # Vega proxy (volatility sensitivity)
        vol_change = current_vol - data.iloc[-40:].std() * np.sqrt(252)
        vega_proxy = vol_change / current_vol
        
        # Combined greeks signal
        greeks_signal = (0.4 * delta_proxy + 
                        0.2 * gamma_proxy + 
                        0.2 * (-theta_proxy) +  # Negative theta is good
                        0.2 * vega_proxy)
        
        return greeks_signal

class FractalMarketEfficiency(BaseAlpha):
    """Exploits temporary market inefficiencies using fractal analysis."""
    
    def __init__(self, fractal_period: int = 14, efficiency_threshold: float = 0.5):
        super().__init__(
            name=f"fractal_market_efficiency_{fractal_period}d_{efficiency_threshold}",
            description=f"Fractal market efficiency {fractal_period}d threshold {efficiency_threshold}"
        )
        self.fractal_period = fractal_period
        self.efficiency_threshold = efficiency_threshold
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.fractal_period + 10:
            return pd.Series(dtype=float)
        
        signals = {}
        
        for col in data.columns:
            recent_data = data[col].iloc[-self.fractal_period-5:]
            
            # Calculate fractal efficiency
            # Method: Compare actual path length to straight-line distance
            cumulative_returns = recent_data.cumsum()
            total_distance = cumulative_returns.iloc[-1] - cumulative_returns.iloc[0]  # Straight line
            actual_path = recent_data.abs().sum()  # Actual path length
            
            # Efficiency ratio
            efficiency = abs(total_distance) / actual_path if actual_path > 0 else 1
            
            # When market is inefficient (low efficiency), expect mean reversion
            # When market is efficient (high efficiency), expect trend continuation
            if efficiency < self.efficiency_threshold:
                # Inefficient market: mean reversion
                signals[col] = -recent_data.iloc[-3:].sum()  # Fade recent moves
            else:
                # Efficient market: momentum
                signals[col] = recent_data.iloc[-5:].sum()  # Follow trend
        
        return pd.Series(signals)

class EconomicRegimeTransition(BaseAlpha):
    """Trades regime transitions in economic cycles."""
    
    def __init__(self, regime_window: int = 60, transition_sensitivity: float = 1.5):
        super().__init__(
            name=f"economic_regime_transition_{regime_window}d_{transition_sensitivity}",
            description=f"Economic regime transition {regime_window}d sensitivity {transition_sensitivity}"
        )
        self.regime_window = regime_window
        self.transition_sensitivity = transition_sensitivity
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.regime_window + 20:
            return pd.Series(dtype=float)
        
        # Economic regime proxies
        market_returns = data.mean(axis=1)
        
        # Volatility regime
        current_vol = market_returns.iloc[-self.regime_window:].std()
        historical_vol = market_returns.std()
        vol_regime = current_vol / historical_vol
        
        # Growth regime (trend)
        growth_trend = market_returns.iloc[-self.regime_window:].sum()
        historical_trend = market_returns.mean() * self.regime_window
        growth_regime = growth_trend / abs(historical_trend) if historical_trend != 0 else 0
        
        # Cross-sectional dispersion regime
        dispersion = data.iloc[-self.regime_window:].std(axis=1).mean()
        historical_dispersion = data.std(axis=1).mean()
        dispersion_regime = dispersion / historical_dispersion
        
        # Detect regime transitions
        vol_transition = abs(vol_regime - 1) > self.transition_sensitivity
        growth_transition = abs(growth_regime) > self.transition_sensitivity
        dispersion_transition = abs(dispersion_regime - 1) > self.transition_sensitivity
        
        # Strategy based on regime transitions
        if vol_transition:
            # Volatility regime change: favor low beta assets
            recent_returns = data.iloc[-10:].sum()
            market_exposure = recent_returns.corr(market_returns.iloc[-10:])
            signal = -market_exposure  # Favor low correlation
        elif growth_transition:
            # Growth regime change: momentum/reversal based on direction
            momentum = data.iloc[-20:].sum()
            if growth_regime > 0:
                signal = momentum  # Growth acceleration: momentum
            else:
                signal = -momentum  # Growth deceleration: reversal
        elif dispersion_transition:
            # Dispersion regime change: favor convergence/divergence
            returns = data.iloc[-self.regime_window//2:].sum()
            cross_sectional_rank = returns.rank(pct=True)
            signal = 0.5 - cross_sectional_rank  # Favor extreme performers
        else:
            # No transition: neutral
            signal = pd.Series(0.0, index=data.columns)
        
        return signal

class RiskAdjustedCarry(BaseAlpha):
    """Enhanced carry strategy with dynamic risk adjustment."""
    
    def __init__(self, carry_window: int = 40, risk_window: int = 20, alpha: float = 0.1):
        super().__init__(
            name=f"risk_adjusted_carry_{carry_window}d_{risk_window}d_{alpha}",
            description=f"Risk-adjusted carry {carry_window}/{risk_window}d alpha {alpha}"
        )
        self.carry_window = carry_window
        self.risk_window = risk_window
        self.alpha = alpha
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < max(self.carry_window, self.risk_window) + 10:
            return pd.Series(dtype=float)
        
        # Expected return (carry)
        expected_return = data.iloc[-self.carry_window:].mean()
        
        # Risk measures
        volatility = data.iloc[-self.risk_window:].std()
        
        # Downside risk (semi-deviation)
        downside_returns = data.iloc[-self.risk_window:].copy()
        downside_returns[downside_returns > 0] = 0
        downside_risk = downside_returns.std()
        
        # Tail risk (VaR approximation)
        var_5pct = data.iloc[-self.risk_window:].quantile(0.05)
        tail_risk = abs(var_5pct)
        
        # Dynamic risk adjustment based on market regime
        market_stress = data.std(axis=1).iloc[-self.risk_window:].mean()
        historical_stress = data.std(axis=1).mean()
        stress_multiplier = (market_stress / historical_stress) ** self.alpha
        
        # Adjusted risk measure
        adjusted_risk = (volatility + downside_risk + tail_risk) * stress_multiplier
        
        # Risk-adjusted carry
        risk_adjusted_carry = expected_return / adjusted_risk.replace(0, np.nan)
        
        return risk_adjusted_carry.fillna(0)

class MarketMakerSignal(BaseAlpha):
    """Mimics market maker inventory management strategies."""
    
    def __init__(self, inventory_window: int = 10, spread_proxy_window: int = 5):
        super().__init__(
            name=f"market_maker_signal_{inventory_window}d_{spread_proxy_window}d",
            description=f"Market maker signal {inventory_window}/{spread_proxy_window}d"
        )
        self.inventory_window = inventory_window
        self.spread_proxy_window = spread_proxy_window
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < max(self.inventory_window, self.spread_proxy_window) + 10:
            return pd.Series(dtype=float)
        
        # Inventory proxy: cumulative signed returns
        inventory_proxy = data.iloc[-self.inventory_window:].sum()
        
        # Spread proxy: short-term volatility
        spread_proxy = data.iloc[-self.spread_proxy_window:].std()
        
        # Order flow imbalance proxy: return skewness
        order_flow_imbalance = data.iloc[-self.inventory_window:].skew()
        
        # Market maker would want to:
        # 1. Reduce inventory when it's extreme
        # 2. Widen spreads in volatile conditions
        # 3. Lean against order flow imbalances
        
        # Inventory management signal
        inventory_signal = -np.sign(inventory_proxy) * abs(inventory_proxy) / spread_proxy
        
        # Order flow lean signal
        flow_signal = -order_flow_imbalance * 0.5
        
        # Combined market maker signal
        mm_signal = inventory_signal + flow_signal
        
        # Scale by spread (wider spreads allow more aggressive positions)
        return mm_signal * (1 + spread_proxy)

class EarningsQualityMomentum(BaseAlpha):
    """Momentum based on earnings quality metrics."""
    
    def __init__(self, earnings_window: int = 20, quality_window: int = 60):
        super().__init__(
            name=f"earnings_quality_momentum_{earnings_window}d_{quality_window}d",
            description=f"Earnings quality momentum {earnings_window}/{quality_window}d"
        )
        self.earnings_window = earnings_window
        self.quality_window = quality_window
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.quality_window:
            return pd.Series(dtype=float)
        
        # Proxy for earnings events: large single-day moves
        daily_returns = data
        vol_threshold = daily_returns.rolling(30).std() * 2  # 2-sigma threshold
        
        # Identify potential earnings days
        earnings_candidates = daily_returns.abs() > vol_threshold
        
        signals = {}
        
        for col in data.columns:
            # Recent earnings-like events
            recent_earnings = earnings_candidates[col].iloc[-self.earnings_window:].sum()
            
            if recent_earnings > 0:
                # Quality metrics for earnings
                earnings_periods = daily_returns[col].iloc[-self.quality_window:][earnings_candidates[col].iloc[-self.quality_window:]]
                
                if len(earnings_periods) > 1:
                    # Earnings quality: consistency of direction
                    earnings_consistency = (earnings_periods > 0).mean()
                    
                    # Earnings momentum
                    earnings_momentum = earnings_periods.sum()
                    
                    # Quality-adjusted momentum
                    quality_adjustment = abs(earnings_consistency - 0.5) * 2  # 0 to 1 scale
                    signals[col] = earnings_momentum * quality_adjustment
                else:
                    signals[col] = 0.0
            else:
                # No recent earnings: use regular momentum
                signals[col] = daily_returns[col].iloc[-self.earnings_window:].sum() * 0.5
        
        return pd.Series(signals)

class VolatilityClusteringBreakout(BaseAlpha):
    """Breakout strategy enhanced with volatility clustering."""
    
    def __init__(self, cluster_window: int = 15, breakout_threshold: float = 1.8):
        super().__init__(
            name=f"vol_clustering_breakout_{cluster_window}d_{breakout_threshold}",
            description=f"Volatility clustering breakout {cluster_window}d threshold {breakout_threshold}"
        )
        self.cluster_window = cluster_window
        self.breakout_threshold = breakout_threshold
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.cluster_window + 20:
            return pd.Series(dtype=float)
        
        # Volatility clustering detection
        rolling_vol = data.rolling(self.cluster_window).std()
        vol_autocorr = rolling_vol.iloc[-30:].apply(lambda x: x.autocorr(lag=1))
        
        # High autocorrelation indicates clustering
        clustering_strength = vol_autocorr.fillna(0)
        
        # Current volatility vs recent average
        current_vol = rolling_vol.iloc[-1]
        avg_vol = rolling_vol.iloc[-self.cluster_window:].mean()
        vol_ratio = current_vol / avg_vol
        
        # Breakout detection
        recent_returns = data.iloc[-3:].sum()
        return_magnitude = abs(recent_returns)
        
        # Enhanced breakout signal
        breakout_condition = vol_ratio > self.breakout_threshold
        
        # Combine volatility clustering with breakout
        signals = {}
        for col in data.columns:
            if breakout_condition[col] and clustering_strength[col] > 0.3:
                # Strong breakout with volatility clustering
                direction = np.sign(recent_returns[col])
                strength = min(vol_ratio[col] / self.breakout_threshold, 2.0)
                clustering_boost = 1 + clustering_strength[col]
                signals[col] = direction * strength * clustering_boost
            elif vol_ratio[col] < 0.8 and clustering_strength[col] > 0.5:
                # Low vol with clustering: expect volatility expansion
                signals[col] = np.sign(recent_returns[col]) * 0.5
            else:
                signals[col] = 0.0
        
        return pd.Series(signals)