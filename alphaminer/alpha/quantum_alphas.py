import pandas as pd
import numpy as np
from .base_alpha import BaseAlpha

class QuantumInfoRatioAlpha(BaseAlpha):
    """Enhanced information ratio with volatility-adjusted benchmark."""
    
    def __init__(self, signal_window: int = 15, benchmark_window: int = 35, vol_adj: float = 0.8):
        super().__init__(
            name=f"quantum_info_ratio_{signal_window}d_bench{benchmark_window}d_vol{vol_adj}",
            description=f"Quantum info ratio {signal_window}d vs {benchmark_window}d with vol adjustment"
        )
        self.signal_window = signal_window
        self.benchmark_window = benchmark_window
        self.vol_adj = vol_adj
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.benchmark_window:
            return pd.Series(dtype=float)
        
        # Calculate excess returns vs dynamic benchmark
        returns = data.iloc[-self.signal_window:].sum()
        
        # Dynamic benchmark: volatility-weighted average
        recent_vol = data.iloc[-self.benchmark_window:].std()
        inv_vol_weights = (1 / recent_vol.replace(0, np.inf))
        inv_vol_weights = inv_vol_weights / inv_vol_weights.sum()
        benchmark_return = (data.iloc[-self.benchmark_window:].mean(axis=1) * inv_vol_weights).sum()
        
        excess_returns = returns - benchmark_return
        
        # Enhanced tracking error with volatility adjustment
        tracking_error = data.iloc[-self.signal_window:].std() * (1 + self.vol_adj * recent_vol)
        
        # Quantum information ratio
        quantum_info_ratio = excess_returns / tracking_error.replace(0, np.nan)
        
        return quantum_info_ratio.fillna(0)

class AdvancedMicrostructureAlpha(BaseAlpha):
    """Advanced microstructure patterns with order flow simulation."""
    
    def __init__(self, window: int = 12, flow_sensitivity: float = 0.7, spread_factor: float = 1.2):
        super().__init__(
            name=f"advanced_microstructure_{window}d_flow{flow_sensitivity}_spread{spread_factor}",
            description=f"Advanced microstructure {window}d with flow sensitivity {flow_sensitivity}"
        )
        self.window = window
        self.flow_sensitivity = flow_sensitivity
        self.spread_factor = spread_factor
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.window + 5:
            return pd.Series(dtype=float)
        
        recent_data = data.iloc[-self.window:]
        
        # Enhanced bid-ask spread proxy using volatility clustering
        vol_clustering = recent_data.rolling(3).std().std()
        spread_proxy = vol_clustering * self.spread_factor
        
        # Order flow imbalance using return autocorrelation
        flow_imbalance = recent_data.apply(lambda x: x.autocorr(lag=1) * self.flow_sensitivity)
        flow_imbalance = flow_imbalance.fillna(0)
        
        # Market impact proxy using squared returns
        market_impact = (recent_data ** 2).mean()
        
        # Combined microstructure signal
        microstructure_signal = (-spread_proxy + 0.6 * flow_imbalance - 0.3 * market_impact)
        
        return microstructure_signal

class CrossSectionalMomentumDecayQuantum(BaseAlpha):
    """Quantum cross-sectional momentum with exponential decay."""
    
    def __init__(self, momentum_window: int = 18, decay_window: int = 6, quantum_factor: float = 0.85):
        super().__init__(
            name=f"cs_momentum_decay_quantum_{momentum_window}d_{decay_window}d_q{quantum_factor}",
            description=f"Quantum CS momentum decay {momentum_window}d/{decay_window}d with quantum factor"
        )
        self.momentum_window = momentum_window
        self.decay_window = decay_window
        self.quantum_factor = quantum_factor
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.momentum_window:
            return pd.Series(dtype=float)
        
        # Calculate momentum with quantum exponential weighting
        weights = np.array([self.quantum_factor**i for i in range(self.momentum_window-1, -1, -1)])
        weights = weights / weights.sum()
        
        momentum_data = data.iloc[-self.momentum_window:]
        quantum_momentum = (momentum_data.values * weights.reshape(-1, 1)).sum(axis=0)
        quantum_momentum = pd.Series(quantum_momentum, index=data.columns)
        
        # Cross-sectional ranking with decay adjustment
        cs_ranks = quantum_momentum.rank(pct=True)
        
        # Apply decay based on recent volatility
        decay_vol = data.iloc[-self.decay_window:].std()
        decay_factor = 1 / (1 + decay_vol)
        
        # Quantum momentum signal
        quantum_signal = (cs_ranks - 0.5) * 2 * decay_factor  # Convert to [-1, 1]
        
        return quantum_signal

class QuantumDispersionAlpha(BaseAlpha):
    """Quantum cross-sectional dispersion with regime detection."""
    
    def __init__(self, dispersion_window: int = 25, regime_window: int = 60, quantum_threshold: float = 1.3):
        super().__init__(
            name=f"quantum_dispersion_{dispersion_window}d_regime{regime_window}d_thresh{quantum_threshold}",
            description=f"Quantum dispersion {dispersion_window}d with regime detection"
        )
        self.dispersion_window = dispersion_window
        self.regime_window = regime_window
        self.quantum_threshold = quantum_threshold
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.regime_window:
            return pd.Series(dtype=float)
        
        recent_data = data.iloc[-self.dispersion_window:]
        regime_data = data.iloc[-self.regime_window:]
        
        # Calculate quantum dispersion
        daily_std = recent_data.std(axis=1)
        current_dispersion = daily_std.iloc[-1]
        avg_dispersion = daily_std.mean()
        quantum_dispersion_ratio = current_dispersion / avg_dispersion
        
        # Regime detection using volatility clustering
        vol_regime = regime_data.std(axis=1)
        high_vol_regime = vol_regime.iloc[-10:].mean() > vol_regime.quantile(0.7)
        
        if quantum_dispersion_ratio > self.quantum_threshold:
            if high_vol_regime:
                # High dispersion + high vol: strong mean reversion
                returns = recent_data.iloc[-3:].sum()
                signal = -returns * 1.5
            else:
                # High dispersion + low vol: mild mean reversion
                returns = recent_data.iloc[-5:].sum()
                signal = -returns * 0.8
        else:
            if high_vol_regime:
                # Low dispersion + high vol: momentum with caution
                returns = recent_data.iloc[-8:].sum()
                signal = returns * 0.6
            else:
                # Low dispersion + low vol: momentum
                returns = recent_data.iloc[-10:].sum()
                signal = returns
        
        return signal

class QuantumVolSurfaceAlpha(BaseAlpha):
    """Quantum volatility surface with multi-horizon arbitrage."""
    
    def __init__(self, short_vol: int = 7, medium_vol: int = 21, long_vol: int = 42, quantum_decay: float = 0.9):
        super().__init__(
            name=f"quantum_vol_surface_{short_vol}d_{medium_vol}d_{long_vol}d_decay{quantum_decay}",
            description=f"Quantum vol surface {short_vol}/{medium_vol}/{long_vol}d with quantum decay"
        )
        self.short_vol = short_vol
        self.medium_vol = medium_vol
        self.long_vol = long_vol
        self.quantum_decay = quantum_decay
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.long_vol:
            return pd.Series(dtype=float)
        
        # Calculate multi-horizon volatilities
        short_vol = data.iloc[-self.short_vol:].std()
        medium_vol = data.iloc[-self.medium_vol:].std()
        long_vol = data.iloc[-self.long_vol:].std()
        
        # Quantum volatility surface slopes
        slope1 = (medium_vol - short_vol) / short_vol
        slope2 = (long_vol - medium_vol) / medium_vol
        
        # Quantum decay weighting
        quantum_weights = [1.0, self.quantum_decay, self.quantum_decay**2]
        quantum_slope = (quantum_weights[0] * slope1 + 
                        quantum_weights[1] * slope2) / sum(quantum_weights[:2])
        
        # Term structure curvature
        curvature = slope2 - slope1
        
        # Combined quantum signal favoring inverted and curved structures
        quantum_signal = -quantum_slope + 0.3 * curvature
        
        return quantum_signal

class MultihorizonMomentumQuantum(BaseAlpha):
    """Multi-horizon momentum with quantum interference patterns."""
    
    def __init__(self, horizons: list = [5, 15, 30], quantum_interference: float = 0.3):
        super().__init__(
            name=f"multihorizon_momentum_quantum_{len(horizons)}h_interference{quantum_interference}",
            description=f"Multi-horizon quantum momentum with {len(horizons)} horizons"
        )
        self.horizons = horizons
        self.quantum_interference = quantum_interference
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < max(self.horizons):
            return pd.Series(dtype=float)
        
        # Calculate momentum signals across horizons
        momentum_signals = []
        for horizon in self.horizons:
            momentum = data.iloc[-horizon:].sum()
            momentum_signals.append(momentum / np.sqrt(horizon))  # Time-scaled
        
        # Quantum interference: constructive when signals align
        signal_correlation = np.corrcoef([s.values for s in momentum_signals]).mean()
        interference_factor = 1 + self.quantum_interference * signal_correlation
        
        # Weighted combination with quantum interference
        weights = np.array([1/h for h in self.horizons])
        weights = weights / weights.sum()
        
        combined_momentum = sum(w * s for w, s in zip(weights, momentum_signals))
        quantum_momentum = combined_momentum * interference_factor
        
        return quantum_momentum

class VolatilityQuantumTunneling(BaseAlpha):
    """Volatility quantum tunneling through resistance levels."""
    
    def __init__(self, tunnel_window: int = 20, resistance_factor: float = 1.5, tunnel_probability: float = 0.7):
        super().__init__(
            name=f"vol_quantum_tunneling_{tunnel_window}d_resist{resistance_factor}_prob{tunnel_probability}",
            description=f"Vol quantum tunneling {tunnel_window}d with resistance factor {resistance_factor}"
        )
        self.tunnel_window = tunnel_window
        self.resistance_factor = resistance_factor
        self.tunnel_probability = tunnel_probability
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.tunnel_window + 10:
            return pd.Series(dtype=float)
        
        recent_data = data.iloc[-self.tunnel_window:]
        
        # Calculate volatility resistance levels
        vol_resistance = recent_data.std() * self.resistance_factor
        current_vol = recent_data.iloc[-5:].std()
        
        # Quantum tunneling probability
        tunnel_ratio = current_vol / vol_resistance
        tunnel_signal = np.where(tunnel_ratio > 1, 
                                np.log(tunnel_ratio) * self.tunnel_probability, 
                                -np.exp(tunnel_ratio - 1))
        
        # Direction based on recent momentum
        momentum_direction = np.sign(recent_data.iloc[-3:].sum())
        
        return pd.Series(tunnel_signal * momentum_direction, index=data.columns)

class QuantumMeanReversionHarmonic(BaseAlpha):
    """Quantum mean reversion using harmonic oscillator model."""
    
    def __init__(self, oscillation_period: int = 14, damping_factor: float = 0.8, spring_constant: float = 1.2):
        super().__init__(
            name=f"quantum_mean_reversion_harmonic_{oscillation_period}d_damp{damping_factor}_spring{spring_constant}",
            description=f"Quantum harmonic mean reversion {oscillation_period}d periods"
        )
        self.oscillation_period = oscillation_period
        self.damping_factor = damping_factor
        self.spring_constant = spring_constant
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.oscillation_period * 2:
            return pd.Series(dtype=float)
        
        recent_data = data.iloc[-self.oscillation_period * 2:]
        
        # Calculate equilibrium position (harmonic center)
        equilibrium = recent_data.mean()
        current_position = recent_data.iloc[-1]
        
        # Displacement from equilibrium
        displacement = current_position - equilibrium
        
        # Quantum harmonic oscillator force
        restoring_force = -self.spring_constant * displacement
        
        # Apply damping based on recent volatility
        damping = self.damping_factor * recent_data.iloc[-self.oscillation_period:].std()
        
        # Quantum mean reversion signal
        quantum_harmonic_signal = restoring_force * (1 - damping)
        
        return quantum_harmonic_signal

class AdaptiveQuantumMomentum(BaseAlpha):
    """Adaptive quantum momentum with uncertainty principle."""
    
    def __init__(self, base_period: int = 10, max_period: int = 30, uncertainty_factor: float = 0.5):
        super().__init__(
            name=f"adaptive_quantum_momentum_{base_period}d_max{max_period}d_uncertainty{uncertainty_factor}",
            description=f"Adaptive quantum momentum with uncertainty principle"
        )
        self.base_period = base_period
        self.max_period = max_period
        self.uncertainty_factor = uncertainty_factor
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.max_period:
            return pd.Series(dtype=float)
        
        # Adaptive period based on market volatility
        market_vol = data.iloc[-self.max_period:].std(axis=1).mean()
        vol_percentile = (data.iloc[-lookback_window:].std(axis=1) < market_vol).mean()
        
        # Uncertainty principle: higher precision in time -> lower precision in momentum
        adaptive_period = int(self.base_period + (1 - vol_percentile) * (self.max_period - self.base_period))
        
        # Calculate quantum momentum with uncertainty
        momentum = data.iloc[-adaptive_period:].sum()
        momentum_variance = data.iloc[-adaptive_period:].var()
        
        # Uncertainty adjustment
        uncertainty_adjustment = 1 - self.uncertainty_factor * momentum_variance / (momentum_variance + 1e-6)
        
        quantum_momentum = momentum * uncertainty_adjustment
        
        return quantum_momentum

class QuantumEntanglementAlpha(BaseAlpha):
    """Quantum entanglement between correlated assets."""
    
    def __init__(self, entanglement_window: int = 30, correlation_threshold: float = 0.6, quantum_strength: float = 0.8):
        super().__init__(
            name=f"quantum_entanglement_{entanglement_window}d_corr{correlation_threshold}_strength{quantum_strength}",
            description=f"Quantum entanglement with correlation threshold {correlation_threshold}"
        )
        self.entanglement_window = entanglement_window
        self.correlation_threshold = correlation_threshold
        self.quantum_strength = quantum_strength
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.entanglement_window:
            return pd.Series(dtype=float)
        
        recent_data = data.iloc[-self.entanglement_window:]
        
        # Calculate correlation matrix
        correlation_matrix = recent_data.corr()
        
        # Find quantum entangled pairs (high correlation)
        entangled_signals = pd.Series(0.0, index=data.columns)
        
        for i, asset1 in enumerate(data.columns):
            for j, asset2 in enumerate(data.columns):
                if i < j and abs(correlation_matrix.loc[asset1, asset2]) > self.correlation_threshold:
                    # Quantum entanglement: when one moves, predict the other
                    asset1_momentum = recent_data[asset1].iloc[-3:].sum()
                    asset2_momentum = recent_data[asset2].iloc[-3:].sum()
                    
                    # Entanglement signal: predict asset2 based on asset1
                    entanglement_signal = np.sign(asset1_momentum) * self.quantum_strength
                    entangled_signals[asset2] += entanglement_signal
                    
                    # Symmetric entanglement
                    entanglement_signal = np.sign(asset2_momentum) * self.quantum_strength
                    entangled_signals[asset1] += entanglement_signal
        
        # Normalize entanglement signals
        max_entanglement = abs(entangled_signals).max()
        if max_entanglement > 0:
            entangled_signals = entangled_signals / max_entanglement
        
        return entangled_signals