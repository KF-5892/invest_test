import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from .base_alpha import BaseAlpha

class MultiScaleEntropyFilter(BaseAlpha):
    """Multi-scale entropy-based signal filtering for noise reduction."""
    
    def __init__(self, scales: list = [5, 10, 20], entropy_threshold: float = 0.6):
        super().__init__(
            name=f"multi_scale_entropy_{scales[0]}_{scales[1]}_{scales[2]}_thresh{entropy_threshold}",
            description=f"Multi-scale entropy filter with scales {scales}"
        )
        self.scales = scales
        self.entropy_threshold = entropy_threshold
    
    def sample_entropy(self, data, m=2, r=0.2):
        """Calculate sample entropy for time series."""
        N = len(data)
        if N < m + 1:
            return 0
        
        patterns = np.array([data[i:i+m] for i in range(N-m+1)])
        matches = 0
        conditional_matches = 0
        
        for i in range(len(patterns)):
            template = patterns[i]
            distances = np.max(np.abs(patterns - template), axis=1)
            matches += np.sum(distances <= r) - 1  # exclude self-match
            
            if i < len(patterns) - 1:
                template_ext = np.append(template, data[i+m])
                patterns_ext = np.array([data[j:j+m+1] for j in range(N-m) if j != i])
                if len(patterns_ext) > 0:
                    distances_ext = np.max(np.abs(patterns_ext - template_ext), axis=1)
                    conditional_matches += np.sum(distances_ext <= r)
        
        if matches == 0 or conditional_matches == 0:
            return 0
        
        return -np.log(conditional_matches / matches)
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < max(self.scales) + 10:
            return pd.Series(dtype=float)
        
        signals = {}
        for col in data.columns:
            series = data[col].iloc[-50:].dropna()
            if len(series) < max(self.scales):
                signals[col] = 0
                continue
            
            # Calculate entropy at different scales
            entropies = []
            for scale in self.scales:
                if len(series) >= scale:
                    scaled_data = series.iloc[-scale:]
                    entropy = self.sample_entropy(scaled_data.values)
                    entropies.append(entropy)
            
            if not entropies:
                signals[col] = 0
                continue
            
            # Multi-scale entropy score
            entropy_score = np.mean(entropies)
            
            # Signal strength based on entropy
            if entropy_score > self.entropy_threshold:
                # High entropy = high noise, use mean reversion
                recent_return = series.iloc[-5:].sum()
                signals[col] = -recent_return * (entropy_score - self.entropy_threshold)
            else:
                # Low entropy = trend, use momentum
                recent_return = series.iloc[-10:].sum()
                signals[col] = recent_return * (self.entropy_threshold - entropy_score)
        
        return pd.Series(signals)

class BayesianRegimeSwitching(BaseAlpha):
    """Bayesian approach to regime switching with uncertainty quantification."""
    
    def __init__(self, regime_window: int = 30, confidence_threshold: float = 0.7):
        super().__init__(
            name=f"bayesian_regime_{regime_window}d_conf{confidence_threshold}",
            description=f"Bayesian regime switching with {regime_window}d window"
        )
        self.regime_window = regime_window
        self.confidence_threshold = confidence_threshold
    
    def calculate_regime_probabilities(self, returns):
        """Calculate probability of being in high/low volatility regime."""
        vol = returns.std()
        mean_vol = returns.rolling(60).std().mean() if len(returns) >= 60 else vol
        
        # Simple Bayesian update
        # Prior: equal probability for high/low vol regime
        prior_high = 0.5
        
        # Likelihood: probability of current vol given regime
        if vol > mean_vol:
            likelihood_high = 0.8  # High vol more likely in high vol regime
            likelihood_low = 0.2
        else:
            likelihood_high = 0.3  # Low vol less likely in high vol regime
            likelihood_low = 0.7
        
        # Posterior probability using Bayes' theorem
        evidence = likelihood_high * prior_high + likelihood_low * (1 - prior_high)
        posterior_high = (likelihood_high * prior_high) / evidence
        
        return posterior_high, 1 - posterior_high
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.regime_window + 10:
            return pd.Series(dtype=float)
        
        signals = {}
        recent_data = data.iloc[-self.regime_window:]
        
        for col in data.columns:
            series = recent_data[col].dropna()
            if len(series) < 20:
                signals[col] = 0
                continue
            
            # Calculate regime probabilities
            prob_high_vol, prob_low_vol = self.calculate_regime_probabilities(series)
            
            # Only trade when confident about regime
            max_prob = max(prob_high_vol, prob_low_vol)
            if max_prob < self.confidence_threshold:
                signals[col] = 0
                continue
            
            recent_return = series.iloc[-5:].sum()
            
            if prob_high_vol > prob_low_vol:
                # High volatility regime: mean reversion
                signals[col] = -recent_return * prob_high_vol
            else:
                # Low volatility regime: momentum
                signals[col] = recent_return * prob_low_vol
        
        return pd.Series(signals)

class VolatilitySmileArbitrage(BaseAlpha):
    """Exploit volatility smile patterns across different strikes."""
    
    def __init__(self, lookback: int = 20, smile_window: int = 10):
        super().__init__(
            name=f"vol_smile_arb_{lookback}d_{smile_window}w",
            description=f"Volatility smile arbitrage {lookback}d lookback"
        )
        self.lookback = lookback
        self.smile_window = smile_window
    
    def estimate_implied_vol_smile(self, returns):
        """Estimate implied volatility smile from return distribution."""
        if len(returns) < 10:
            return 0, 0
        
        # Calculate skewness and kurtosis as smile proxies
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # Smile convexity measure
        smile_convexity = abs(skewness) + 0.5 * abs(kurtosis - 3)
        
        return skewness, smile_convexity
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.lookback + 10:
            return pd.Series(dtype=float)
        
        signals = {}
        recent_data = data.iloc[-self.lookback:]
        
        for col in data.columns:
            returns = recent_data[col].dropna()
            if len(returns) < 15:
                signals[col] = 0
                continue
            
            # Estimate current smile
            current_skew, current_convexity = self.estimate_implied_vol_smile(returns.iloc[-self.smile_window:])
            
            # Historical smile baseline
            hist_skew, hist_convexity = self.estimate_implied_vol_smile(returns)
            
            # Smile mean reversion signal
            skew_signal = -(current_skew - hist_skew)
            convexity_signal = -(current_convexity - hist_convexity)
            
            # Combine signals
            signals[col] = 0.6 * skew_signal + 0.4 * convexity_signal
        
        return pd.Series(signals)

class NeuralNetworkMomentum(BaseAlpha):
    """Simple neural network-inspired momentum with adaptive weights."""
    
    def __init__(self, input_nodes: int = 10, hidden_nodes: int = 5, learning_rate: float = 0.01):
        super().__init__(
            name=f"neural_momentum_{input_nodes}_{hidden_nodes}_lr{learning_rate}",
            description=f"Neural network momentum with {hidden_nodes} hidden nodes"
        )
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.learning_rate = learning_rate
        self.weights1 = None
        self.weights2 = None
    
    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def initialize_weights(self):
        """Initialize network weights."""
        self.weights1 = np.random.randn(self.input_nodes, self.hidden_nodes) * 0.1
        self.weights2 = np.random.randn(self.hidden_nodes, 1) * 0.1
    
    def forward_pass(self, X):
        """Forward pass through network."""
        if self.weights1 is None:
            self.initialize_weights()
        
        hidden = self.sigmoid(np.dot(X, self.weights1))
        output = np.tanh(np.dot(hidden, self.weights2))
        return output.flatten(), hidden
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.input_nodes + 20:
            return pd.Series(dtype=float)
        
        signals = {}
        
        for col in data.columns:
            series = data[col].iloc[-50:].dropna()
            if len(series) < self.input_nodes + 10:
                signals[col] = 0
                continue
            
            # Prepare features (rolling windows of different lengths)
            features = []
            for i in range(1, self.input_nodes + 1):
                if len(series) >= i:
                    features.append(series.iloc[-i:].sum())
                else:
                    features.append(0)
            
            X = np.array(features).reshape(1, -1)
            X = (X - X.mean()) / (X.std() + 1e-8)  # Normalize
            
            # Forward pass
            prediction, _ = self.forward_pass(X)
            signals[col] = prediction[0]
        
        return pd.Series(signals)

class CopulaBasedPairsTrading(BaseAlpha):
    """Copula-based pairs trading using dependence structure."""
    
    def __init__(self, lookback: int = 60, pairs_threshold: float = 0.5):
        super().__init__(
            name=f"copula_pairs_{lookback}d_thresh{pairs_threshold}",
            description=f"Copula-based pairs trading {lookback}d lookback"
        )
        self.lookback = lookback
        self.pairs_threshold = pairs_threshold
    
    def kendall_tau(self, x, y):
        """Calculate Kendall's tau correlation."""
        if len(x) != len(y) or len(x) < 3:
            return 0
        return stats.kendalltau(x, y)[0]
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.lookback or len(data.columns) < 2:
            return pd.Series(dtype=float)
        
        recent_data = data.iloc[-self.lookback:]
        signals = pd.Series(0.0, index=data.columns)
        
        # Find pairs with strong copula dependence
        columns = list(data.columns)
        
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns[i+1:], i+1):
                series1 = recent_data[col1].dropna()
                series2 = recent_data[col2].dropna()
                
                if len(series1) < 20 or len(series2) < 20:
                    continue
                
                # Align series
                common_idx = series1.index.intersection(series2.index)
                if len(common_idx) < 20:
                    continue
                
                s1_aligned = series1[common_idx]
                s2_aligned = series2[common_idx]
                
                # Calculate rank-based correlation (copula measure)
                tau = self.kendall_tau(s1_aligned.values, s2_aligned.values)
                
                if abs(tau) > self.pairs_threshold:
                    # Calculate spread
                    spread = s1_aligned.iloc[-5:].sum() - s2_aligned.iloc[-5:].sum()
                    
                    # Mean reversion signal
                    spread_signal = -spread * abs(tau)
                    
                    signals[col1] += spread_signal * 0.5
                    signals[col2] += -spread_signal * 0.5
        
        return signals

class WaveletTransformAlpha(BaseAlpha):
    """Wavelet-based multi-resolution analysis for trading signals."""
    
    def __init__(self, decomp_levels: int = 3, wavelet_type: str = 'db4'):
        super().__init__(
            name=f"wavelet_alpha_L{decomp_levels}_{wavelet_type}",
            description=f"Wavelet transform alpha with {decomp_levels} levels"
        )
        self.decomp_levels = decomp_levels
        self.wavelet_type = wavelet_type
    
    def simple_wavelet_decomposition(self, signal):
        """Simple wavelet-like decomposition using moving averages."""
        if len(signal) < 8:
            return [signal], [np.array([])]
        
        approximations = [signal]
        details = []
        
        current = signal.copy()
        for level in range(self.decomp_levels):
            if len(current) < 4:
                break
            
            # Simple averaging (approximation)
            window_size = 2 ** (level + 1)
            if len(current) < window_size:
                break
            
            approx = pd.Series(current).rolling(window_size).mean().dropna()
            
            # Detail as difference between consecutive approximations
            if len(approx) > 1:
                detail = np.diff(approx.values)
                details.append(detail)
                current = approx.values
            else:
                break
        
        return approximations, details
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < 32:  # Need sufficient data for wavelet analysis
            return pd.Series(dtype=float)
        
        signals = {}
        
        for col in data.columns:
            series = data[col].iloc[-64:].dropna()  # Use recent 64 points
            if len(series) < 32:
                signals[col] = 0
                continue
            
            # Wavelet decomposition
            approximations, details = self.simple_wavelet_decomposition(series.values)
            
            if not details or len(details) == 0:
                signals[col] = 0
                continue
            
            # Combine signals from different frequency components
            signal_components = []
            
            # High frequency (noise): fade recent moves
            if len(details) > 0 and len(details[0]) > 0:
                high_freq_signal = -np.mean(details[0][-3:]) if len(details[0]) >= 3 else 0
                signal_components.append(high_freq_signal)
            
            # Medium frequency (short-term trend): follow
            if len(details) > 1 and len(details[1]) > 0:
                med_freq_signal = np.mean(details[1][-2:]) if len(details[1]) >= 2 else 0
                signal_components.append(med_freq_signal)
            
            # Low frequency (long-term trend): follow with less weight
            if len(approximations) > 0 and len(approximations[0]) > 2:
                low_freq_trend = approximations[0][-1] - approximations[0][-3]
                signal_components.append(0.5 * low_freq_trend)
            
            # Weighted combination
            if signal_components:
                signals[col] = np.mean(signal_components)
            else:
                signals[col] = 0
        
        return pd.Series(signals)

class OptimalTransportDistance(BaseAlpha):
    """Optimal transport distance for distribution-based trading."""
    
    def __init__(self, lookback: int = 30, transport_reg: float = 0.1):
        super().__init__(
            name=f"optimal_transport_{lookback}d_reg{transport_reg}",
            description=f"Optimal transport distance {lookback}d lookback"
        )
        self.lookback = lookback
        self.transport_reg = transport_reg
    
    def wasserstein_distance_1d(self, u_values, v_values):
        """Compute 1D Wasserstein distance between two distributions."""
        u_sorted = np.sort(u_values)
        v_sorted = np.sort(v_values)
        
        # Pad to same length
        min_len = min(len(u_sorted), len(v_sorted))
        if min_len == 0:
            return 0
        
        u_sorted = u_sorted[:min_len]
        v_sorted = v_sorted[:min_len]
        
        return np.mean(np.abs(u_sorted - v_sorted))
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.lookback + 10:
            return pd.Series(dtype=float)
        
        signals = {}
        recent_data = data.iloc[-self.lookback:]
        historical_data = data.iloc[-60:-self.lookback] if len(data) >= 60 + self.lookback else recent_data
        
        for col in data.columns:
            recent_returns = recent_data[col].dropna()
            hist_returns = historical_data[col].dropna()
            
            if len(recent_returns) < 10 or len(hist_returns) < 10:
                signals[col] = 0
                continue
            
            # Calculate optimal transport distance
            ot_distance = self.wasserstein_distance_1d(
                recent_returns.values, 
                hist_returns.values
            )
            
            # Normalize by volatility
            vol_norm = recent_returns.std() + 1e-8
            normalized_distance = ot_distance / vol_norm
            
            # Signal: revert when distribution has changed significantly
            distance_threshold = 0.02
            if normalized_distance > distance_threshold:
                # Distribution has changed, expect reversion
                recent_mean = recent_returns.iloc[-5:].mean()
                signals[col] = -recent_mean * normalized_distance
            else:
                # Distribution stable, use momentum
                momentum = recent_returns.iloc[-5:].sum()
                signals[col] = momentum * (1 - normalized_distance)
        
        return pd.Series(signals)

class HiddenMarkovModel(BaseAlpha):
    """Hidden Markov Model for regime detection and trading."""
    
    def __init__(self, n_states: int = 3, lookback: int = 60):
        super().__init__(
            name=f"hmm_{n_states}states_{lookback}d",
            description=f"Hidden Markov Model with {n_states} states"
        )
        self.n_states = n_states
        self.lookback = lookback
    
    def simple_hmm_viterbi(self, observations):
        """Simplified Viterbi algorithm for HMM state estimation."""
        n_obs = len(observations)
        if n_obs < 3:
            return np.zeros(n_obs)
        
        # Simple state transition probabilities
        trans_prob = np.ones((self.n_states, self.n_states)) / self.n_states
        
        # Emission probabilities based on return magnitude
        states = np.zeros(n_obs)
        
        # Classify observations into states based on volatility
        vol_threshold_low = np.percentile(np.abs(observations), 33)
        vol_threshold_high = np.percentile(np.abs(observations), 67)
        
        for i, obs in enumerate(observations):
            abs_obs = abs(obs)
            if abs_obs <= vol_threshold_low:
                states[i] = 0  # Low volatility state
            elif abs_obs <= vol_threshold_high:
                states[i] = 1  # Medium volatility state
            else:
                states[i] = 2  # High volatility state
        
        return states
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.lookback:
            return pd.Series(dtype=float)
        
        signals = {}
        recent_data = data.iloc[-self.lookback:]
        
        for col in data.columns:
            returns = recent_data[col].dropna()
            if len(returns) < 20:
                signals[col] = 0
                continue
            
            # Estimate HMM states
            states = self.simple_hmm_viterbi(returns.values)
            current_state = states[-1] if len(states) > 0 else 1
            
            # Trading strategy based on current state
            recent_return = returns.iloc[-5:].sum()
            
            if current_state == 0:  # Low vol state
                # Momentum strategy
                signals[col] = recent_return
            elif current_state == 1:  # Medium vol state
                # Neutral/small mean reversion
                signals[col] = -0.3 * recent_return
            else:  # High vol state
                # Strong mean reversion
                signals[col] = -recent_return
        
        return pd.Series(signals)

class FractalDimensionTrading(BaseAlpha):
    """Trading based on fractal dimension analysis."""
    
    def __init__(self, lookback: int = 50, dimension_threshold: float = 1.5):
        super().__init__(
            name=f"fractal_dimension_{lookback}d_thresh{dimension_threshold}",
            description=f"Fractal dimension trading {lookback}d lookback"
        )
        self.lookback = lookback
        self.dimension_threshold = dimension_threshold
    
    def hurst_exponent(self, ts):
        """Calculate Hurst exponent as proxy for fractal dimension."""
        if len(ts) < 10:
            return 0.5
        
        # Remove mean
        ts = ts - np.mean(ts)
        
        # Calculate R/S statistic
        lags = range(2, min(20, len(ts)//2))
        rs_values = []
        
        for lag in lags:
            # Split series
            chunks = [ts[i:i+lag] for i in range(0, len(ts)-lag, lag)]
            
            if not chunks:
                continue
            
            rs_chunk = []
            for chunk in chunks:
                if len(chunk) < 2:
                    continue
                
                # Cumulative sum
                cum_sum = np.cumsum(chunk - np.mean(chunk))
                
                # Range
                R = np.max(cum_sum) - np.min(cum_sum)
                
                # Standard deviation
                S = np.std(chunk)
                
                if S > 0:
                    rs_chunk.append(R / S)
            
            if rs_chunk:
                rs_values.append(np.mean(rs_chunk))
        
        if len(rs_values) < 3:
            return 0.5
        
        # Linear regression to find Hurst exponent
        log_lags = np.log(lags[:len(rs_values)])
        log_rs = np.log(rs_values)
        
        # Simple linear regression
        n = len(log_lags)
        sum_x = np.sum(log_lags)
        sum_y = np.sum(log_rs)
        sum_xy = np.sum(log_lags * log_rs)
        sum_x2 = np.sum(log_lags ** 2)
        
        if n * sum_x2 - sum_x ** 2 == 0:
            return 0.5
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        return slope
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.lookback:
            return pd.Series(dtype=float)
        
        signals = {}
        recent_data = data.iloc[-self.lookback:]
        
        for col in data.columns:
            returns = recent_data[col].dropna()
            if len(returns) < 20:
                signals[col] = 0
                continue
            
            # Calculate Hurst exponent (related to fractal dimension)
            hurst = self.hurst_exponent(returns.values)
            
            # Fractal dimension approximation: D = 2 - H
            fractal_dim = 2 - hurst
            
            recent_return = returns.iloc[-5:].sum()
            
            if fractal_dim > self.dimension_threshold:
                # High fractal dimension = more chaotic = mean reversion
                signals[col] = -recent_return * (fractal_dim - 1)
            else:
                # Low fractal dimension = more trending = momentum
                signals[col] = recent_return * (2 - fractal_dim)
        
        return pd.Series(signals)

class MachineLearningEnsemble(BaseAlpha):
    """Ensemble of simple ML-inspired trading strategies."""
    
    def __init__(self, n_estimators: int = 5, feature_window: int = 20):
        super().__init__(
            name=f"ml_ensemble_{n_estimators}est_{feature_window}d",
            description=f"ML ensemble with {n_estimators} estimators"
        )
        self.n_estimators = n_estimators
        self.feature_window = feature_window
    
    def create_features(self, returns):
        """Create technical features from return series."""
        if len(returns) < self.feature_window:
            return np.array([])
        
        features = []
        
        # Momentum features
        features.append(returns.iloc[-5:].sum())  # 5-day momentum
        features.append(returns.iloc[-10:].sum())  # 10-day momentum
        
        # Mean reversion features
        features.append(-returns.iloc[-3:].sum())  # 3-day reversal
        features.append(returns.mean())  # Long-term mean
        
        # Volatility features
        features.append(returns.std())  # Volatility
        features.append(returns.iloc[-5:].std())  # Recent volatility
        
        # Trend features
        if len(returns) >= 10:
            x = np.arange(len(returns[-10:]))
            slope = np.polyfit(x, returns.iloc[-10:].values, 1)[0]
            features.append(slope)
        else:
            features.append(0)
        
        return np.array(features)
    
    def simple_decision_tree(self, features, tree_id):
        """Simple decision tree-like logic."""
        if len(features) < 5:
            return 0
        
        # Different tree logic based on tree_id
        if tree_id == 0:
            # Momentum tree
            if features[0] > 0 and features[4] < 0.02:  # Positive momentum, low vol
                return features[0]
            else:
                return -0.5 * features[0]
        
        elif tree_id == 1:
            # Mean reversion tree
            if abs(features[2]) > 0.01:  # Strong recent move
                return features[2]
            else:
                return 0
        
        elif tree_id == 2:
            # Volatility tree
            if features[4] > features[5]:  # Vol increasing
                return -features[1]  # Fade momentum
            else:
                return features[1]  # Follow momentum
        
        elif tree_id == 3:
            # Trend tree
            if abs(features[6]) > 0.001:  # Strong trend
                return features[6] * 10
            else:
                return -features[0] * 0.3  # Weak trend, slight reversion
        
        else:
            # Ensemble tree
            return 0.2 * (features[0] - features[2] + features[6] * 5)
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.feature_window + 10:
            return pd.Series(dtype=float)
        
        signals = {}
        
        for col in data.columns:
            returns = data[col].iloc[-self.feature_window-10:].dropna()
            if len(returns) < self.feature_window:
                signals[col] = 0
                continue
            
            # Create features
            features = self.create_features(returns)
            if len(features) == 0:
                signals[col] = 0
                continue
            
            # Ensemble prediction
            predictions = []
            for i in range(self.n_estimators):
                pred = self.simple_decision_tree(features, i)
                predictions.append(pred)
            
            # Average predictions
            signals[col] = np.mean(predictions)
        
        return pd.Series(signals)