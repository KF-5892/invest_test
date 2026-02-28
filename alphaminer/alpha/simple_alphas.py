import pandas as pd
import numpy as np
from .base_alpha import BaseAlpha

class GapReversal(BaseAlpha):
    """Gap reversal strategy."""
    
    def __init__(self, gap_threshold: float = 0.02):
        super().__init__(
            name=f"gap_reversal_{gap_threshold}",
            description=f"Gap reversal with {gap_threshold} threshold"
        )
        self.gap_threshold = gap_threshold
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < 2:
            return pd.Series(dtype=float)
        
        # Calculate gap (today vs yesterday)
        gap = data.iloc[-1] - data.iloc[-2]
        
        # Reversal signal for significant gaps
        signals = pd.Series(0.0, index=gap.index)
        signals[gap > self.gap_threshold] = -1.0  # Gap up -> sell
        signals[gap < -self.gap_threshold] = 1.0  # Gap down -> buy
        
        return signals

class SimpleMA(BaseAlpha):
    """Simple moving average crossover."""
    
    def __init__(self, ma_period: int = 10):
        super().__init__(
            name=f"simple_ma_{ma_period}d",
            description=f"Simple MA crossover {ma_period}d"
        )
        self.ma_period = ma_period
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.ma_period:
            return pd.Series(dtype=float)
        
        # Calculate MA of cumulative returns
        cum_returns = data.cumsum()
        ma = cum_returns.rolling(self.ma_period).mean()
        
        # Signal: current vs MA
        current = cum_returns.iloc[-1]
        ma_value = ma.iloc[-1]
        
        return np.sign(current - ma_value)

class VolatilityRank(BaseAlpha):
    """Volatility percentile rank."""
    
    def __init__(self, vol_period: int = 20, rank_period: int = 60):
        super().__init__(
            name=f"vol_rank_{vol_period}d_{rank_period}d",
            description=f"Volatility rank {vol_period}d over {rank_period}d"
        )
        self.vol_period = vol_period
        self.rank_period = rank_period
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.rank_period:
            return pd.Series(dtype=float)
        
        # Calculate current volatility
        current_vol = data.iloc[-self.vol_period:].std()
        
        # Calculate historical volatility ranks
        hist_data = data.iloc[-self.rank_period:]
        vol_series = hist_data.rolling(self.vol_period).std()
        
        # Calculate percentile rank
        ranks = vol_series.rank(pct=True).iloc[-1]
        
        # Mean revert high vol (high rank -> negative signal)
        return 1 - ranks

class PriceVelocity(BaseAlpha):
    """Price velocity (acceleration)."""
    
    def __init__(self, period: int = 5):
        super().__init__(
            name=f"price_velocity_{period}d",
            description=f"Price velocity over {period}d"
        )
        self.period = period
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.period * 2:
            return pd.Series(dtype=float)
        
        # Calculate velocity (change in momentum)
        cum_returns = data.cumsum()
        
        # Recent momentum
        recent_mom = cum_returns.iloc[-self.period:].iloc[-1] - cum_returns.iloc[-self.period:].iloc[0]
        
        # Previous momentum
        prev_mom = cum_returns.iloc[-self.period*2:-self.period].iloc[-1] - cum_returns.iloc[-self.period*2:-self.period].iloc[0]
        
        # Velocity = change in momentum
        return recent_mom - prev_mom

class RollingBeta(BaseAlpha):
    """Rolling beta vs market."""
    
    def __init__(self, beta_period: int = 30):
        super().__init__(
            name=f"rolling_beta_{beta_period}d",
            description=f"Rolling beta {beta_period}d"
        )
        self.beta_period = beta_period
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.beta_period:
            return pd.Series(dtype=float)
        
        recent_data = data.iloc[-self.beta_period:]
        market = recent_data.mean(axis=1)
        
        betas = {}
        for col in recent_data.columns:
            # Calculate beta
            covariance = np.cov(recent_data[col], market)[0,1]
            market_var = np.var(market)
            beta = covariance / market_var if market_var > 0 else 0
            betas[col] = beta
        
        betas_series = pd.Series(betas)
        
        # Favor low beta stocks
        return 2 - betas_series

class TrendConsistency(BaseAlpha):
    """Trend consistency score."""
    
    def __init__(self, period: int = 10):
        super().__init__(
            name=f"trend_consistency_{period}d",
            description=f"Trend consistency over {period}d"
        )
        self.period = period
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.period:
            return pd.Series(dtype=float)
        
        recent_data = data.iloc[-self.period:]
        
        # Count positive days
        positive_days = (recent_data > 0).sum()
        consistency = positive_days / self.period
        
        # Convert to signals (-1 to 1)
        return 2 * consistency - 1

class ReturnSkew(BaseAlpha):
    """Return skewness strategy."""
    
    def __init__(self, period: int = 30):
        super().__init__(
            name=f"return_skew_{period}d",
            description=f"Return skewness over {period}d"
        )
        self.period = period
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.period:
            return pd.Series(dtype=float)
        
        recent_data = data.iloc[-self.period:]
        
        # Calculate skewness
        skewness = recent_data.skew()
        
        # Favor negative skew (fat left tail)
        return -skewness

class DrawdownRecovery(BaseAlpha):
    """Drawdown recovery strategy."""
    
    def __init__(self, lookback: int = 60):
        super().__init__(
            name=f"drawdown_recovery_{lookback}d",
            description=f"Drawdown recovery over {lookback}d"
        )
        self.lookback = lookback
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.lookback:
            return pd.Series(dtype=float)
        
        recent_data = data.iloc[-self.lookback:]
        cum_returns = recent_data.cumsum()
        
        # Calculate max drawdown
        rolling_max = cum_returns.expanding().max()
        drawdown = (cum_returns - rolling_max) / rolling_max.abs()
        current_dd = drawdown.iloc[-1]
        
        # Favor stocks in deep drawdown (recovery play)
        return -current_dd

class VolumePrice(BaseAlpha):
    """Volume-price relationship."""
    
    def __init__(self, period: int = 20):
        super().__init__(
            name=f"volume_price_{period}d",
            description=f"Volume-price relationship over {period}d"
        )
        self.period = period
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.period:
            return pd.Series(dtype=float)
        
        recent_data = data.iloc[-self.period:]
        
        # Proxy volume with absolute returns
        abs_returns = recent_data.abs()
        
        # Current "volume" vs average
        current_vol = abs_returns.iloc[-1]
        avg_vol = abs_returns.mean()
        
        # Volume breakout signal
        vol_ratio = current_vol / avg_vol
        
        # Direction from actual returns
        direction = np.sign(recent_data.iloc[-1])
        
        return vol_ratio * direction

class SeasonalTrend(BaseAlpha):
    """Simple seasonal trend."""
    
    def __init__(self, days_back: int = 252):
        super().__init__(
            name=f"seasonal_trend_{days_back}d",
            description=f"Seasonal trend {days_back}d back"
        )
        self.days_back = days_back
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.days_back + 5:
            return pd.Series(dtype=float)
        
        # Compare current 5-day return to same period last year
        current_5d = data.iloc[-5:].sum()
        
        # Same 5-day period last year (approximately)
        year_ago_5d = data.iloc[-(self.days_back+5):-(self.days_back)].sum()
        
        # Seasonal momentum
        return np.sign(year_ago_5d)

class ReturnDensity(BaseAlpha):
    """Return density around zero."""
    
    def __init__(self, period: int = 30, threshold: float = 0.005):
        super().__init__(
            name=f"return_density_{period}d_{threshold}",
            description=f"Return density {period}d threshold {threshold}"
        )
        self.period = period
        self.threshold = threshold
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.period:
            return pd.Series(dtype=float)
        
        recent_data = data.iloc[-self.period:]
        
        # Count returns near zero
        near_zero = (recent_data.abs() < self.threshold).sum()
        density = near_zero / self.period
        
        # High density suggests compression -> expect breakout
        return density

class MomentumDecay(BaseAlpha):
    """Momentum decay strategy."""
    
    def __init__(self, short: int = 5, long: int = 20):
        super().__init__(
            name=f"momentum_decay_{short}d_{long}d",
            description=f"Momentum decay {short}d vs {long}d"
        )
        self.short = short
        self.long = long
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.long:
            return pd.Series(dtype=float)
        
        # Short and long momentum
        short_mom = data.iloc[-self.short:].sum()
        long_mom = data.iloc[-self.long:].sum() / (self.long / self.short)  # Normalize
        
        # Decay = recent momentum weaker than long-term
        decay = long_mom - short_mom
        
        return decay

class VolBreakout(BaseAlpha):
    """Simple volatility breakout."""
    
    def __init__(self, vol_period: int = 20, multiplier: float = 1.5):
        super().__init__(
            name=f"vol_breakout_{vol_period}d_{multiplier}x",
            description=f"Vol breakout {vol_period}d {multiplier}x"
        )
        self.vol_period = vol_period
        self.multiplier = multiplier
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.vol_period + 1:
            return pd.Series(dtype=float)
        
        # Current vs average volatility
        current_vol = data.iloc[-1].abs()
        avg_vol = data.iloc[-self.vol_period:-1].abs().mean()
        
        # Breakout signal
        breakout = current_vol > (avg_vol * self.multiplier)
        
        # Direction
        direction = np.sign(data.iloc[-1])
        
        return breakout.astype(float) * direction

class MeanReversionSpeed(BaseAlpha):
    """Mean reversion speed."""
    
    def __init__(self, period: int = 15):
        super().__init__(
            name=f"mean_reversion_speed_{period}d",
            description=f"Mean reversion speed {period}d"
        )
        self.period = period
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.period:
            return pd.Series(dtype=float)
        
        recent_data = data.iloc[-self.period:]
        
        # Distance from mean
        mean_return = recent_data.mean()
        current_return = recent_data.iloc[-1]
        distance = current_return - mean_return
        
        # Speed of reversion (how fast it returns to mean)
        # Look at autocorrelation
        autocorr = recent_data.apply(lambda x: x.autocorr(lag=1))
        
        # Lower autocorr = faster mean reversion
        speed = 1 - autocorr.abs()
        
        # Signal strength based on distance and speed
        return -distance * speed

class TailRisk(BaseAlpha):
    """Tail risk strategy."""
    
    def __init__(self, period: int = 60, percentile: float = 0.05):
        super().__init__(
            name=f"tail_risk_{period}d_{percentile}",
            description=f"Tail risk {period}d {percentile} percentile"
        )
        self.period = period
        self.percentile = percentile
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.period:
            return pd.Series(dtype=float)
        
        recent_data = data.iloc[-self.period:]
        
        # Calculate tail risk (extreme negative returns)
        tail_threshold = recent_data.quantile(self.percentile)
        tail_returns = recent_data[recent_data <= tail_threshold]
        
        # Average tail loss
        if len(tail_returns) > 0:
            avg_tail_loss = tail_returns.mean()
        else:
            avg_tail_loss = pd.Series(0.0, index=recent_data.columns)
        
        # Favor stocks with less tail risk
        return -avg_tail_loss

class ReturnStability(BaseAlpha):
    """Return stability measure."""
    
    def __init__(self, period: int = 30):
        super().__init__(
            name=f"return_stability_{period}d",
            description=f"Return stability over {period}d"
        )
        self.period = period
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.period:
            return pd.Series(dtype=float)
        
        recent_data = data.iloc[-self.period:]
        
        # Coefficient of variation (std/mean)
        mean_ret = recent_data.mean()
        std_ret = recent_data.std()
        
        # Avoid division by zero
        cv = std_ret / mean_ret.abs().replace(0, np.nan)
        
        # Lower CV = more stable returns
        stability = 1 / (1 + cv.abs())
        
        return stability

class SimpleMomentum(BaseAlpha):
    """Ultra simple momentum."""
    
    def __init__(self, days: int = 7):
        super().__init__(
            name=f"simple_momentum_{days}d",
            description=f"Simple {days}d momentum"
        )
        self.days = days
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.days:
            return pd.Series(dtype=float)
        
        # Sum of last N days
        return data.iloc[-self.days:].sum()

class NoiseRatio(BaseAlpha):
    """Signal-to-noise ratio."""
    
    def __init__(self, period: int = 20):
        super().__init__(
            name=f"noise_ratio_{period}d",
            description=f"Signal-to-noise ratio {period}d"
        )
        self.period = period
    
    def calculate(self, data: pd.DataFrame, lookback_window: int = 252) -> pd.Series:
        if len(data) < self.period:
            return pd.Series(dtype=float)
        
        recent_data = data.iloc[-self.period:]
        
        # Signal = absolute total return
        signal = recent_data.sum().abs()
        
        # Noise = sum of absolute daily returns
        noise = recent_data.abs().sum()
        
        # Signal-to-noise ratio
        snr = signal / noise.replace(0, np.nan)
        
        # Favor high signal-to-noise
        return snr