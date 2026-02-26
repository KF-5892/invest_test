
"""
================================================================================
日本株システムトレード バックテストコード集
2026年版 - 最新戦略の実装
================================================================================

必要ライブラリ:
pip install pandas numpy yfinance matplotlib scipy

使用方法:
1. 各戦略のクラスをインスタンス化
2. run_backtest()メソッドでバックテスト実行
3. 結果をCSVとグラフで出力
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy import stats

# 日本語フォント設定（環境に応じて調整）
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ================================================================================
# 戦略1: モメンタム戦略（中期順張り）
# ================================================================================
class MomentumStrategy:
    """
    2026年最も有効とされるモメンタム戦略
    - 過去のパフォーマンス上位銘柄を保有
    - 月次リバランス
    - 業績連動型の順張り
    """

    def __init__(self, tickers, lookback_period=60, holding_period=20, 
                 top_n=5, initial_capital=1000000):
        """
        Parameters:
        -----------
        tickers : list
            対象銘柄のティッカーリスト（例: ['7203.T', '6758.T']）
        lookback_period : int
            モメンタム計算期間（営業日）
        holding_period : int
            保有期間（営業日）
        top_n : int
            上位何銘柄を保有するか
        initial_capital : float
            初期資金
        """
        self.tickers = tickers
        self.lookback_period = lookback_period
        self.holding_period = holding_period
        self.top_n = top_n
        self.initial_capital = initial_capital

    def download_data(self, start_date, end_date):
        """株価データをダウンロード"""
        print(f"データダウンロード中: {start_date} から {end_date}")
        data = yf.download(self.tickers, start=start_date, end=end_date)
        return data['Adj Close']

    def calculate_momentum(self, prices):
        """モメンタムスコアを計算（過去N日のリターン）"""
        return (prices / prices.shift(self.lookback_period) - 1) * 100

    def run_backtest(self, start_date='2023-01-01', end_date='2026-02-14'):
        """バックテスト実行"""
        # データ取得
        prices = self.download_data(start_date, end_date)

        # モメンタム計算
        momentum = self.calculate_momentum(prices)

        # ポートフォリオ構築
        portfolio_value = [self.initial_capital]
        holdings = {}
        cash = self.initial_capital

        dates = prices.index[self.lookback_period:]

        for i, date in enumerate(dates):
            if i % self.holding_period == 0:  # リバランス
                # 現在のポジション清算
                if holdings:
                    for ticker, shares in holdings.items():
                        cash += shares * prices.loc[date, ticker]
                    holdings = {}

                # 新規ポジション構築
                current_momentum = momentum.loc[date].dropna()
                top_stocks = current_momentum.nlargest(self.top_n)

                if len(top_stocks) > 0:
                    allocation_per_stock = cash / len(top_stocks)

                    for ticker in top_stocks.index:
                        price = prices.loc[date, ticker]
                        if price > 0:
                            shares = allocation_per_stock / price
                            holdings[ticker] = shares
                            cash -= shares * price

            # ポートフォリオ評価
            portfolio_val = cash
            for ticker, shares in holdings.items():
                portfolio_val += shares * prices.loc[date, ticker]

            portfolio_value.append(portfolio_val)

        # 結果DataFrame作成
        results = pd.DataFrame({
            'Date': [prices.index[self.lookback_period-1]] + list(dates),
            'Portfolio_Value': portfolio_value
        })
        results.set_index('Date', inplace=True)

        # パフォーマンス指標計算
        total_return = (results['Portfolio_Value'].iloc[-1] / self.initial_capital - 1) * 100
        returns = results['Portfolio_Value'].pct_change().dropna()
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        max_drawdown = ((results['Portfolio_Value'].cummax() - results['Portfolio_Value']) / 
                        results['Portfolio_Value'].cummax()).max() * 100

        print("\n" + "="*60)
        print("モメンタム戦略 バックテスト結果")
        print("="*60)
        print(f"期間: {start_date} ～ {end_date}")
        print(f"初期資金: ¥{self.initial_capital:,.0f}")
        print(f"最終資産: ¥{results['Portfolio_Value'].iloc[-1]:,.0f}")
        print(f"総リターン: {total_return:.2f}%")
        print(f"シャープレシオ: {sharpe_ratio:.3f}")
        print(f"最大ドローダウン: {max_drawdown:.2f}%")

        return results


# ================================================================================
# 戦略2: ペアトレード戦略（統計的アービトラージ）
# ================================================================================
class PairsTradingStrategy:
    """
    相関の高い2銘柄の価格差を利用
    市場中立戦略
    """

    def __init__(self, ticker1, ticker2, window=20, entry_z=2.0, 
                 exit_z=0.5, initial_capital=1000000):
        """
        Parameters:
        -----------
        ticker1, ticker2 : str
            ペアとなる2銘柄のティッカー
        window : int
            移動平均・標準偏差の計算期間
        entry_z : float
            エントリーのZスコア閾値
        exit_z : float
            手仕舞いのZスコア閾値
        """
        self.ticker1 = ticker1
        self.ticker2 = ticker2
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.initial_capital = initial_capital

    def run_backtest(self, start_date='2023-01-01', end_date='2026-02-14'):
        """バックテスト実行"""
        # データ取得
        print(f"\nペアトレード: {self.ticker1} vs {self.ticker2}")
        data1 = yf.download(self.ticker1, start=start_date, end=end_date)['Adj Close']
        data2 = yf.download(self.ticker2, start=start_date, end=end_date)['Adj Close']

        # データ結合
        df = pd.DataFrame({'Asset1': data1, 'Asset2': data2}).dropna()

        # スプレッド計算
        df['Spread'] = df['Asset1'] - df['Asset2']
        df['Spread_MA'] = df['Spread'].rolling(window=self.window).mean()
        df['Spread_Std'] = df['Spread'].rolling(window=self.window).std()
        df['Z_Score'] = (df['Spread'] - df['Spread_MA']) / df['Spread_Std']

        # トレードシグナル
        df['Position'] = 0
        df.loc[df['Z_Score'] > self.entry_z, 'Position'] = -1  # ショート
        df.loc[df['Z_Score'] < -self.entry_z, 'Position'] = 1  # ロング
        df.loc[abs(df['Z_Score']) < self.exit_z, 'Position'] = 0  # 手仕舞い

        # ポジション継続
        df['Position'] = df['Position'].replace(0, method='ffill').fillna(0)

        # リターン計算
        df['Strategy_Return'] = df['Position'].shift(1) * (df['Spread'].pct_change())
        df['Cumulative_Return'] = (1 + df['Strategy_Return']).cumprod()
        df['Portfolio_Value'] = self.initial_capital * df['Cumulative_Return']

        # パフォーマンス指標
        total_return = (df['Portfolio_Value'].iloc[-1] / self.initial_capital - 1) * 100
        sharpe = (df['Strategy_Return'].mean() / df['Strategy_Return'].std()) * np.sqrt(252)
        max_dd = ((df['Portfolio_Value'].cummax() - df['Portfolio_Value']) / 
                  df['Portfolio_Value'].cummax()).max() * 100

        print("="*60)
        print("ペアトレード戦略 バックテスト結果")
        print("="*60)
        print(f"総リターン: {total_return:.2f}%")
        print(f"シャープレシオ: {sharpe:.3f}")
        print(f"最大ドローダウン: {max_dd:.2f}%")
        print(f"トレード数: {(df['Position'].diff() != 0).sum()}")

        return df[['Portfolio_Value', 'Z_Score', 'Position']]


# ================================================================================
# 戦略3: 移動平均クロス戦略（トレンドフォロー）
# ================================================================================
class MovingAverageCrossStrategy:
    """
    短期移動平均と長期移動平均のクロスでトレード
    シンプルで堅実な戦略
    """

    def __init__(self, ticker, short_window=25, long_window=75, initial_capital=1000000):
        self.ticker = ticker
        self.short_window = short_window
        self.long_window = long_window
        self.initial_capital = initial_capital

    def run_backtest(self, start_date='2023-01-01', end_date='2026-02-14'):
        """バックテスト実行"""
        print(f"\n移動平均クロス戦略: {self.ticker}")

        # データ取得
        data = yf.download(self.ticker, start=start_date, end=end_date)
        df = pd.DataFrame({'Close': data['Adj Close']})

        # 移動平均計算
        df['SMA_Short'] = df['Close'].rolling(window=self.short_window).mean()
        df['SMA_Long'] = df['Close'].rolling(window=self.long_window).mean()

        # シグナル生成
        df['Signal'] = 0
        df.loc[df['SMA_Short'] > df['SMA_Long'], 'Signal'] = 1
        df.loc[df['SMA_Short'] <= df['SMA_Long'], 'Signal'] = 0

        # ポジション
        df['Position'] = df['Signal'].diff()

        # リターン計算
        df['Returns'] = df['Close'].pct_change()
        df['Strategy_Returns'] = df['Signal'].shift(1) * df['Returns']
        df['Cumulative_Returns'] = (1 + df['Strategy_Returns']).cumprod()
        df['Portfolio_Value'] = self.initial_capital * df['Cumulative_Returns']

        # パフォーマンス指標
        total_return = (df['Portfolio_Value'].iloc[-1] / self.initial_capital - 1) * 100
        sharpe = (df['Strategy_Returns'].mean() / df['Strategy_Returns'].std()) * np.sqrt(252)
        max_dd = ((df['Portfolio_Value'].cummax() - df['Portfolio_Value']) / 
                  df['Portfolio_Value'].cummax()).max() * 100

        print("="*60)
        print("移動平均クロス戦略 バックテスト結果")
        print("="*60)
        print(f"総リターン: {total_return:.2f}%")
        print(f"シャープレシオ: {sharpe:.3f}")
        print(f"最大ドローダウン: {max_dd:.2f}%")

        return df[['Close', 'SMA_Short', 'SMA_Long', 'Portfolio_Value']]


# ================================================================================
# 実行例
# ================================================================================
if __name__ == "__main__":

    print("="*80)
    print("日本株システムトレード バックテスト実行")
    print("="*80)

    # 戦略1: モメンタム戦略
    # 日本の主要銘柄（例）
    momentum_tickers = [
        '7203.T',  # トヨタ
        '6758.T',  # ソニー
        '9984.T',  # ソフトバンクG
        '6861.T',  # キーエンス
        '8306.T',  # 三菱UFJ
    ]

    momentum_strategy = MomentumStrategy(
        tickers=momentum_tickers,
        lookback_period=60,
        holding_period=20,
        top_n=3
    )

    try:
        momentum_results = momentum_strategy.run_backtest(
            start_date='2023-01-01',
            end_date='2026-02-14'
        )
        momentum_results.to_csv('momentum_backtest_results.csv')
        print("\n✓ モメンタム戦略の結果を保存しました")
    except Exception as e:
        print(f"エラー: {e}")


    # 戦略2: ペアトレード
    pairs_strategy = PairsTradingStrategy(
        ticker1='7203.T',  # トヨタ
        ticker2='7267.T',  # ホンダ
        window=20,
        entry_z=2.0
    )

    try:
        pairs_results = pairs_strategy.run_backtest(
            start_date='2023-01-01',
            end_date='2026-02-14'
        )
        pairs_results.to_csv('pairs_trading_backtest_results.csv')
        print("\n✓ ペアトレード戦略の結果を保存しました")
    except Exception as e:
        print(f"エラー: {e}")


    # 戦略3: 移動平均クロス
    ma_strategy = MovingAverageCrossStrategy(
        ticker='1321.T',  # 日経225ETF
        short_window=25,
        long_window=75
    )

    try:
        ma_results = ma_strategy.run_backtest(
            start_date='2023-01-01',
            end_date='2026-02-14'
        )
        ma_results.to_csv('ma_cross_backtest_results.csv')
        print("\n✓ 移動平均クロス戦略の結果を保存しました")
    except Exception as e:
        print(f"エラー: {e}")

    print("\n" + "="*80)
    print("すべてのバックテスト完了")
    print("="*80)
