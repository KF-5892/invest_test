"""
トレンドフォロー戦略（EMA + ATR）バックテスト

出典: Perplexityレポート (2026年2月27日版) から抽出・改良

注意事項:
- 元レポートのバックテスト結果（PF 12.93、8トレード）は統計的に無意味です。
  8トレードは統計検定の最低ライン（30回）を大幅に下回り、
  PF > 4.0はオーバーフィッティングの警告閾値です。
- 本コードは戦略ロジックの参考実装であり、実運用前に十分なサンプル数
  （最低100トレード以上）でのOut-of-Sample検証が必須です。
- 元レポートではテスト対象銘柄が未記載でした。本コードでは日経225 ETF (1321.T) を
  デフォルトとしています。

必要ライブラリ:
    pip install pandas numpy yfinance matplotlib
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['MS Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class TrendFollowStrategy:
    """
    トレンドフォロー戦略（EMA + ATR）

    - EMAクロスでトレンド方向を判定
    - ATRベースのストップロス/テイクプロフィットで出口管理
    - ロング/ショート両方対応

    Parameters
    ----------
    fast_ema : int
        短期EMA期間（デフォルト: 25）
    slow_ema : int
        長期EMA期間（デフォルト: 75）
    atr_period : int
        ATR計算期間（デフォルト: 14）
    atr_sl_mult : float
        ストップロス倍率（ATRの何倍か、デフォルト: 1.5）
    atr_tp_mult : float
        テイクプロフィット倍率（ATRの何倍か、デフォルト: 3.0）
    commission : float
        片道手数料率（デフォルト: 0.0025 = 0.25%）
    initial_capital : float
        初期資金（デフォルト: 1,000,000）
    """

    def __init__(
        self,
        fast_ema=25,
        slow_ema=75,
        atr_period=14,
        atr_sl_mult=1.5,
        atr_tp_mult=3.0,
        commission=0.0025,
        initial_capital=1_000_000,
    ):
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.atr_period = atr_period
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.commission = commission
        self.initial_capital = initial_capital

    def calculate_indicators(self, df):
        """EMA、ATR、クロスシグナルを計算"""
        out = df.copy()

        out['EMA_Fast'] = out['Close'].ewm(span=self.fast_ema, adjust=False).mean()
        out['EMA_Slow'] = out['Close'].ewm(span=self.slow_ema, adjust=False).mean()

        high_low = out['High'] - out['Low']
        high_close = abs(out['High'] - out['Close'].shift())
        low_close = abs(out['Low'] - out['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        out['ATR'] = tr.rolling(window=self.atr_period).mean()

        # クロス検出
        out['Signal'] = 0
        golden = (out['EMA_Fast'] > out['EMA_Slow']) & (
            out['EMA_Fast'].shift(1) <= out['EMA_Slow'].shift(1)
        )
        dead = (out['EMA_Fast'] < out['EMA_Slow']) & (
            out['EMA_Fast'].shift(1) >= out['EMA_Slow'].shift(1)
        )
        out.loc[golden, 'Signal'] = 1   # ロングエントリー
        out.loc[dead, 'Signal'] = -1    # ショートエントリー

        return out

    def run_backtest(self, ticker='1321.T', start_date='2015-01-01', end_date='2026-02-27'):
        """
        バックテスト実行

        Parameters
        ----------
        ticker : str
            対象銘柄ティッカー
        start_date, end_date : str
            バックテスト期間

        Returns
        -------
        trades : list[dict]
            各トレードの詳細
        equity_curve : pd.Series
            資産推移
        """
        print(f"データ取得中: {ticker} ({start_date} ~ {end_date})")
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"{ticker} のデータ取得に失敗しました")

        # MultiIndex対策
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        df = self.calculate_indicators(data)

        trades = []
        position = 0  # 0: ノーポジ, 1: ロング, -1: ショート
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        entry_date = None

        equity = self.initial_capital
        equity_history = []

        for i in range(len(df)):
            row = df.iloc[i]
            date = df.index[i]

            if pd.isna(row['ATR']):
                equity_history.append(equity)
                continue

            # ポジション保有中: SL/TP判定
            if position != 0:
                if position == 1:  # ロング
                    if row['Low'] <= stop_loss:
                        pnl = (stop_loss - entry_price) / entry_price
                        exit_reason = 'SL'
                        exit_price = stop_loss
                    elif row['High'] >= take_profit:
                        pnl = (take_profit - entry_price) / entry_price
                        exit_reason = 'TP'
                        exit_price = take_profit
                    else:
                        equity_history.append(equity)
                        continue
                else:  # ショート
                    if row['High'] >= stop_loss:
                        pnl = (entry_price - stop_loss) / entry_price
                        exit_reason = 'SL'
                        exit_price = stop_loss
                    elif row['Low'] <= take_profit:
                        pnl = (entry_price - take_profit) / entry_price
                        exit_reason = 'TP'
                        exit_price = take_profit
                    else:
                        equity_history.append(equity)
                        continue

                # 手数料控除
                pnl -= self.commission * 2  # 往復

                trades.append({
                    'entry_date': entry_date,
                    'exit_date': date,
                    'direction': 'Long' if position == 1 else 'Short',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_pct': pnl * 100,
                    'exit_reason': exit_reason,
                })

                equity *= (1 + pnl)
                position = 0

            # 新規エントリーシグナル
            if row['Signal'] == 1 and position == 0:
                position = 1
                entry_price = row['Close']
                entry_date = date
                stop_loss = entry_price - self.atr_sl_mult * row['ATR']
                take_profit = entry_price + self.atr_tp_mult * row['ATR']
            elif row['Signal'] == -1 and position == 0:
                position = -1
                entry_price = row['Close']
                entry_date = date
                stop_loss = entry_price + self.atr_sl_mult * row['ATR']
                take_profit = entry_price - self.atr_tp_mult * row['ATR']

            equity_history.append(equity)

        equity_curve = pd.Series(equity_history, index=df.index[:len(equity_history)])

        return trades, equity_curve, df

    def print_results(self, trades, equity_curve):
        """結果の表示"""
        print("\n" + "=" * 70)
        print("トレンドフォロー戦略（EMA + ATR）バックテスト結果")
        print("=" * 70)

        if len(trades) == 0:
            print("トレードなし")
            return

        trades_df = pd.DataFrame(trades)
        wins = trades_df[trades_df['pnl_pct'] > 0]
        losses = trades_df[trades_df['pnl_pct'] <= 0]

        total_return = (equity_curve.iloc[-1] / self.initial_capital - 1) * 100
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        years = days / 365.25
        annual_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100 if years > 0 else 0

        gross_profit = wins['pnl_pct'].sum() if len(wins) > 0 else 0
        gross_loss = abs(losses['pnl_pct'].sum()) if len(losses) > 0 else 0
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # 最大ドローダウン
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_dd = drawdown.min() * 100

        print(f"期間: {equity_curve.index[0].date()} ~ {equity_curve.index[-1].date()} ({years:.1f}年)")
        print(f"初期資金: ¥{self.initial_capital:,.0f}")
        print(f"最終資産: ¥{equity_curve.iloc[-1]:,.0f}")
        print(f"総リターン: {total_return:+.2f}%")
        print(f"年率リターン: {annual_return:+.2f}%")
        print(f"トレード数: {len(trades)}")
        print(f"勝率: {len(wins) / len(trades) * 100:.1f}% ({len(wins)}勝{len(losses)}敗)")
        print(f"プロフィットファクター: {pf:.2f}")
        print(f"最大ドローダウン: {max_dd:.2f}%")

        # 統計的信頼性の警告
        print("\n--- 統計的信頼性 ---")
        if len(trades) < 30:
            print(f"[警告] トレード数 {len(trades)} は統計的最低ライン(30)未満です。")
            print("        結果は偶然の可能性があり、信頼できません。")
        elif len(trades) < 100:
            print(f"[注意] トレード数 {len(trades)} は基本的な信頼性(100)未満です。")
            print("        慎重な解釈が必要です。")
        else:
            print(f"[OK] トレード数 {len(trades)} は基本的な信頼性基準を満たしています。")

        if pf > 4.0 and len(trades) < 100:
            print(f"[警告] PF {pf:.2f} > 4.0 かつ少ないトレード数はオーバーフィッティングの兆候です。")

        # 個別トレード
        print("\n--- 個別トレード ---")
        for i, t in enumerate(trades, 1):
            print(
                f"  #{i:2d} {t['direction']:5s} "
                f"{t['entry_date'].date()} -> {t['exit_date'].date()} "
                f"[{t['exit_reason']}] "
                f"{t['pnl_pct']:+.2f}%"
            )


def main():
    """実行例"""
    strategy = TrendFollowStrategy(
        fast_ema=25,
        slow_ema=75,
        atr_period=14,
        atr_sl_mult=1.5,
        atr_tp_mult=3.0,
        commission=0.0025,
        initial_capital=1_000_000,
    )

    # 十分な期間でテスト（統計的信頼性のため長期間を推奨）
    trades, equity_curve, df = strategy.run_backtest(
        ticker='1321.T',       # 日経225 ETF
        start_date='2015-01-01',
        end_date='2026-02-27',
    )

    strategy.print_results(trades, equity_curve)


if __name__ == '__main__':
    main()
