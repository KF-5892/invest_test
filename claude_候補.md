<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# これは、システムトレードのシステムを作成して運用するプロジェクトです。イメージとしては、izanamiのような、日本株のバックテストができるだけでなく、戦略の立案までAIでできる(Claudeその他)ことを期待します。その前提で、適切なClaude.mdを作成してください

以下が、このプロジェクト用に最適化した `CLAUDE.md` です。イザナミを参考に、AI戦略立案機能を持つ日本株システムトレードシステムとして設計しています。

***

```markdown
# CLAUDE.md — AI日本株システムトレードプロジェクト

## プロジェクト概要

日本株を対象とした**AIネイティブなシステムトレード基盤**の構築プロジェクト。  
イザナミ相当のバックテスト機能に加え、Claudeによる**戦略立案・評価・改善サイクル**をAIで完結させることを目標とする。

### コアコンセプト
- **バックテストエンジン**：Pythonで日本株の過去データを検証（イザナミ互換ロジック）
- **AI戦略アシスタント**：Claudeによるファンダメンタル/テクニカル分析・戦略提案
- **フォワードテスト**：戦略の実運用前検証
- **ポートフォリオ管理**：複数戦略の組み合わせと資金配分

---

## 技術スタック

```

言語         : Python 3.11+
データ操作   : pandas, numpy
テクニカル指標: pandas-ta, TA-Lib
バックテスト : backtesting.py または自作エンジン
株価データ   : yfinance（検証用）/ JPX公式 / kabuステーションAPI（本番）
可視化       : plotly（チャート）, matplotlib（バックテスト結果）
AI連携       : anthropic SDK（Claude API）
DB           : SQLite（ローカル）/ PostgreSQL（本番想定）
設定管理     : python-dotenv
テスト       : pytest

```

### ディレクトリ構成

```

project/
├── CLAUDE.md
├── .env                    \# APIキー等（git管理外）
├── .env.example
├── data/
│   ├── raw/                \# 取得済み株価データ（CSV）
│   ├── processed/          \# 前処理済みデータ
│   └── universe/           \# 銘柄ユニバース定義
├── strategies/             \# 戦略定義ファイル（Python）
│   ├── base_strategy.py
│   ├── mean_reversion/
│   └── momentum/
├── backtest/
│   ├── engine.py           \# バックテストエンジン本体
│   ├── metrics.py          \# 成績指標計算
│   └── reporter.py         \# 結果レポート生成
├── ai_advisor/             \# AI戦略アシスタント
│   ├── strategy_generator.py   \# 戦略立案プロンプト
│   ├── evaluator.py            \# バックテスト結果の評価
│   └── optimizer.py            \# パラメータ最適化提案
├── data_fetcher/
│   ├── price_fetcher.py    \# 株価取得
│   └── fundamental_fetcher.py  \# ファンダメンタルデータ取得
├── tests/
└── notebooks/              \# 分析・実験用Jupyter

```

---

## 開発ルール

### コードスタイル
- 型ヒント（Type Hints）を必ず付与する
- pandas DataFrameのカラム名は**英語スネークケース**統一（例: `close_price`, `volume_ma20`）
- 戦略クラスは必ず `BaseStrategy` を継承する
- バックテスト結果は必ず `BacktestResult` データクラスで返す

### 重要な制約 (**MUST**)
- **未来データ漏洩（ルックアヘッドバイアス）を絶対に防ぐ**。シグナル生成には当日終値以降のデータを使用しない
- 手数料・スリッページは必ずシミュレーションに含める（デフォルト：往復0.2%）
- 売買シグナルは「翌日寄付執行」を前提とする（イザナミ標準）
- バックテスト期間は最低でも**5年以上**のデータで検証する

### 命名規則
```python
# 戦略ファイル名
strategies/mean_reversion/rsi_oversold_v1.py

# 関数名
def generate_buy_signal(df: pd.DataFrame) -> pd.Series: ...
def calculate_position_size(capital: float, risk_ratio: float) -> int: ...

# バックテスト結果変数
result: BacktestResult = backtest.run(strategy, universe, period)
```


---

## バックテスト成績の評価基準

Claudeがバックテスト結果を評価する際は、以下の指標を**すべて**確認すること：


| 指標 | 最低基準 | 優良基準 |
| :-- | :-- | :-- |
| 期待値（1トレードあたり） | > 0 | > 0.5% |
| 勝率 | > 40% | > 50% |
| プロフィットファクター | > 1.2 | > 1.5 |
| 最大ドローダウン | < 30% | < 15% |
| シャープレシオ | > 0.5 | > 1.0 |
| 取引回数（年間） | > 20回 | > 50回 |

**過学習チェック必須**：IS（In-Sample）期間とOOS（Out-of-Sample）期間で成績乖離が30%以上の場合は警告を出す。

---

## AI戦略アシスタントの使い方

### 戦略立案フロー（AIに依頼する手順）

```
1. 市場環境の指定（例：東証プライム・流動性上位500銘柄・逆張り系）
2. Claudeが複数の戦略仮説を提案
3. Pythonで戦略コードを自動生成
4. バックテスト実行
5. Claudeが結果を評価し改善点を提案
6. パラメータ最適化（ウォークフォワード必須）
7. フォワードテスト移行判断
```


### Claudeへのプロンプト作成ガイドライン

戦略評価を依頼する際は以下の情報を必ず含める：

- バックテスト結果（`BacktestResult`をJSON形式で渡す）
- 対象ユニバース（市場・規模・業種）
- 検証期間
- 比較ベンチマーク（例：TOPIX）

---

## よく使うコマンド

```bash
# データ取得
python data_fetcher/price_fetcher.py --start 2010-01-01 --universe topix500

# バックテスト実行
python backtest/engine.py --strategy strategies/mean_reversion/rsi_oversold_v1.py --period 2015-2024

# AI戦略提案
python ai_advisor/strategy_generator.py --theme "出来高急増 逆張り" --universe topix500

# テスト実行
pytest tests/ -v

# バックテスト結果レポート生成
python backtest/reporter.py --result results/latest.json --output reports/
```


---

## データ仕様

### 株価DataFrameの標準スキーマ

```python
# df.columns の標準構成
{
    "date": "datetime64[ns]",       # 日付
    "code": "str",                  # 銘柄コード（例: "7203"）
    "open": "float64",              # 始値（権利調整済）
    "high": "float64",              # 高値
    "low": "float64",               # 安値
    "close": "float64",             # 終値
    "volume": "int64",              # 出来高
    "adj_close": "float64",         # 調整済終値
    "market_cap": "float64",        # 時価総額（オプション）
    "per": "float64",               # PER（オプション）
    "pbr": "float64",               # PBR（オプション）
}
```


---

## テスト方針

- バックテストエンジンの**ルックアヘッドバイアステスト**は必須（`tests/test_no_lookahead.py`）
- 各戦略クラスに対してシグナル生成の**単体テスト**を作成する
- 既知の相場パターン（2008年リーマン、2020年コロナ等）で**ストレステスト**を実施

---

## セキュリティ・運用ルール

- APIキー・証券会社認証情報は`.env`のみに記載し、コードにハードコードしない
- 実際の発注処理は`live_trading/`ディレクトリに分離し、`--dry-run`フラグなしでは実行不可にする
- バックテストと本番ロジックは必ずコードレベルで分離する

---

## コンパクション時の保持事項

セッションのコンパクト時は以下を必ず保持すること：

- 現在検証中の戦略名とパラメータ
- 直近のバックテスト結果サマリー（期待値・PF・最大DD）
- 次のアクション（改善点・未実装タスク）

```

***

このCLAUDE.mdのポイントを補足します。

## 設計の意図

**ルックアヘッドバイアス防止の明示**[^1]
イザナミのようなシステムトレードソフトでは、売買シグナルを翌日寄付執行とするのが標準であり、未来データ漏洩（ルックアヘッドバイアス）の防止ルールをCLAUDE.mdに明記することで、AIがコード生成時に自動的にこの制約を守るようになります 。[^2]

**評価指標の定量化**[^3]
イザナミの売買ルール検証では「期待値・勝率・プロフィットファクター・最大ドローダウン」が特に重要な指標とされており、Claudeが結果を評価する際の基準として数値で定義しています 。[^3]

**AI戦略立案フローの組み込み**[^4]
CLAUDE.mdはClaude Codeがすべてのセッション開始時に自動読み込みする特別なファイルであり、戦略生成→バックテスト→評価→改善というループをClaude自身が自律的に実行できるよう、フローを明示的に記述しています 。[^5][^4]

**コンパクション指示の追加**[^5]
長期プロジェクトでは会話履歴が圧縮されることがあるため、「現在の戦略・結果・次のアクション」を必ず保持するよう指示することで、セッションをまたいでも開発コンテキストが維持されます 。[^5]
<span style="display:none">[^10][^11][^12][^13][^14][^15][^6][^7][^8][^9]</span>

<div align="center">⁂</div>

[^1]: https://note.com/izanami_k/n/n76a947fcc3d1
[^2]: https://www.izanami.jp/v2support/tutorial_step_01.html
[^3]: https://www.izanami.jp/basic_s.html
[^4]: https://institute.sfeir.com/en/claude-code/claude-code-resources/best-practices/
[^5]: https://code.claude.com/docs/ja/best-practices
[^6]: https://www.izanami.jp/v2support/faq_category_00.html
[^7]: https://qiita.com/NT1123/items/b229311f9126d280d756
[^8]: https://note.com/tanakaconsul/n/neee898dc639d
[^9]: https://brian0111.com/mt4-python-integration/
[^10]: https://code.claude.com/docs/en/best-practices
[^11]: https://qiita.com/digitalwrite01/items/e38f5d9566da9c1e1605
[^12]: https://x.com/izanami_info
[^13]: https://zenn.dev/farstep/articles/claude-code-best-practices
[^14]: https://note.com/huge_slug5265/n/ndc64f5e0417e
[^15]: https://www.youtube.com/watch?v=g1yv-zQwoKc```

