# CLAUDE.md — AIネイティブ日本株システムトレード基盤

## プロジェクト概要

AIネイティブな日本株システムトレード基盤。Claudeが戦略提案→バックテスト→評価→改善を自律的に回すことを目指す。

**既存資産: AlphaMiner** — 119クラス/300+戦略インスタンスのアルファリサーチエンジンとVercelダッシュボード（稼働中）。Fama-French 48業種ポートフォリオでの戦略評価基盤。

## ディレクトリ構成

```
project/
├── CLAUDE.md
├── .env                       # APIキー（git管理外）
├── alphaminer/                # アルファリサーチ＋ダッシュボード（稼働中）
│   ├── alpha/                 # 119クラス（BaseAlpha継承）
│   │   ├── base_alpha.py      # BaseAlpha ABC + AlphaEngine
│   │   ├── momentum_alphas.py
│   │   ├── statistical_alphas.py
│   │   ├── enhanced_alphas.py
│   │   ├── simple_alphas.py
│   │   ├── advanced_alphas.py
│   │   ├── new_alphas.py
│   │   ├── innovative_alphas.py
│   │   ├── next_gen_alphas.py
│   │   ├── quantum_alphas.py
│   │   ├── professional_alphas.py
│   │   ├── custom_alphas.py
│   │   └── elite_alphas.py
│   ├── data/                  # Fama-French 48業種日次リターン
│   ├── run_alphas.py          # 全戦略実行→alpha_results.json等を生成
│   ├── generate_source_code.py # AST→source_code.json
│   ├── generate_vercel_data.py # フロントエンド用JSONデータ生成
│   ├── deploy.sh              # run→generate→build→vercel --prod
│   └── vercel-frontend/       # Next.js 14 + Tailwind + Recharts
├── 実装/
│   └── バックテスト/           # 日本株バックテストコード（yfinance使用）
│       ├── japan_trading_backtest_2026.py  # Momentum/Pairs/MA-Cross
│       └── trend_follow_ema_atr.py        # EMA+ATRトレンドフォロー
├── 知識ベース/
│   ├── 01_戦略/               # 戦略ランキング・LLM-Traders論文・PCA残差リターン
│   ├── 02_レポート/           # Perplexityレポート・ファクトチェック
│   ├── 03_データ/             # データソース情報（J-Quants, yfinance等）
│   ├── 04_インフラ/           # システム要件・証券API・2FA自動化
│   └── 05_リサーチ/           # テンバガー研究・AIエージェント動向
├── 参考資料/
├── data/                      # （今後構築）株価データ格納
├── strategies/                # （今後構築）本番戦略
├── backtest/                  # （今後構築）バックテストエンジン
├── ai_advisor/                # （今後構築）AI戦略アシスタント
└── data_fetcher/              # （今後構築）データ取得パイプライン
```

## 技術スタック

| カテゴリ | ツール |
|---------|-------|
| 言語 | Python 3.12+ |
| データ処理 | pandas≥2.0, numpy≥1.24, scipy≥1.10 |
| テクニカル指標 | pandas-ta, TA-Lib |
| ML | LightGBM, XGBoost, CatBoost |
| DL | PyTorch, Transformer/LSTM |
| 日本語NLP | transformers + fugashi + FinDeBERTaV2 |
| バックテスト | backtrader, vectorbt, AlphaMiner自作エンジン |
| データソース | J-Quants API（優先）, yfinance（検証用）, タワースコープ, DataGet2 |
| 証券API | 立花証券e支店API, auカブコムAPI |
| AI連携 | anthropic SDK (Claude API) |
| フロントエンド | Next.js 14 + Tailwind + Recharts（AlphaMiner） |
| DB | SQLite（開発）, PostgreSQL（本番）, BigQuery（大規模分析） |
| インフラ | GCP e2-micro（無料枠）, AWS Windows t3.small（Kabu Station用）, Docker |
| 監視 | Slack Webhook, Grafana |

> **注意**: FinBERTは英語専用。日本語センチメント分析にはFinDeBERTaV2を使うこと。
> **注意**: yfinanceは日本株でデータ欠損あり。本番データはJ-Quants APIを使用する。

## 開発ルール

### コーディング規約
- **型ヒント必須**
- DataFrameカラム名は**英語スネークケース**（例: `close_price`, `volume_ma20`）
- 戦略クラスはBaseAlpha（AlphaMiner）またはBaseStrategy（本番）を継承
- pandas処理は**ベクトル化**を優先（forループ禁止ではないが最終手段）

### バックテスト必須要件
- **ルックアヘッドバイアス防止**: 当日終値以降のデータをシグナル生成に使用禁止。翌日寄付執行が前提
- **手数料・スリッページ**: 往復0.2%をデフォルト設定。省略禁止
- **検証期間**: 最低5年分のデータ
- **ウォークフォワード検証**: IS/OOS乖離30%超は要警告

### 銘柄コード
- 2024年1月以降、英数字コード（例: 132A）が導入済み
- 銘柄コードは常に**文字列型**（str）で管理すること
- J-Quants APIでは `pd.to_numeric(errors='coerce')` でフィルタリング

## 投資戦略方針（2026年版）

### 優先アプローチ
- **モメンタム/トレンドフォロー優先**（2026年市場環境ではモメンタム >> 逆張り）
- ML/DLアンサンブル: GBDT:NN ≈ 6:4（Optiver知見）
- **特徴量エンジニアリング > 複雑なモデル構造**（Optiver知見）

### LLM活用戦略
- マルチプロンプトアンサンブル（日英デュアルプロンプト）
- ボラティリティ連動信頼度調整（高ボラ時は信頼度低下）
- シグナル閾値: prob ≥ 0.6 → Long, prob ≤ 0.4 → Short

### ファクターニュートラル
- PCA残差リターン抽出（主成分等価法・JSAI2024）
- 共通ファクター除去後の個別リターンでロングショート構築

## バックテスト評価基準

| 指標 | 最低基準 | 優良基準 | 注意 |
|------|---------|---------|------|
| 期待値（1トレード） | > 0 | > 0.5% | |
| 勝率 | > 40% | > 50% | |
| プロフィットファクター | > 1.2 | > 1.5 | **PF > 4.0は過学習の赤旗** |
| 最大ドローダウン | < 30% | < 15% | |
| シャープレシオ | > 0.5 | > 1.0 | |
| 年間取引回数 | > 30回 | > 100回 | **< 30回は統計的に無意味** |
| IS/OOS乖離 | < 30% | < 15% | ウォークフォワード必須 |

> **過学習警告**: 少サンプル（<30回）でPF>4.0は統計的に無意味。trend_follow_ema_atr.pyの教訓参照。

## 株価DataFrameスキーマ

```python
{
    "date": "datetime64[ns]",   # インデックスまたはカラム
    "code": "str",              # 銘柄コード（文字列型必須、例: "7203", "132A"）
    "open": "float64",          # 始値（調整済み）
    "high": "float64",          # 高値
    "low": "float64",           # 安値
    "close": "float64",         # 終値
    "volume": "int64",          # 出来高
    "adj_close": "float64",     # 調整後終値
    "market_cap": "float64",    # 時価総額（optional）
    "per": "float64",           # PER（optional）
    "pbr": "float64",           # PBR（optional）
}
```

### データソース制約
- **J-Quants API**: 日足のみ（プレミアムプランで前場/後場の日中足）。分足なし
- **yfinance**: 直近数日分の1分足/5分足あり。日本株はデータ欠損注意
- **タワースコープ**: 全銘柄1分足（Web API、60秒ポーリング）
- **DataGet2**: 1分足OHLCV（API、開発者登録不要）

## 証券会社API・認証

| 証券会社 | OS要件 | クラウド | 備考 |
|---------|--------|---------|------|
| 立花証券e支店 | Linux/Docker | GCP/AWS | HTTPS+JSON。クラウド推奨 |
| 三菱UFJ eスマート（旧カブコム） | Windows必須 | Windows VPS | Kabu Station必須。日次再起動 |
| 楽天RSS | Windows必須 | Windows VPS | MarketSpeed必須 |
| 岡三RSS | Windows必須 | Windows VPS | 岡三ネットトレーダー必須 |

### 2FA自動化
- **カブコム**: IMAP→OTP正規表現抽出→自動入力（2025年6月以降メール2FA必須。5分OTP窓に注意）
- **立花証券**: 2025年7月以降電話認証→Twilio 050番号→Python SIP自動応答

## インフラ・運用

### 日次運用フロー
```
06:00  データ更新バッチ（前日終値・指標再計算）
08:00  証券API認証＋2FA自動化
08:20  プレマーケットスクリーニング＋シグナル生成
09:00  市場オープン→自動発注、30秒〜1分ポーリング、SL/TP監視
11:30  前場終了
12:30  後場開始
15:00  市場クローズ
15:30  日次決算バッチ＋Slack/メール日次レポート
```

### コスト目安
| 環境 | 月額 | 用途 |
|------|------|------|
| GCP e2-micro | 無料枠 | 立花e支店、Docker、日次バッチ |
| AWS t3.small (Windows) | 約¥3,000（週末停止で¥1,000） | Kabu Station / 楽天 / 岡三 |
| GCP Cloud Functions | 従量課金 | 軽量な日次予測→発注 |
| GCE Spot | 低コスト | ML訓練、大規模バックテスト |

## 段階的導入ロードマップ

| Phase | 期間 | 内容 |
|-------|------|------|
| 1 | 1-2週 | バックテスト環境構築（pandas/numpy/yfinance） |
| 2 | 2-4週 | データパイプライン（J-Quants + SQLite/PostgreSQL） |
| 3 | 1-3ヶ月 | ペーパートレード（ライブシグナル、発注なし） |
| 4 | 3ヶ月+ | 小規模実運用（リスク管理付き） |

## セキュリティ・運用ルール

- APIキーは`.env`に格納。コードへのハードコード**厳禁**
- `--dry-run`フラグなしでの実発注**禁止**
- バックテストと本番ロジックのコードレベル分離
- `.env`, `credentials.json`等は`.gitignore`に含めること

## よく使うコマンド

### AlphaMiner
```bash
# 全戦略実行
cd alphaminer && python run_alphas.py

# ソースコード抽出（フロントエンド用）
python generate_source_code.py

# フロントエンドデータ生成
python generate_vercel_data.py

# フルデプロイ（実行→生成→ビルド→Vercel）
./deploy.sh

# フロントエンド開発サーバー
cd vercel-frontend && npm run dev
```

### バックテスト
```bash
# 日本株バックテスト実行
python 実装/バックテスト/japan_trading_backtest_2026.py
python 実装/バックテスト/trend_follow_ema_atr.py

# 今後構築予定
# python data_fetcher/price_fetcher.py --start 2010-01-01 --universe topix500
# python backtest/engine.py --strategy strategies/momentum/v1.py --period 2015-2024
# python ai_advisor/strategy_generator.py --theme "出来高急増 逆張り"
```

## コンパクション時の保持事項

会話が長くなりコンテキストが圧縮される場合、以下を必ず保持すること:

- 現在検証中の戦略名とパラメータ
- 直近のバックテスト結果サマリー（主要指標）
- AlphaMinerリーダーボード上位戦略
- 次のアクション（何をすべきか）
- 発見した問題点・制約事項
