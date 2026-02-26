マケデコ Discord #システム構築チャンネルの会話内容から、システムトレードに応用するための学びを以下にまとめました（2022年9月～2025年9月の履歴）。[[discord](https://discord.com/channels/1001344948597178409/1015195093659041834)]

## 1. 証券会社API選定

## 推奨API環境

- **立花証券e支店API**: クラウドサーバーにデプロイして自動売買可能。Dockerやクラウド（GCP、AWS）で動作可能[[discord](https://discord.com/channels/1001344948597178409/1015195093659041834)]
- **カブコム証券API**: Kabuステーション経由。Windows環境必須だがデイトレード信用取引の金利が0%というメリット[[discord](https://discord.com/channels/1001344948597178409/1015195093659041834)]
- **楽天RSS/岡三RSS**: Windows環境必須。発注には各証券会社のソフトウェアを起動している必要がある[[discord](https://discord.com/channels/1001344948597178409/1015195093659041834)]

## プラットフォーム別の制約

- **Linux/Docker環境**: 立花証券e支店APIが最適。カブコムはKabuステーションが必要でWindows必須[[discord](https://discord.com/channels/1001344948597178409/1015195093659041834)]
- **Windows環境**: VPSならAWS t3.smallで月30USD程度。休日停止で10USD未満も可能[[discord](https://discord.com/channels/1001344948597178409/1015195093659041834)]

## 2. データ取得手段

## 過去データ

- **J-Quants API**: 基本は日足のみ。プレミアムプランで前場後場の価格データ取得可能[[discord](https://discord.com/channels/1001344948597178409/1015195093659041834)]
- **yfinance**: 過去数日分の1分足・5分足が取得可能[[discord](https://discord.com/channels/1001344948597178409/1015195093659041834)]
- **DataGet2**: 1分足データをAPI経由で取得可能。5分足はリサンプリングで作成[[discord](https://discord.com/channels/1001344948597178409/1015195093659041834)]
- **タワースコープ**: Web API対応。全銘柄の1分足を60秒に1回リクエスト可能[[discord](https://discord.com/channels/1001344948597178409/1015195093659041834)]

## データ管理の注意点

- 2024年1月からTokyo Pro Marketで銘柄コードにアルファベットが含まれる銘柄（132A等）が登場。整数前提の処理は要修正[[discord](https://discord.com/channels/1001344948597178409/1015195093659041834)]
- 対処法: `pd.to_numeric(df['Code'], errors='coerce').notnull()`でフィルタリング[[discord](https://discord.com/channels/1001344948597178409/1015195093659041834)]

## 3. インフラ・実行環境

## GPU/機械学習環境

- **Google Colab**: 手軽だが学習1回で1ヶ月分のクレジット消費のケースも[[discord](https://discord.com/channels/1001344948597178409/1015195093659041834)]
- **GCPスポットインスタンス**: Colabより低コスト。ライブラリのバージョン管理も厳密に可能[[discord](https://discord.com/channels/1001344948597178409/1015195093659041834)]
- **Google Batch**: コンテナ化によるバッチジョブ実行に最適[[discord](https://discord.com/channels/1001344948597178409/1015195093659041834)]
- **Vertex AI Workbench**: Notebook作業に適したフルマネージドサービス[[discord](https://discord.com/channels/1001344948597178409/1015195093659041834)]

## データ分析基盤

- **BigQuery**: パブリックデータとして公開すると使いやすいが課金設計が課題[[discord](https://discord.com/channels/1001344948597178409/1015195093659041834)]
- **Connected Sheet/Looker Studio**: 無料で統計処理や線形回帰が可能[[discord](https://discord.com/channels/1001344948597178409/1015195093659041834)]
- **Google Colaboratory**: 無料でPythonによる分析が可能[[discord](https://discord.com/channels/1001344948597178409/1015195093659041834)]
- **Grafana**: ダッシュボード作成。J-Quants API連携プラグインのニーズあり[[discord](https://discord.com/channels/1001344948597178409/1015195093659041834)]

## クラウド運用

- **GCPのロケーション固定**: Colabはロケーション不定でGCS転送に3-400円/回かかるが、GCPでロケーション固定すればコスト削減可能[[discord](https://discord.com/channels/1001344948597178409/1015195093659041834)]

## 4. 認証・セキュリティ対策

## 2段階認証の自動化課題

- **Kabuステーション**: 2025年6月からメール2段階認証が必須。毎朝強制ログアウトあり[[discord](https://discord.com/channels/1001344948597178409/1015195093659041834)]
  - 対策: IMAPでメール受信→正規表現で認証コード抽出→自動入力[[discord](https://discord.com/channels/1001344948597178409/1015195093659041834)]
  - Gmail APIまたはS3転送も選択肢[[discord](https://discord.com/channels/1001344948597178409/1015195093659041834)]
  - 問題点: メール到着が5分の有効期限内に来ないケースあり[[discord](https://discord.com/channels/1001344948597178409/1015195093659041834)]
- **立花証券e支店**: 2025年7月末から電話番号認証が必須[[discord](https://discord.com/channels/1001344948597178409/1015195093659041834)]
  - 対策: TwilioでIP電話（050番号）を契約→Python SIPで自動発信[[discord](https://discord.com/channels/1001344948597178409/1015195093659041834)]
  - コスト: 比較的安価[[discord](https://discord.com/channels/1001344948597178409/1015195093659041834)]

## セッション管理

- API方式はSecret Key方式への移行が望まれる（現状は毎日のログイン処理が必要）[[discord](https://discord.com/channels/1001344948597178409/1015195093659041834)]

## 5. システム設計のモチベーション

## システムトレード採用理由

- **行動経済学的観点**: 人間の感情（落ちるナイフを掴む失敗等）に依存しない取引を実現[[discord](https://discord.com/channels/1001344948597178409/1015195093659041834)]
- **本業との両立**: 平日昼休みに証券アプリを見て衝動的な取引をしてしまう失敗を防止[[discord](https://discord.com/channels/1001344948597178409/1015195093659041834)]

## 6. トレーディングプラットフォーム

## MT5 (MetaTrader 5)

- FX/CFD特化の歴史あるプラットフォーム[[discord](https://discord.com/channels/1001344948597178409/1015195093659041834)]
- スクリプト言語（C++ライク）でバックテストと実運用を同時実行可能[[discord](https://discord.com/channels/1001344948597178409/1015195093659041834)]
- Python経由のAPI操作にも対応[[discord](https://discord.com/channels/1001344948597178409/1015195093659041834)]
- 分散トレーディング、アービトラージ、ロング・ショート戦略等の開発基盤構築に活用可能[[discord](https://discord.com/channels/1001344948597178409/1015195093659041834)]

## 7. 運用上の実践的TIPS

- **Selenium自動化**: ヘッドレスモードでの発注は時間がかかりエラーも多い（非推奨）[[discord](https://discord.com/channels/1001344948597178409/1015195093659041834)]
- **UIAutomation**: Kabuステーションのログイン自動化に利用可能[[discord](https://discord.com/channels/1001344948597178409/1015195093659041834)]
- **デイトレ想定構成**: 前場前に50銘柄リストアップ→30秒ごとに株価取得→条件適合で売買[[discord](https://discord.com/channels/1001344948597178409/1015195093659041834)]
- **API仕様の特殊性**: 立花証券APIは特殊だがクラウドデプロイ可能な点が大きなメリット[[discord](https://discord.com/channels/1001344948597178409/1015195093659041834)]

## 8. コスト最適化

- データ転送費用を抑えるためロケーションを固定[[discord](https://discord.com/channels/1001344948597178409/1015195093659041834)]
- オンデマンドGPUはプリエンプティブル選択でコスト削減[[discord](https://discord.com/channels/1001344948597178409/1015195093659041834)]
- Windows Server運用は休日停止で大幅なコスト削減が可能[[discord](https://discord.com/channels/1001344948597178409/1015195093659041834)]

これらのTIPSは実際の運用者の経験に基づいており、特に認証自動化やインフラ選定において実践的な価値があります。