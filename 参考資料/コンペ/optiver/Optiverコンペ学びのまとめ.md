# Optiverコンペ上位解法から学ぶシステムトレード戦略への応用

## 1. コンペティションの概要

Kaggle「Optiver - Trading at the Close」は、NASDAQ市場のクロージング・クロス（大引けオークション）直前の**10分間**における株価変動を予測するコンペティションである。[^1][^2][^3]

- **データ**: 200銘柄 × 481日分、需給（Bid/Ask）・価格・出来高など17列の板情報[^3]
- **ターゲット**: 60秒後のスペシフィックリターン（個別銘柄リターン − 加重平均リターン）[^3]
- **評価指標**: MAE（平均絶対誤差）[^1][^3]
- **特徴**: ターゲットの分布は裾が広く、極端な値を含む。インバランス系の特徴量がターゲットと高い相関を示した[^4]

マケデコ勉強会（2024年5月23日開催）では、西本氏（データ基礎解析＋1位解法分析）、tonic氏（89位・銀メダル解法）、richwomanbtc氏（上位解法モデル比較）の3名が発表を行った。[^4]

***

## 2. 上位解法の共通パターン

上位解法には以下の共通要素が存在した。[^2][^4]

| 要素 | 内容 | 採用例 |
|---|---|---|
| **GBDTとNNのアンサンブル** | 決定木系モデルとニューラルネットの組み合わせ | 1位: CatBoost+GRU+Transformer、89位: LightGBM+Transformer |
| **オンライン学習** | 評価期間中のデータで定期的にモデルを再学習 | 1位: 12日ごとに再学習 |
| **時間帯グループ化特徴量** | 秒数を区分し、区分内での統計量を特徴量化 | 1位: 0-300秒/300-480秒/480秒以上の3群 |
| **ポストプロセッシング** | 予測値の合計を0に補正（マーケットニュートラル化） | 1位、9位 |

***

## 3. 特徴量エンジニアリング ― システムトレードへの応用

### 3-1. インバランス特徴量の有効性

コンペで最も重要だったのは**需給インバランス系の特徴量**であった。具体的には以下が高い予測力を持つ。[^3][^4]

- **liquidity_imbalance** = (bid_size − ask_size) / (bid_size + ask_size)
- **market_urgency** = (ask_price − bid_price) × liquidity_imbalance
- **size_imbalance** = bid_size / ask_size

これらはピアソン相関係数0.10超とかなり高い相関を示した。学術研究でも、板のインバランスは短期的な価格変動の強力な予測因子であり、特にHFT環境下でその有効性が確認されている。[^5][^6][^3]

**▶ システムトレードへの応用:**
リアルタイムの板情報から、bid/askのサイズ比やインバランス指標を計算し、短期的な方向性シグナルとして活用できる。特にスキャルピングやマーケットメイキング戦略において、エントリー/エグジットのタイミング判断に直結する。

### 3-2. 時間帯グループ化（Magic Features）

1位解法のHYD氏が用いた「Magic Features」は、コンペ最大の差別化要因であった。[^4][^3]

**グループ化の軸:**

1. **時間内グループ化（同一銘柄・秒群間）**: 0-300秒/300-480秒/480秒以上の3つのグループに分割し、各グループ内で「初期値との比率」「移動平均との比率」を算出[^3]
2. **クロスセクショナルグループ化（同一秒・銘柄間）**: 同一タイムスタンプ内の全銘柄について「平均値との比率」「ランクの百分位数」を算出[^3]

この分割は、大引けオークションにおけるmatched/unmatched sizeやbid/ask sizeの挙動が時間帯によって大きく異なることに基づいている。[^3]

**▶ システムトレードへの応用:**
市場のセッション内における時間帯別の挙動パターン（例：前場寄付き直後、昼休み前、大引け前など）を認識し、**時間帯ごとに異なるモデルやパラメータを適用する**設計が有効。また、同一時刻における銘柄間の相対的な位置づけ（ランク・偏差）を特徴量として取り込むことで、セクターローテーションやペアトレード戦略のシグナル精度を向上できる。

### 3-3. ラグ・累積・統計集約特徴量

7位解法ベースの特徴量体系は以下の階層構造を持つ。[^2]

- **生特徴量**: 元データそのまま
- **ベース特徴量**: spread、volume、imbalance_ratio等の基本指標
- **ラグ特徴量**: 過去値（shift/diff/pct_change）
- **累積特徴量**: 各種サイズ・インバランスのrolling集約
- **中央値乖離特徴量**: date_id × seconds_in_bucketでグループ化し、中央値からの乖離
- **Global features**: stock_idごとの統計量（median, std, ptp等）

**▶ システムトレードへの応用:**
特徴量設計は「シンプルなものの組み合わせ」が最も効果的であった。複雑な指標よりも、基本的なspread・volume・imbalanceから派生する特徴量を体系的に生成し、モデルに選択させるアプローチが実運用でも再現性が高い。

***

## 4. モデル選択とアンサンブル戦略

### 4-1. GBDT vs NN の性能比較

richwomanbtc氏の再現実験結果は以下の通り。[^2]

| モデル | スコア | スコア(PP適用後) |
|---|---|---|
| LightGBM | 5.83807 | 5.83807 |
| XGBoost | 5.84004 | 5.84004 |
| CatBoost | 5.85796 | 5.85791 |
| CNN | 5.86448 | - |
| GRU | 5.86424 | - |
| LSTM | 5.86781 | - |
| Transformer | 5.87409 | - |

**単体ではLightGBMが最も高精度**であり、GBDT系がNNよりも扱いやすく精度が出やすい傾向が確認された。[^2][^4]

### 4-2. アンサンブルの威力

ただし、上位解法ではGBDTとNNの**アンサンブルが必須**であった。[^4]

- **1位**: CatBoost(0.5) + GRU(0.3) + Transformer(0.2)[^2][^3]
- **89位（tonic氏）**: LightGBM(0.61) + Transformer(0.39)[^1]
- **実験**: LGB×0.6 + XGB×0.3 + CAT×0.1 → スコア5.83871（わずかに改善）[^2]

tonic氏の分析では、LightGBMとTransformerの予測値の散布図が**かなり異なる傾向**を示しており、予測特性の多様性がアンサンブル効果の源泉であった。[^1]

**▶ システムトレードへの応用:**
- GBDTは欠損値に強く、特徴量の重要度が解釈しやすいため**メインモデル**として適する
- NNは銘柄間のクロスセクショナルな関係性（Transformer）や時系列の依存構造（GRU/LSTM）を捉えるため**補助モデル**として有効
- 実運用ではGBDT：NN = 6：4程度のブレンドが安定的[^1]
- 学術研究でもGBDT＋LSTM等のハイブリッドアンサンブルが単体モデルに対し10-15%の精度改善を達成している[^7]

### 4-3. Transformerの活用方法

tonic氏は**スペシフィックリターン予測のためにTransformer Encoder**を使用した。全銘柄の情報をまとめて入力し、Attention機構で銘柄間の関連性を学習させるアプローチである。[^1]

1位のHYD氏はGRUで**時系列方向**の情報を、Transformerで**銘柄間（クロスセクショナル）方向**の情報を学習させる役割分担を行っていた。[^3]

**▶ システムトレードへの応用:**
複数銘柄を同時に予測するポートフォリオ戦略（ロング・ショート戦略等）では、Transformerの**Self-Attention機構**が銘柄間の相互依存関係を自動的に学習する点が強力。セクター間のシフトや相関変動を動的に捉えることが期待できる。

***

## 5. オンライン学習（Adaptive Learning）

### 5-1. なぜオンライン学習が必要だったか

西本氏の分析により、相関係数がdate_idの進行とともに**低下する傾向**が確認された。つまり、学習データの古い期間の特徴量はターゲットとの相関が徐々に弱まっていく（ドメインシフト）。[^3]

tonic氏も最大の敗因として**オンライン学習の欠如**を挙げており、学習データとテストデータの乖離に対応しきれなかったと分析している。[^4][^1]

### 5-2. 1位解法の実装

1位のHYD氏は**12日ごと（計5回）にモデルを追加学習**させた。これにより、評価期間中の市場レジームの変化に適応できた。[^4][^3]

**▶ システムトレードへの応用:**
金融市場は非定常であり、特徴量とリターンの関係は時間とともに変化する。実運用では以下のアプローチが有効：[^8][^9]

1. **定期再学習**: 一定間隔（1-2週間ごと）でモデルを最新データで再学習
2. **直近データへの重み付け**: 9位解法のように最新45日間のデータに1.5倍の重みを設定する手法[^2]
3. **レジーム検知**: 相関係数やモデル精度のモニタリングにより、再学習タイミングを動的に判断
4. **ウォームスタート**: 既存モデルのパラメータを初期値として追加学習（CatBoost/LightGBMで容易に実装可能）

***

## 6. ポストプロセッシング ― マーケットニュートラル補正

### 6-1. 予測値の合計を0にする補正

1位解法のポストプロセッシングは以下のロジック。[^2]

```
prediction["target"] = (
    prediction["target"]
    - (prediction["target"] * prediction["stock_weights"]).sum()
    / prediction["stock_weights"].sum()
)
```

これは、全銘柄の加重予測値の合計がゼロになるように補正する処理である。スペシフィックリターンの定義上、市場全体のリターンを差し引いた残差を予測しているため、合計がゼロであるべきという問題の構造を反映している。[^4][^2]

**▶ システムトレードへの応用:**
- **マーケットニュートラル戦略**では、ロング・ショートのポジションが市場リスクを打ち消すように構成する必要がある。予測値の合計をゼロに補正するロジックは、ポートフォリオ構築時のドルニュートラル/ベータニュートラル制約と直接対応する。[^10][^11]
- 予測値をそのまま使うのではなく、**市場全体の方向性バイアスを除去した上で銘柄選択に使う**という発想は、ファクター投資やスタットアーブ戦略の基本原則と一致する。

***

## 7. 交差検証（CV）の設計

### 7-1. 時系列分割の重要性

richwomanbtc氏の考察によれば、**単純な時間による2分割CV**（400日train/81日valid）で十分であった。5分割のKFoldはLB（リーダーボード）との相関があまりなさそうであった。[^2]

tonic氏は8Fold CVを採用し、MAEとCV間の安定性（stable_loss）をモニタリングしていた。[^1]

**▶ システムトレードへの応用:**
- 金融時系列では**ルックアヘッドバイアス**を避けるため、時系列ベースの分割が必須[^12]
- 実運用では**ウォークフォワード検証**（学習→検証→学習期間をスライド）が最も信頼性が高い
- LB（実環境）との相関が高いCV設計を最初に見つけることが、以降の実験の信頼性を左右する[^2]
- CVの安定性（分散が小さいこと）は、実運用での**ロバスト性**の代理指標として有用

***

## 8. 失敗から学ぶ ― 効かなかった手法

以下のアプローチは効果がなかった。[^1]

| 失敗した手法 | 理由 |
|---|---|
| **2-stage予測（符号＋絶対値分離）** | LightGBMと予測傾向が似通い、アンサンブルに寄与せず |
| **Rank Gauss変換** | アンサンブルに寄与せず |
| **ボラティリティ予測による補正** | 効果なし |

**▶ システムトレードへの応用:**
- 予測を分解する手法は、**分解の各ステージが十分に異なる情報を捉える場合**にのみ有効
- アンサンブルの効果は「予測特性の多様性」に依存するため、似た手法を複数組み合わせても効果は限定的
- ボラティリティ予測による補正が効かなかった点は、**ボラティリティ自体の予測が難しい**ことを示唆しており、実運用でもボラティリティのレジームに応じたポジションサイジングの方が、予測値補正よりも実用的

***

## 9. 実運用への統合 ― システムトレード戦略設計の指針

上記の学びを統合すると、以下のシステムトレード戦略フレームワークが導かれる。

### Phase 1: データ基盤とパイプライン構築
- tonic氏のように「前処理→特徴量計算→学習→予測→後処理→評価」の**パイプラインをパッケージ化**する[^1]
- 実験の高速化・正確性が後続の試行錯誤の質を決定する

### Phase 2: 特徴量設計
- 板情報からインバランス系特徴量を体系的に生成
- 時間帯グループ化により市場マイクロストラクチャの構造を反映
- 銘柄間のクロスセクショナル特徴量（ランク、平均乖離）を追加
- Global features（銘柄ごとの統計量）でコンテキストを付与

### Phase 3: モデル構築
- **メインモデル**: LightGBM（高精度・高速・欠損値に強い）
- **補助モデル**: Transformer（銘柄間関係）+ GRU/LSTM（時系列依存）
- **アンサンブル**: OOF最適化でブレンド比率を決定（GBDT:NN ≈ 6:4が目安）

### Phase 4: 運用・適応
- **定期再学習**（1-2週間サイクル）による市場レジーム変化への適応
- 直近データへの重み付けによるドメインシフト軽減
- 予測値のマーケットニュートラル補正（ロング・ショート制約の適用）

### Phase 5: 監視・改善
- CV安定性と実運用パフォーマンスの乖離をモニタリング
- Feature importanceの時間推移を追跡し、特徴量の陳腐化を検知
- 相関係数の推移からオンライン学習の頻度を動的に調整

***

## 10. まとめ ― 核心的な学び

このコンペティションから得られる、システムトレードに直結する核心的な学びは以下の5点である。

1. **「特徴量がすべて」である**: モデルの精度改善は、アーキテクチャの工夫よりも、問題の構造を反映した特徴量設計によるところが大きい[^12][^4]
2. **市場は非定常であり、適応が必須**: オンライン学習の有無が上位と中位を分ける最大の要因であった[^4][^1]
3. **予測特性の多様性がアンサンブルの価値を生む**: GBDTとNNの組み合わせは、単なる精度向上ではなく、異なる視点からの予測が安定性をもたらす[^1][^2]
4. **ポストプロセッシングは戦略の一部**: 予測値の補正（ニュートラル化）は、問題の数学的構造を活かしたリスク管理手法として実運用に直結する[^4][^2]
5. **信頼できるCVの設計が全ての基盤**: 実験の方向性を正しく導くCV設計なくして、特徴量もモデルも意味をなさない[^1][^2]

---

## References

1. [OptiverCan-Zhan-Ji-_Yin-metaruJie-Fa.pdf](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/41285876/21e34422-cc0a-41e6-9606-811ae4b522e4/OptiverCan-Zhan-Ji-_Yin-metaruJie-Fa.pdf?AWSAccessKeyId=ASIA2F3EMEYEQTYUCFT5&Signature=rT6Yxbh%2BnU3AML7g18mcW%2FNMlyU%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEDwaCXVzLWVhc3QtMSJHMEUCIDsltygbPMVOzVMF7HCOR8OWsN44j296NtyMDn96jaG1AiEAtdVn2rZ75v0hqVJnk4dRAmHR8KNGbePgdguhvNBEV6Qq8wQIBBABGgw2OTk3NTMzMDk3MDUiDNDKS31wvP52s1RTgyrQBON769HlIYOc0qnkRY%2Frmc%2FWVqw4V4fB88aN3ryDkUxVOVs3FBcxrdMN4XT6MTXbANWtJKAjbF4LCYCzlHja%2BUbKHd5wkg68g7MHbdxaRbBdHB8%2BvdR0zmDJGoUb6bo8iDI0X5YTpZ6F%2B9JBXElA5zywOaVYzSf0QR1UqcdI7SvZPgCrpOYgyySiAcn%2Bs7EpLZJX1qizK9Lw1hexIdLH9y7hzzjuhPW4FLqhNyYU4mo%2B5hyL%2BMgvMrSBPyJUZmE3B2iWuwRe%2FWjNBrwM6zTYj%2B9XdrdPwH%2B8NFDoj2vHT3H%2Fl3AZkCcJpwVUgGgD02rp09qacEOM5iIp0tdL1cmm6qPzBSGpWmGLK%2FhBVPA6kA%2F8DtOxvT%2B%2BdcwyBlW8LksEaodGGYsrlwzDa0UQliH8BppdNWFrg8oFDXdWDSqigEFN2JfNQUKR2tgIxtHLioKvjxzP6cmGucWRl7y4g7fLJxtnsJhC13WGltQZt6nqCRiEDETn5%2B719KinMsawhoZNTChNdKzclQStBGj2sx1nzLnZ7nAJ1Z7NEvpoE8kaZYjMtD%2FTF4kst8AqaDJploTzuVIOqzW2Fu0HdvlghoS%2FfT4mBZukm55kwug%2BifcZK24h1DXQZXl7TfksW6nUgFZhnDHj1ZK5TgwdMJGJuq%2FsCTIKYwMfEQLf1LwUzafFWVjwLvoVOeTBdxy2xPR9CJKbkwAewgMDIj19nY7BeqWiMM7EuW4c%2FUYEhkD4RYf9kV3gljFgnrIlOsctNotHnjvXncREw62P%2BLzz7vegK%2FA0ugwwyrDBzAY6mAEOWfObM7w48iUJFWv8ouuxb%2BQ%2BoNFGl6CAS3ECs3f7pihX2bTzQDGMdmQxpKegA6ezuqDUo2H%2BJfClA0ZfJxNHis62wu7nNpkogH9vDKLfdJp2t5QtltuSnpTO6RF15bkMXMGxuLOdbkFUCpkZ6KZ%2Bmt0ingOfRl63x57J96JhG6IlRRRUK8k9ClaZXShiExu9dJQIa0%2FoZQ%3D%3D&Expires=1771071737) - Optiver 参戦記& 銀メダル解法
tonic
2024/05/23

1. 自己紹介
tonic（＠tonic3561）
 フリーランスDS
 Kaggler

目次
1. ざっくり解法
2. ...

2. [slide.pdf](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/41285876/9d8ce460-4f28-41af-832f-3bea00eda629/slide.pdf?AWSAccessKeyId=ASIA2F3EMEYEQTYUCFT5&Signature=OmTpSer4yqPtRq9AWDdyFPyvsew%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEDwaCXVzLWVhc3QtMSJHMEUCIDsltygbPMVOzVMF7HCOR8OWsN44j296NtyMDn96jaG1AiEAtdVn2rZ75v0hqVJnk4dRAmHR8KNGbePgdguhvNBEV6Qq8wQIBBABGgw2OTk3NTMzMDk3MDUiDNDKS31wvP52s1RTgyrQBON769HlIYOc0qnkRY%2Frmc%2FWVqw4V4fB88aN3ryDkUxVOVs3FBcxrdMN4XT6MTXbANWtJKAjbF4LCYCzlHja%2BUbKHd5wkg68g7MHbdxaRbBdHB8%2BvdR0zmDJGoUb6bo8iDI0X5YTpZ6F%2B9JBXElA5zywOaVYzSf0QR1UqcdI7SvZPgCrpOYgyySiAcn%2Bs7EpLZJX1qizK9Lw1hexIdLH9y7hzzjuhPW4FLqhNyYU4mo%2B5hyL%2BMgvMrSBPyJUZmE3B2iWuwRe%2FWjNBrwM6zTYj%2B9XdrdPwH%2B8NFDoj2vHT3H%2Fl3AZkCcJpwVUgGgD02rp09qacEOM5iIp0tdL1cmm6qPzBSGpWmGLK%2FhBVPA6kA%2F8DtOxvT%2B%2BdcwyBlW8LksEaodGGYsrlwzDa0UQliH8BppdNWFrg8oFDXdWDSqigEFN2JfNQUKR2tgIxtHLioKvjxzP6cmGucWRl7y4g7fLJxtnsJhC13WGltQZt6nqCRiEDETn5%2B719KinMsawhoZNTChNdKzclQStBGj2sx1nzLnZ7nAJ1Z7NEvpoE8kaZYjMtD%2FTF4kst8AqaDJploTzuVIOqzW2Fu0HdvlghoS%2FfT4mBZukm55kwug%2BifcZK24h1DXQZXl7TfksW6nUgFZhnDHj1ZK5TgwdMJGJuq%2FsCTIKYwMfEQLf1LwUzafFWVjwLvoVOeTBdxy2xPR9CJKbkwAewgMDIj19nY7BeqWiMM7EuW4c%2FUYEhkD4RYf9kV3gljFgnrIlOsctNotHnjvXncREw62P%2BLzz7vegK%2FA0ugwwyrDBzAY6mAEOWfObM7w48iUJFWv8ouuxb%2BQ%2BoNFGl6CAS3ECs3f7pihX2bTzQDGMdmQxpKegA6ezuqDUo2H%2BJfClA0ZfJxNHis62wu7nNpkogH9vDKLfdJp2t5QtltuSnpTO6RF15bkMXMGxuLOdbkFUCpkZ6KZ%2Bmt0ingOfRl63x57J96JhG6IlRRRUK8k9ClaZXShiExu9dJQIa0%2FoZQ%3D%3D&Expires=1771071737) - Optiver trading at the close 上位解法まとめ
 @richwomanbtc
1

概要
上位解法まとめ
特に7th solutionモデルに着目
Kaggle初心者目線
K...

3. [docswell-K6YQ3E.pdf](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/41285876/c52db49d-6411-445a-b028-eb667f3365ec/docswell-K6YQ3E.pdf?AWSAccessKeyId=ASIA2F3EMEYEQTYUCFT5&Signature=zh%2FrOC2bM8jFCitmDkI3kUzZvC8%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEDwaCXVzLWVhc3QtMSJHMEUCIDsltygbPMVOzVMF7HCOR8OWsN44j296NtyMDn96jaG1AiEAtdVn2rZ75v0hqVJnk4dRAmHR8KNGbePgdguhvNBEV6Qq8wQIBBABGgw2OTk3NTMzMDk3MDUiDNDKS31wvP52s1RTgyrQBON769HlIYOc0qnkRY%2Frmc%2FWVqw4V4fB88aN3ryDkUxVOVs3FBcxrdMN4XT6MTXbANWtJKAjbF4LCYCzlHja%2BUbKHd5wkg68g7MHbdxaRbBdHB8%2BvdR0zmDJGoUb6bo8iDI0X5YTpZ6F%2B9JBXElA5zywOaVYzSf0QR1UqcdI7SvZPgCrpOYgyySiAcn%2Bs7EpLZJX1qizK9Lw1hexIdLH9y7hzzjuhPW4FLqhNyYU4mo%2B5hyL%2BMgvMrSBPyJUZmE3B2iWuwRe%2FWjNBrwM6zTYj%2B9XdrdPwH%2B8NFDoj2vHT3H%2Fl3AZkCcJpwVUgGgD02rp09qacEOM5iIp0tdL1cmm6qPzBSGpWmGLK%2FhBVPA6kA%2F8DtOxvT%2B%2BdcwyBlW8LksEaodGGYsrlwzDa0UQliH8BppdNWFrg8oFDXdWDSqigEFN2JfNQUKR2tgIxtHLioKvjxzP6cmGucWRl7y4g7fLJxtnsJhC13WGltQZt6nqCRiEDETn5%2B719KinMsawhoZNTChNdKzclQStBGj2sx1nzLnZ7nAJ1Z7NEvpoE8kaZYjMtD%2FTF4kst8AqaDJploTzuVIOqzW2Fu0HdvlghoS%2FfT4mBZukm55kwug%2BifcZK24h1DXQZXl7TfksW6nUgFZhnDHj1ZK5TgwdMJGJuq%2FsCTIKYwMfEQLf1LwUzafFWVjwLvoVOeTBdxy2xPR9CJKbkwAewgMDIj19nY7BeqWiMM7EuW4c%2FUYEhkD4RYf9kV3gljFgnrIlOsctNotHnjvXncREw62P%2BLzz7vegK%2FA0ugwwyrDBzAY6mAEOWfObM7w48iUJFWv8ouuxb%2BQ%2BoNFGl6CAS3ECs3f7pihX2bTzQDGMdmQxpKegA6ezuqDUo2H%2BJfClA0ZfJxNHis62wu7nNpkogH9vDKLfdJp2t5QtltuSnpTO6RF15bkMXMGxuLOdbkFUCpkZ6KZ%2Bmt0ingOfRl63x57J96JhG6IlRRRUK8k9ClaZXShiExu9dJQIa0%2FoZQ%3D%3D&Expires=1771071737) - マケデコ勉強会 
2024/05/23 
 
Optiver 2023 勉強会 
データの基礎的な解析＋1st Solution

-
Yuichiro Nishimoto 
-
Twitter: @...

4. [optiverkonheYi-Shi-Lu.markdown](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/41285876/0f253ba4-a643-4319-88a1-64ec4bab3695/optiverkonheYi-Shi-Lu.markdown?AWSAccessKeyId=ASIA2F3EMEYEQTYUCFT5&Signature=WRpakPz24iLGHeE2H7rFwJo6VhM%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEDwaCXVzLWVhc3QtMSJHMEUCIDsltygbPMVOzVMF7HCOR8OWsN44j296NtyMDn96jaG1AiEAtdVn2rZ75v0hqVJnk4dRAmHR8KNGbePgdguhvNBEV6Qq8wQIBBABGgw2OTk3NTMzMDk3MDUiDNDKS31wvP52s1RTgyrQBON769HlIYOc0qnkRY%2Frmc%2FWVqw4V4fB88aN3ryDkUxVOVs3FBcxrdMN4XT6MTXbANWtJKAjbF4LCYCzlHja%2BUbKHd5wkg68g7MHbdxaRbBdHB8%2BvdR0zmDJGoUb6bo8iDI0X5YTpZ6F%2B9JBXElA5zywOaVYzSf0QR1UqcdI7SvZPgCrpOYgyySiAcn%2Bs7EpLZJX1qizK9Lw1hexIdLH9y7hzzjuhPW4FLqhNyYU4mo%2B5hyL%2BMgvMrSBPyJUZmE3B2iWuwRe%2FWjNBrwM6zTYj%2B9XdrdPwH%2B8NFDoj2vHT3H%2Fl3AZkCcJpwVUgGgD02rp09qacEOM5iIp0tdL1cmm6qPzBSGpWmGLK%2FhBVPA6kA%2F8DtOxvT%2B%2BdcwyBlW8LksEaodGGYsrlwzDa0UQliH8BppdNWFrg8oFDXdWDSqigEFN2JfNQUKR2tgIxtHLioKvjxzP6cmGucWRl7y4g7fLJxtnsJhC13WGltQZt6nqCRiEDETn5%2B719KinMsawhoZNTChNdKzclQStBGj2sx1nzLnZ7nAJ1Z7NEvpoE8kaZYjMtD%2FTF4kst8AqaDJploTzuVIOqzW2Fu0HdvlghoS%2FfT4mBZukm55kwug%2BifcZK24h1DXQZXl7TfksW6nUgFZhnDHj1ZK5TgwdMJGJuq%2FsCTIKYwMfEQLf1LwUzafFWVjwLvoVOeTBdxy2xPR9CJKbkwAewgMDIj19nY7BeqWiMM7EuW4c%2FUYEhkD4RYf9kV3gljFgnrIlOsctNotHnjvXncREw62P%2BLzz7vegK%2FA0ugwwyrDBzAY6mAEOWfObM7w48iUJFWv8ouuxb%2BQ%2BoNFGl6CAS3ECs3f7pihX2bTzQDGMdmQxpKegA6ezuqDUo2H%2BJfClA0ZfJxNHis62wu7nNpkogH9vDKLfdJp2t5QtltuSnpTO6RF15bkMXMGxuLOdbkFUCpkZ6KZ%2Bmt0ingOfRl63x57J96JhG6IlRRRUK8k9ClaZXShiExu9dJQIa0%2FoZQ%3D%3D&Expires=1771071737) - マケデコ「Optiverコンペ Kaggle上位解法勉強会」議事録

**開催日時**: 2024年5月23日 19:30〜 **主催**: マケデコ (Market API Developer Co...

5. [Leveraging Limit Order Book Imbalances for Profitable Trading](https://electronictradinghub.com/leveraging-limit-order-book-imbalances-for-profitable-trading-a-deep-dive-into-recent-research-and-practical-tools/) - This article has taken a deep dive into the concept of Limit Order Book (LOB) imbalances and their p...

6. [Impact of High‐Frequency Trading with an Order Book Imbalance ...](https://onlinelibrary.wiley.com/doi/10.1155/2023/3996948) - In the present study, we analysed the impacts of HFT taking into account the correlation between ord...

7. [Gradient Boosting Decision Tree with LSTM for Investment ...](https://arxiv.org/html/2505.23084v1) - This paper proposes a hybrid modeling framework that synergistically integrates LSTM (Long Short-Ter...

8. [Machine Learning in Financial Markets: Applications, Effectiveness ...](https://www.subex.com/blog/machine-learning-in-financial-markets-applications-effectiveness-and-limitations/) - Adaptive algorithms allow machine learning models to adjust and learn from new patterns and market c...

9. [[PDF] Online Adaptive Machine Learning Based Algorithm for Implied ...](https://arxiv.org/pdf/1706.01833.pdf) - A handful of financial applications using adaptive machine learning models have been recently develo...

10. [Market Neutral Strategies](https://hedgenordic.com/wp-content/uploads/2015/12/MarketNeutralReport_2015-1.pdf) - A dollar neutral strategy has zero net investment (i.e., equal dollar amounts in long and short posi...

11. [Market Neutral Strategy | Definition + Portfolio Construction](https://www.wallstreetprep.com/knowledge/market-neutral-strategy/) - The market neutral strategy is designed to profit from fluctuations in the pricing of securities, wh...

12. [liyiyan128/optiver-trading-at-the-close](https://github.com/liyiyan128/optiver-trading-at-the-close) - The competition goal is to predict the future price movements of stocks relative to the price future...

