# 第1章: nanochatプロジェクト概要

## 目次
1. [nanochatとは](#nanochatとは)
2. [プロジェクトの目的と特徴](#プロジェクトの目的と特徴)
3. [訓練パイプライン全体像](#訓練パイプライン全体像)
4. [ディレクトリ構造](#ディレクトリ構造)
5. [主要コンポーネント](#主要コンポーネント)
6. [必要な環境](#必要な環境)
7. [次章への導入](#次章への導入)

---

## nanochatとは

**nanochat**は、ChatGPTのようなLLM（大規模言語モデル）をフルスタックで実装した、シンプルで最小限のコードベースです。

### キーコンセプト
> The best ChatGPT that $100 can buy.
> （100ドルで手に入る最高のChatGPT）

このキャッチフレーズが示すように、nanochatは**低予算**（100〜1000ドル）で、ChatGPT風のAIチャットボットを**ゼロから訓練**し、**実際に会話できる**ようにするプロジェクトです。

### 実現できること
- **トークナイザーの訓練**: テキストをトークンに分割する仕組みの構築
- **事前訓練（Pretraining）**: 大量のテキストデータからLLMを訓練
- **中間訓練（Mid-training）**: 会話や特殊タスクへの適応
- **教師あり微調整（SFT）**: 対話データでの精緻化
- **強化学習（RL）**: 報酬ベースでの最適化（オプション）
- **推論（Inference）**: 訓練したモデルとの対話
- **Webサービング**: ChatGPT風のWebUIでの提供

### 具体的な成果物
- **d32モデル**: 32層のTransformer、19億パラメータ
- **訓練データ**: 380億トークン
- **訓練時間**: 約4時間（8XH100 GPU）
- **訓練コスト**: 約100ドル（speedrun.sh）、最大1000ドル
- **パフォーマンス**: GPT-2（2019年）を上回る性能

---

## プロジェクトの目的と特徴

### 教育的目的
nanochatは、Eureka Labsが開発中の**LLM101n**コースのキャップストーンプロジェクトとして設計されています。以下の点で初学者に最適です:

1. **シンプルなコードベース**
   - 全体で約8,300行、44ファイル
   - 巨大な設定オブジェクトや複雑な分岐がない
   - 読みやすく、ハック可能（改造しやすい）

2. **フルスタック実装**
   - トークン化から推論まで、すべてのステップを網羅
   - 各ステップが独立したスクリプトで実行可能

3. **明確なメトリクス**
   - CORE、ARC、GSM8K、MMLU、HumanEvalなどのベンチマーク
   - 訓練の進捗と成果が数値で確認可能

### 技術的特徴
- **最新技術の採用**:
  - Rotary Embeddings（RoPE）による位置エンコーディング
  - Multi-Query Attention（MQA）によるメモリ効率化
  - RMSNormによる正規化
  - Muonオプティマイザーによる効率的な最適化

- **実用性**:
  - 単一の8XH100ノードで完結（8XA100も可）
  - 単一GPUでも動作可能（時間は8倍）
  - メモリに応じてバッチサイズを調整可能

- **依存関係の最小化**:
  - 主要な依存: PyTorch, HuggingFace, FastAPI
  - Rustによる高速トークナイザー（RustBPE）

---

## 訓練パイプライン全体像

nanochatの訓練は、以下の6つのフェーズで構成されます。

```
┌─────────────────────────────────────────────────────────────────┐
│                    nanochat 訓練パイプライン                      │
└─────────────────────────────────────────────────────────────────┘

  [1] トークナイザー訓練 (Tokenizer Training)
      │  入力: 20億文字のテキストデータ
      │  処理: BPE（Byte Pair Encoding）アルゴリズム
      │  出力: 語彙サイズ65,536のトークナイザー
      │  コスト: ~5分
      ▼

  [2] 事前訓練 (Base Pretraining)
      │  入力: 380億トークンのfineweb-eduデータセット
      │  処理: Causal Language Modeling（次トークン予測）
      │  出力: base.pt（基礎モデルチェックポイント）
      │  評価: CORE メトリクス
      │  コスト: ~2.5時間、$60
      ▼

  [3] 中間訓練 (Mid-training)
      │  入力: タスクミックス（ARC, GSM8K, MMLU, SmolTalk）
      │  処理: 会話特殊トークン、ツール使用、多肢選択の学習
      │  出力: mid.pt（中間訓練済みモデル）
      │  評価: ARC, GSM8K, MMLU, HumanEval, ChatCORE
      │  コスト: ~40分、$16
      ▼

  [4] 教師あり微調整 (Supervised Fine-Tuning / SFT)
      │  入力: 会話データセット
      │  処理: ドメイン適応（対話スタイルの学習）
      │  出力: sft.pt（微調整済みモデル）
      │  評価: 各種ベンチマークの再評価
      │  コスト: ~15分、$6
      ▼

  [5] 強化学習 (Reinforcement Learning / RL) ※オプション
      │  入力: GSM8Kタスク
      │  処理: 報酬ベースの最適化
      │  出力: rl.pt（RL最適化モデル）
      │  評価: GSM8K
      │  コスト: （speedrun.shではデフォルトでスキップ）
      ▼

  [6] 推論・サービング (Inference & Serving)
      │  CLI: scripts/chat_cli.py（コマンドライン対話）
      │  Web UI: scripts/chat_web.py（ChatGPT風インターフェース）
      │  エンジン: nanochat/engine.py（KVキャッシュ、サンプリング）
      │  コスト: 推論は低コスト（GPU1枚で十分）

```

### フェーズごとの詳細

#### [1] トークナイザー訓練
**目的**: テキストを数値トークンに変換する辞書を作成

- **アルゴリズム**: BPE（Byte Pair Encoding）
- **語彙サイズ**: 65,536（2^16）
- **訓練データ**: 約20億文字（fineweb-eduの最初の8シャード）
- **実装**: `scripts/tok_train.py` + `rustbpe`（Rust実装）
- **圧縮率**: 約4.8文字/トークン（評価: `scripts/tok_eval.py`）

**詳細は第3章で解説します。**

#### [2] 事前訓練（Base Pretraining）
**目的**: 一般的な言語理解能力を獲得

- **モデル**: GPT風のDecoder-only Transformer（d20 = 20層、561Mパラメータ）
- **訓練データ**: fineweb-edu（HuggingFaceデータセット、高品質な教育コンテンツ）
- **訓練トークン数**: 約112億トークン（Chinchillaルール: 20倍パラメータ数）
- **訓練方法**: Causal Language Modeling（次のトークンを予測）
- **最適化**: Muonオプティマイザー（行列パラメータ）+ AdamW（埋め込み）
- **実装**: `scripts/base_train.py`
- **評価**: CORE メトリクス（`scripts/base_eval.py`）

**Chinchillaルール**: 最適な訓練には「トークン数 = 20 × パラメータ数」が推奨される経験則。

**詳細は第4章、第5章で解説します。**

#### [3] 中間訓練（Mid-training）
**目的**: 特定のタスクや会話形式への適応

- **タスクミックス**:
  - **ARC**: 科学の多肢選択問題
  - **GSM8K**: 小学生レベルの数学推論
  - **MMLU**: 広範な知識問題
  - **SmolTalk**: 会話データ
- **特殊トークン**: `<|user|>`, `<|assistant|>`, `<|im_start|>`, `<|im_end|>`など
- **ツール使用**: 電卓機能（計算タスク用）
- **実装**: `scripts/mid_train.py`, `tasks/`ディレクトリ

**詳細は第5章、第8章で解説します。**

#### [4] 教師あり微調整（SFT）
**目的**: 会話スタイルの洗練と、ユーザー指示への追従性向上

- **データ**: 会話形式のデータセット
- **訓練方法**: 1つの会話シーケンス全体を1サンプルとして学習
- **実装**: `scripts/chat_sft.py`
- **評価**: 各種ベンチマークで性能向上を確認

**詳細は第5章で解説します。**

#### [5] 強化学習（RL）
**目的**: 正しい答えに対する報酬を最大化

- **タスク**: GSM8K（数学推論）
- **手法**: 報酬ベースの最適化
- **実装**: `scripts/chat_rl.py`
- **注意**: デフォルトではスキップ（speedrun.shではコメントアウト済み）

**詳細は第6章で解説します。**

#### [6] 推論・サービング
**目的**: 訓練済みモデルと対話

- **CLI**: `python -m scripts.chat_cli -p "Why is the sky blue?"`
- **Web UI**: `python -m scripts.chat_web`（ポート8000でサーバー起動）
- **エンジン**: `nanochat/engine.py`
  - KVキャッシュによる高速化
  - Top-k サンプリング
  - 温度パラメータによる多様性制御

**詳細は第7章で解説します。**

---

## ディレクトリ構造

nanochatプロジェクトは、以下のディレクトリ構造で構成されています。

```
nanochat/
├── nanochat/               # コアライブラリ（モデル、データ、訓練ロジック）
│   ├── gpt.py              # GPT Transformerモデルの実装
│   ├── tokenizer.py        # トークナイザーラッパー
│   ├── dataset.py          # Parquetファイル管理（HuggingFace fineweb-edu）
│   ├── dataloader.py       # データストリーミング＆トークン化
│   ├── engine.py           # 推論エンジン（KVキャッシュ、サンプリング）
│   ├── muon.py             # Muonオプティマイザー
│   ├── adamw.py            # 分散AdamWオプティマイザー
│   ├── checkpoint_manager.py # チェックポイントの保存/読み込み
│   ├── core_eval.py        # COREベンチマーク評価
│   ├── loss_eval.py        # 検証損失評価
│   ├── report.py           # 訓練レポート生成
│   ├── common.py           # 分散訓練ユーティリティ
│   ├── configurator.py     # CLIパーサー
│   └── execution.py        # 訓練ループ実行
│
├── scripts/                # 訓練・評価・推論スクリプト
│   ├── tok_train.py        # トークナイザー訓練
│   ├── tok_eval.py         # トークナイザー評価
│   ├── base_train.py       # 事前訓練
│   ├── base_loss.py        # 損失評価
│   ├── base_eval.py        # CORE評価
│   ├── mid_train.py        # 中間訓練
│   ├── chat_sft.py         # 教師あり微調整
│   ├── chat_rl.py          # 強化学習
│   ├── chat_eval.py        # 会話モデル評価
│   ├── chat_cli.py         # CLIチャットインターフェース
│   └── chat_web.py         # Web UIチャットインターフェース
│
├── tasks/                  # 評価タスク実装
│   ├── common.py           # Task基底クラス、TaskMixture
│   ├── arc.py              # ARC-Easy/Challenge（多肢選択）
│   ├── gsm8k.py            # GSM8K（数学推論）
│   ├── mmlu.py             # MMLU（知識問題）
│   ├── humaneval.py        # HumanEval（コード生成）
│   └── smoltalk.py         # SmolTalk（会話品質）
│
├── rustbpe/                # Rust実装のBPEトークナイザー
│   ├── Cargo.toml          # Rustプロジェクト設定
│   └── src/                # Rustソースコード
│
├── dev/                    # 開発用スクリプト
│   └── repackage_data_reference.py  # データ再パッケージング
│
├── tests/                  # テストコード
│   └── test_rustbpe.py     # トークナイザーのテスト
│
├── docs/                   # ドキュメント（オリジナル）
│
├── doc/                    # LLM学習ドキュメント（本ドキュメント群）
│   ├── 01-project-overview.md       # 本ファイル
│   ├── 02-transformer-basics.md     # Transformer基礎
│   ├── 03-tokenizer.md              # トークナイザー
│   ├── 04-model-implementation.md   # モデル実装
│   ├── 05-training-pipeline.md      # 訓練パイプライン
│   ├── 06-optimization.md           # 最適化手法
│   ├── 07-inference.md              # 推論エンジン
│   └── 08-evaluation.md             # 評価とベンチマーク
│
├── doc/math/               # 数学用語ドキュメント
│   ├── 01-vectors-matrices.md       # ベクトルと行列
│   ├── 02-matrix-operations.md      # 行列演算
│   ├── 03-softmax.md                # Softmax関数
│   ├── 04-cross-entropy.md          # 交差エントロピー損失
│   ├── 05-gradient-descent.md       # 勾配降下法
│   ├── 06-backpropagation.md        # 誤差逆伝播法
│   ├── 07-attention-mechanism.md    # Attention機構
│   ├── 08-layer-normalization.md    # 正規化手法
│   ├── 09-positional-encoding.md    # 位置エンコーディング
│   ├── 10-optimization-algorithms.md # 最適化アルゴリズム
│   ├── 11-activation-functions.md   # 活性化関数
│   └── 12-probability-sampling.md   # 確率的サンプリング
│
├── todo/                   # ToDoリストと計画書
│   ├── documentation-plan.md        # ドキュメント作成計画
│   └── math-terms-index.md          # 数学用語インデックス
│
├── speedrun.sh             # 4時間訓練スクリプト（$100）
├── run1000.sh              # 長時間訓練スクリプト（$1000）
├── pyproject.toml          # Python依存関係
├── uv.lock                 # 依存関係ロックファイル
├── README.md               # プロジェクト概要
└── AGENTS.md               # AIエージェント向け指示

実行時に生成されるディレクトリ（~/.cache/nanochat/）:
├── data_shards/            # ダウンロードしたデータシャード
├── tokenizers/             # 訓練済みトークナイザー
├── checkpoints/            # モデルチェックポイント（base.pt, mid.pt, sft.pt, rl.pt）
├── eval_bundle/            # CORE評価用データ
└── report/                 # 訓練レポート（report.md）
```

### ディレクトリの役割

#### `nanochat/` - コアライブラリ
プロジェクトの心臓部。モデル定義、データ処理、訓練ロジック、推論エンジンなど、再利用可能なコンポーネントが格納されています。

#### `scripts/` - 実行スクリプト
訓練・評価・推論を実行するエントリーポイント。各スクリプトは独立して実行可能で、コマンドライン引数で設定を変更できます。

#### `tasks/` - 評価タスク
LLMの性能を測定するベンチマークタスクの実装。各タスクは共通の`Task`インターフェースを実装しています。

#### `rustbpe/` - Rustトークナイザー
PythonのBPE実装は遅いため、Rustで高速化されたトークナイザーを使用。Pythonからは`rustbpe`モジュールとして利用できます。

---

## 主要コンポーネント

### 1. モデル（`nanochat/gpt.py`）
GPT風のDecoder-only Transformerモデル。

**主要クラス**:
- `GPT`: メインモデルクラス
- `CausalSelfAttention`: 因果的自己注意機構
- `MLP`: フィードフォワードネットワーク
- `Block`: TransformerブロックAttention + MLP）

**主要な技術**:
- Rotary Embeddings（RoPE）: 位置情報を回転行列で表現
- Multi-Query Attention（MQA）: KeyとValueを共有してメモリ削減
- RMSNorm: パラメータなしの正規化
- ReLU²活性化関数: GELUより高速
- QK Normalization: Attention安定化
- Logits Softcap: 数値安定性向上

**詳細は第2章、第4章で解説します。**

### 2. トークナイザー（`nanochat/tokenizer.py`, `rustbpe/`）
テキストをトークンIDの列に変換。

**アルゴリズム**: BPE（Byte Pair Encoding）
- 頻出するバイト列を1トークンにまとめる
- 語彙サイズ: 65,536
- 圧縮率: 約4.8文字/トークン

**実装**:
- Python側: `tokenizer.py`（ラッパー）
- Rust側: `rustbpe/`（高速エンコード/デコード）

**詳細は第3章で解説します。**

### 3. データパイプライン（`nanochat/dataset.py`, `nanochat/dataloader.py`）
HuggingFaceのfineweb-eduデータセットを効率的にストリーミング。

**データフロー**:
1. `dataset.py`: Parquetファイルをダウンロード＆管理
2. `dataloader.py`: ドキュメントをロード → トークン化 → バッチ化

**データセット**: fineweb-edu（高品質な教育コンテンツ、合計1822シャード）

**詳細は第5章で解説します。**

### 4. 最適化（`nanochat/muon.py`, `nanochat/adamw.py`）
効率的なパラメータ更新。

**Muonオプティマイザー**: 行列パラメータ（線形層）に使用
- 直交化によるパラメータ更新
- 学習率: 0.02

**AdamWオプティマイザー**: 埋め込み層、lm_headに使用
- 重み減衰を分離したAdam
- 学習率: 埋め込み 0.2、lm_head 0.004
- ZeRO-2スタイルの分散最適化

**詳細は第6章で解説します。**

### 5. 推論エンジン（`nanochat/engine.py`）
訓練済みモデルとの対話を実現。

**主要機能**:
- **KVキャッシュ**: 過去のKey/Valueを再利用して高速化
- **サンプリング**: Top-k、温度パラメータによる多様性制御
- **ストリーミング生成**: トークンを1つずつ生成
- **ツール統合**: 電卓機能（計算タスク用）

**詳細は第7章で解説します。**

### 6. 評価（`nanochat/core_eval.py`, `tasks/`）
モデルの性能を客観的に測定。

**CORE評価**: DCLM論文に基づく総合メトリクス
**タスク評価**:
- ARC: 科学の多肢選択問題
- GSM8K: 数学推論
- MMLU: 広範な知識
- HumanEval: コード生成
- SmolTalk/ChatCORE: 会話品質

**詳細は第8章で解説します。**

---

## 必要な環境

### ハードウェア要件

#### 推奨環境（speedrun.sh）
- **GPU**: 8x H100（各80GB VRAM）
- **訓練時間**: 約4時間
- **コスト**: 約$100（$3/GPU/時 × 8 GPU × 4時間）

#### 代替環境
- **8x A100 80GB**: 動作可能（H100より少し遅い）
- **単一GPU**: 動作可能（訓練時間は8倍、約32時間）
- **小容量GPU**: `--device_batch_size`を調整（32 → 16 → 8 → 4 → 2 → 1）

**メモリ不足時の対処法**:
```bash
# デバイスバッチサイズを減らす（勾配累積で補完）
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --device_batch_size=16
```

#### CPU/MPS（実験的）
- Macbook（Apple Silicon）: `--device_type=mps`
- CPU: 動作可能だが非常に遅い
- 参考: [CPU|MPS PR #88](https://github.com/karpathy/nanochat/pull/88)

### ソフトウェア要件

#### 必須
- **Python**: 3.10以上
- **uv**: Pythonパッケージマネージャー（自動インストール）
- **Rust/Cargo**: トークナイザーのビルド用（自動インストール）

#### 主要な依存パッケージ（pyproject.toml）
```toml
dependencies = [
    "torch>=2.8.0",           # ディープラーニングフレームワーク
    "datasets>=4.0.0",        # HuggingFaceデータセット
    "transformers",           # トークナイザーユーティリティ
    "tiktoken>=0.11.0",       # トークンエンコーディング
    "tokenizers>=0.22.0",     # HuggingFaceトークナイザー
    "fastapi>=0.117.1",       # Web API
    "uvicorn>=0.36.0",        # ASGIサーバー
    "wandb>=0.21.3",          # 実験トラッキング（オプション）
    "numpy==1.26.4",          # 数値計算
    "regex>=2025.9.1",        # 正規表現
    "psutil>=7.1.0",          # システム情報
]
```

### クイックスタート

#### 1. GPU環境の準備
Lambda、AWS、GCP、Azureなどで8XH100ノードを起動。

#### 2. リポジトリのクローン
```bash
git clone https://github.com/karpathy/nanochat.git
cd nanochat
```

#### 3. 訓練開始
```bash
# シンプル実行
bash speedrun.sh

# screen セッションで実行（推奨）
screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh

# wandbロギング有効化
WANDB_RUN=speedrun bash speedrun.sh
```

#### 4. 約4時間後、推論開始
```bash
# 仮想環境を有効化
source .venv/bin/activate

# Web UIで対話
python -m scripts.chat_web
# ブラウザで http://公開IP:8000/ にアクセス

# CLIで対話
python -m scripts.chat_cli -p "Why is the sky blue?"
```

#### 5. レポート確認
```bash
cat report.md
```

**レポート例**:
```markdown
| Metric          | BASE     | MID      | SFT      | RL       |
|-----------------|----------|----------|----------|----------|
| CORE            | 0.2219   | -        | -        | -        |
| ARC-Challenge   | -        | 0.2875   | 0.2807   | -        |
| ARC-Easy        | -        | 0.3561   | 0.3876   | -        |
| GSM8K           | -        | 0.0250   | 0.0455   | 0.0758   |
| HumanEval       | -        | 0.0671   | 0.0854   | -        |
| MMLU            | -        | 0.3111   | 0.3151   | -        |
| ChatCORE        | -        | 0.0730   | 0.0884   | -        |
```

---

## 次章への導入

第1章では、nanochatプロジェクトの全体像を把握しました。

### これまでに学んだこと
- nanochatは低予算でChatGPT風LLMを訓練できるフルスタック実装
- 訓練パイプラインは6つのフェーズ（トークナイザー → 事前訓練 → 中間訓練 → SFT → RL → 推論）
- コードベースはシンプルで、教育目的に最適化されている
- 8XH100で4時間、約$100で訓練可能

### 次章で学ぶこと

**第2章: Transformerアーキテクチャの基礎**
- Transformerとは何か
- Decoder-only Transformerの構造
- Self-Attention機構の概要
- GPTモデルの基本原理

Transformerは現代のLLMの中核技術です。次章では、数式の詳細には立ち入らず（それはdoc/math/で扱います）、**概念的な理解**を深めます。

---

**参照ドキュメント**:
- [README.md](/Users/edom18/MyDesktop/PythonProjects/nanochat/README.md)
- [speedrun.sh](/Users/edom18/MyDesktop/PythonProjects/nanochat/speedrun.sh:1)
- [pyproject.toml](/Users/edom18/MyDesktop/PythonProjects/nanochat/pyproject.toml:1)

**関連する数学ドキュメント**:
- 後続の章で随時リンクします

---

**次へ**: [第2章: Transformerアーキテクチャの基礎](02-transformer-basics.md)
**戻る**: [ドキュメント作成計画](../todo/documentation-plan.md)
