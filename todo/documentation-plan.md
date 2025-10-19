# nanochat LLM学習ドキュメント作成 - ToDoリスト

## プロジェクト目標
nanochatプロジェクトの実装を読み解き、LLM初学者が理解しやすいドキュメントを体系的に作成する。

## 対象読者の前提知識
- 高校数学
- CGレンダリングにおける線形代数
- 簡単な統計学の知識

## ドキュメント配置ルール
- **LLM関連の説明**: `doc/` ディレクトリ以下
- **数学的知識**: `doc/math/` ディレクトリ以下
- 用語は別ドキュメントとして整理（煩雑さを避けるため）

---

## 進行状況

### ✅ 完了したタスク
1. [x] プロジェクト全体の構造把握と分析完了
2. [x] ドキュメント構成の設計とToDoリストの作成
3. [x] todo/ディレクトリの作成とToDoリストファイルの保存
4. [x] doc/とdoc/mathディレクトリの作成
5. [x] 数学的用語リストの抽出とドキュメント項目の決定
6. [x] 第1章: プロジェクト概要ドキュメントの作成 (`doc/01-project-overview.md`)
7. [x] 第2章: Transformerアーキテクチャの基礎ドキュメントの作成 (`doc/02-transformer-basics.md`)
8. [x] 第3章: トークナイザーとBPEドキュメントの作成 (`doc/03-tokenizer.md`)
9. [x] 第4章: モデルの詳細実装ドキュメントの作成 (`doc/04-model-implementation.md`)
10. [x] 第5章: データパイプラインと訓練プロセスドキュメントの作成 (`doc/05-training-pipeline.md`)
11. [x] 第6章: 最適化手法（Muon, AdamW）ドキュメントの作成 (`doc/06-optimization.md`)
12. [x] 第7章: 推論エンジンとサンプリングドキュメントの作成 (`doc/07-inference.md`)

### 🔄 進行中のタスク
なし

### 📋 未着手のタスク
13. [ ] 第8章: 評価とベンチマークドキュメントの作成
14. [ ] 数学用語ドキュメント群の作成（doc/math/配下）

---

## 詳細計画

### フェーズ1: 準備と構成設計 (現在のフェーズ)
- [x] プロジェクト構造の把握
- [ ] ドキュメント構成の設計
- [ ] 数学的用語の洗い出し
- [ ] 各章の概要設計

### フェーズ2: 基礎ドキュメント作成
#### 第1章: プロジェクト概要
**ファイル**: `doc/01-project-overview.md`
**内容**:
- nanochatプロジェクトとは
- 全体のアーキテクチャ図
- 訓練パイプライン（Tokenizer → Pretrain → Mid-train → SFT → RL → Inference）
- ファイル構成と各モジュールの役割
- 必要な環境とハードウェア

#### 第2章: Transformerアーキテクチャの基礎
**ファイル**: `doc/02-transformer-basics.md`
**内容**:
- Transformerとは何か
- Decoder-only Transformerの概念
- GPTモデルの基本構造
- Attention機構の概要（詳細な数式はdoc/math/へ）
- 位置エンコーディングの役割

### フェーズ3: コアコンポーネント解説
#### 第3章: トークナイザーとBPE
**ファイル**: `doc/03-tokenizer.md`
**内容**:
- テキストからトークンへの変換
- BPE（Byte Pair Encoding）アルゴリズム
- tokenizer.pyの実装解説
- RustBPEの役割と高速化

#### 第4章: モデルの詳細実装
**ファイル**: `doc/04-model-implementation.md`
**内容**:
- gpt.pyの実装解説
- GPTクラスの構造
- CausalSelfAttentionの実装
- MLPブロックの実装
- 最新技術の解説（RoPE, RMSNorm, Multi-Query Attention等）

### フェーズ4: 訓練と最適化
#### 第5章: データパイプラインと訓練プロセス
**ファイル**: `doc/05-training-pipeline.md`
**内容**:
- dataset.pyとdataloader.pyの解説
- HuggingFace fineweb-eduデータセットの利用
- 訓練ループの実装
- 損失計算（Cross-Entropy Loss）
- 分散訓練（DDP）の仕組み

#### 第6章: 最適化手法
**ファイル**: `doc/06-optimization.md`
**内容**:
- Muonオプティマイザーの解説
- AdamWオプティマイザーの解説
- パラメータグループごとの学習率設定
- 勾配累積の仕組み
- ZeRO-2スタイルの最適化

### フェーズ5: 推論と評価
#### 第7章: 推論エンジンとサンプリング
**ファイル**: `doc/07-inference.md`
**内容**:
- engine.pyの実装解説
- KVキャッシュの仕組み
- サンプリング手法（Top-k, Temperature）
- ストリーミング生成
- チャットインターフェースの実装

#### 第8章: 評価とベンチマーク
**ファイル**: `doc/08-evaluation.md`
**内容**:
- CORE評価メトリクス
- タスク別評価（ARC, GSM8K, MMLU, HumanEval）
- 損失評価（bits per byte）
- 評価スクリプトの使い方

### フェーズ6: 数学用語集（doc/math/配下）

以下の用語について、高校数学+基礎的な線形代数の知識から理解できるよう段階的に解説:

#### 数学用語ドキュメント一覧
1. **`doc/math/01-vectors-matrices.md`**: ベクトルと行列の基礎
2. **`doc/math/02-matrix-operations.md`**: 行列演算（積、転置、内積）
3. **`doc/math/03-softmax.md`**: Softmax関数
4. **`doc/math/04-cross-entropy.md`**: 交差エントロピー損失
5. **`doc/math/05-gradient-descent.md`**: 勾配降下法
6. **`doc/math/06-backpropagation.md`**: 誤差逆伝播法
7. **`doc/math/07-attention-mechanism.md`**: Attention機構の数式
8. **`doc/math/08-layer-normalization.md`**: 正規化手法（LayerNorm, RMSNorm）
9. **`doc/math/09-positional-encoding.md`**: 位置エンコーディング（Rotary Embeddings）
10. **`doc/math/10-optimization-algorithms.md`**: 最適化アルゴリズム（SGD, Adam, AdamW, Muon）
11. **`doc/math/11-activation-functions.md`**: 活性化関数（ReLU, ReLU², GELU, tanh）
12. **`doc/math/12-probability-sampling.md`**: 確率的サンプリング

---

## 各章作成時の注意事項

### ドキュメント作成ガイドライン
1. **段階的な説明**: 基礎から応用へ順序立てて解説
2. **コード引用**: 実際のコード（ファイル名:行番号）を参照
3. **図表の活用**: 必要に応じてASCII artや構造図を挿入
4. **数式の分離**: 複雑な数式は doc/math/ に分離し、リンクで参照
5. **実例の提供**: 具体的な数値例や実行例を示す
6. **用語の統一**: 日本語訳と英語表記を併記
7. **セッション継続性**: 各章完成後、次章への移行を確認

### 進め方
- 各章ごとに作成し、完成後にユーザーに確認を求める
- ユーザーの理解度に応じて詳細度を調整
- 不明点があればコードを再確認し、正確な情報を提供

---

## 現在の状態
- **フェーズ**: フェーズ1 - 準備と構成設計
- **次のアクション**: ToDoリストファイルの保存 → 数学的用語リストの抽出

## セッション再開時の手順
1. `todo/documentation-plan.md` を確認
2. 進行状況セクションで現在のタスクを確認
3. 前回の成果物（doc/配下のファイル）を確認
4. 次のタスクから継続

---

**最終更新**: 2025-10-18
**作成者**: AI Agent
