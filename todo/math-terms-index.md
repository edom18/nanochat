# 数学用語インデックス

このドキュメントは、nanochatプロジェクトで使用される数学的概念を整理し、どのドキュメントで解説されるかをマッピングします。

## 前提知識の確認
ユーザーの既存知識:
- ✅ 高校数学（代数、三角関数、微分積分の基礎）
- ✅ CGレンダリングの線形代数（ベクトル、行列、内積、外積）
- ✅ 簡単な統計学（平均、分散、標準偏差）

---

## 数学用語リスト

### レベル1: 基礎的な線形代数（復習）
これらはユーザーの前提知識に含まれますが、LLMの文脈で再確認が必要な概念です。

| 用語 | 説明 | ドキュメント |
|------|------|--------------|
| ベクトル | 数値の配列（方向と大きさ） | `doc/math/01-vectors-matrices.md` |
| 行列 | 数値の2次元配列 | `doc/math/01-vectors-matrices.md` |
| 行列の積 | 行列同士の乗算 | `doc/math/02-matrix-operations.md` |
| 転置 | 行と列を入れ替え | `doc/math/02-matrix-operations.md` |
| 内積 | ベクトル同士の積（スカラー値） | `doc/math/02-matrix-operations.md` |
| ノルム | ベクトルの長さ | `doc/math/02-matrix-operations.md` |

### レベル2: LLMで必須の基礎数学
高校数学の範囲を少し超える、LLM理解に不可欠な概念です。

| 用語 | 説明 | コード上の出現箇所 | ドキュメント |
|------|------|-------------------|--------------|
| **Softmax関数** | ベクトルを確率分布に変換 | `gpt.py:180` (Attention), `engine.py:45` (Sampling) | `doc/math/03-softmax.md` |
| **交差エントロピー損失** | 分類問題の損失関数 | `gpt.py:247` (forward) | `doc/math/04-cross-entropy.md` |
| **対数尤度** | 確率の対数を取った値 | 損失計算に含まれる | `doc/math/04-cross-entropy.md` |

### レベル3: 機械学習の基礎
CGレンダリングにはあまり登場しないが、機械学習では必須の概念です。

| 用語 | 説明 | コード上の出現箇所 | ドキュメント |
|------|------|-------------------|--------------|
| **勾配降下法** | 損失を最小化するパラメータ更新 | `muon.py`, `adamw.py` | `doc/math/05-gradient-descent.md` |
| **誤差逆伝播法** | 勾配を効率的に計算する手法 | PyTorchの`backward()` | `doc/math/06-backpropagation.md` |
| **学習率** | パラメータ更新の大きさ | `muon.py:120`, `adamw.py:85` | `doc/math/05-gradient-descent.md` |
| **モーメンタム** | 過去の勾配を考慮した更新 | `muon.py:115` (momentum) | `doc/math/10-optimization-algorithms.md` |
| **正則化** | 過学習を防ぐ手法 | `adamw.py` (weight decay) | `doc/math/10-optimization-algorithms.md` |

### レベル4: Transformerの核心メカニズム
Transformerモデル特有の高度な数学的概念です。

| 用語 | 説明 | コード上の出現箇所 | ドキュメント |
|------|------|-------------------|--------------|
| **Attention機構** | 重要な部分に注目する仕組み | `gpt.py:118-185` (CausalSelfAttention) | `doc/math/07-attention-mechanism.md` |
| **Query/Key/Value** | Attentionの3つの要素 | `gpt.py:125-127` | `doc/math/07-attention-mechanism.md` |
| **Scaled Dot-Product** | Q・Kの内積をスケーリング | `gpt.py:168-170` | `doc/math/07-attention-mechanism.md` |
| **Causal Masking** | 未来の情報を隠す | `gpt.py:175-177` | `doc/math/07-attention-mechanism.md` |
| **Multi-Query Attention** | KVを共有する効率化手法 | `gpt.py:129-130` | `doc/math/07-attention-mechanism.md` |

### レベル5: 正規化とエンベディング
モデルの安定性と効率に関わる高度な技術です。

| 用語 | 説明 | コード上の出現箇所 | ドキュメント |
|------|------|-------------------|--------------|
| **Layer Normalization** | 層ごとの正規化 | 一般的な手法（参照用） | `doc/math/08-layer-normalization.md` |
| **RMSNorm** | パラメータなしの正規化 | `gpt.py:30-39` | `doc/math/08-layer-normalization.md` |
| **QK Normalization** | Query/Keyの正規化 | `gpt.py:163-165` | `doc/math/08-layer-normalization.md` |
| **位置エンコーディング** | トークンの位置情報 | 概念説明 | `doc/math/09-positional-encoding.md` |
| **Rotary Embeddings (RoPE)** | 回転行列による位置エンコーディング | `gpt.py:42-115` | `doc/math/09-positional-encoding.md` |
| **複素数による回転** | RoPEの数学的基礎 | `gpt.py:82-90` | `doc/math/09-positional-encoding.md` |

### レベル6: 最適化アルゴリズム
各種オプティマイザーの数学的背景です。

| 用語 | 説明 | コード上の出現箇所 | ドキュメント |
|------|------|-------------------|--------------|
| **SGD** | 確率的勾配降下法 | 基礎概念 | `doc/math/10-optimization-algorithms.md` |
| **Adam** | 適応的モーメント推定 | 基礎概念 | `doc/math/10-optimization-algorithms.md` |
| **AdamW** | 重み減衰を分離したAdam | `adamw.py` 全体 | `doc/math/10-optimization-algorithms.md` |
| **Muon** | 直交化を用いた最適化 | `muon.py` 全体 | `doc/math/10-optimization-algorithms.md` |
| **ZeRO最適化** | 分散訓練の最適化 | `adamw.py:60-120` | `doc/math/10-optimization-algorithms.md` |
| **直交化 (Orthogonalization)** | 行列を直交行列に変換 | `muon.py:50-75` | `doc/math/10-optimization-algorithms.md` |

### レベル7: 活性化関数
非線形変換を行う関数群です。

| 用語 | 説明 | コード上の出現箇所 | ドキュメント |
|------|------|-------------------|--------------|
| **ReLU** | Rectified Linear Unit | 基礎概念 | `doc/math/11-activation-functions.md` |
| **ReLU²** | ReLUの2乗 | `gpt.py:210` (MLP) | `doc/math/11-activation-functions.md` |
| **GELU** | Gaussian Error Linear Unit | 比較対象として説明 | `doc/math/11-activation-functions.md` |
| **tanh** | 双曲線正接関数 | `gpt.py:251` (logits softcap) | `doc/math/11-activation-functions.md` |
| **Softcap** | Logitsの値を制限 | `gpt.py:251` | `doc/math/11-activation-functions.md` |

### レベル8: サンプリングと確率
推論時の生成に関わる確率論的概念です。

| 用語 | 説明 | コード上の出現箇所 | ドキュメント |
|------|------|-------------------|--------------|
| **温度パラメータ (Temperature)** | 確率分布の鋭さを制御 | `engine.py:45-46` | `doc/math/12-probability-sampling.md` |
| **Top-k Sampling** | 上位k個から選択 | `engine.py:47-49` | `doc/math/12-probability-sampling.md` |
| **多項分布** | カテゴリカル分布 | `engine.py:50` | `doc/math/12-probability-sampling.md` |
| **Greedy Decoding** | 最大確率を選択 | `engine.py:43` (temp=0) | `doc/math/12-probability-sampling.md` |

---

## ドキュメント作成の優先順位

### 高優先度（LLM理解に必須）
1. `doc/math/03-softmax.md` - Attention、損失計算で頻出
2. `doc/math/04-cross-entropy.md` - 訓練の核心
3. `doc/math/07-attention-mechanism.md` - Transformerの中核
4. `doc/math/05-gradient-descent.md` - 訓練の基礎

### 中優先度（詳細理解に必要）
5. `doc/math/08-layer-normalization.md` - RMSNormの理解
6. `doc/math/09-positional-encoding.md` - RoPEの理解
7. `doc/math/10-optimization-algorithms.md` - Muon/AdamWの理解
8. `doc/math/11-activation-functions.md` - MLPの理解

### 低優先度（補足的）
9. `doc/math/01-vectors-matrices.md` - 復習用
10. `doc/math/02-matrix-operations.md` - 復習用
11. `doc/math/06-backpropagation.md` - PyTorchが自動化
12. `doc/math/12-probability-sampling.md` - 推論時のみ

---

## コードと数学の対応表

### gpt.py（モデル実装）
| 行番号 | 数学的概念 | ドキュメント参照 |
|--------|-----------|-----------------|
| 30-39 | RMSNorm | `doc/math/08-layer-normalization.md` |
| 42-115 | Rotary Embeddings | `doc/math/09-positional-encoding.md` |
| 118-185 | Attention機構 | `doc/math/07-attention-mechanism.md` |
| 163-165 | QK Normalization | `doc/math/08-layer-normalization.md` |
| 168-170 | Scaled Dot-Product | `doc/math/07-attention-mechanism.md` |
| 180 | Softmax | `doc/math/03-softmax.md` |
| 210 | ReLU² | `doc/math/11-activation-functions.md` |
| 247 | Cross-Entropy Loss | `doc/math/04-cross-entropy.md` |
| 251 | tanh Softcap | `doc/math/11-activation-functions.md` |

### muon.py（Muonオプティマイザー）
| 行番号 | 数学的概念 | ドキュメント参照 |
|--------|-----------|-----------------|
| 50-75 | 直交化 | `doc/math/10-optimization-algorithms.md` |
| 115 | モーメンタム | `doc/math/10-optimization-algorithms.md` |
| 120 | 学習率スケジューリング | `doc/math/05-gradient-descent.md` |

### adamw.py（AdamWオプティマイザー）
| 行番号 | 数学的概念 | ドキュメント参照 |
|--------|-----------|-----------------|
| 60-120 | ZeRO-2最適化 | `doc/math/10-optimization-algorithms.md` |
| 85 | AdamWアルゴリズム | `doc/math/10-optimization-algorithms.md` |

### engine.py（推論エンジン）
| 行番号 | 数学的概念 | ドキュメント参照 |
|--------|-----------|-----------------|
| 43-50 | サンプリング手法 | `doc/math/12-probability-sampling.md` |
| 45 | Softmax with Temperature | `doc/math/03-softmax.md`, `doc/math/12-probability-sampling.md` |

---

**最終更新**: 2025-10-18
**次のステップ**: 第1章（プロジェクト概要）のドキュメント作成
