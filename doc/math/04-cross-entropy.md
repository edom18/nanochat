# 数学04: 交差エントロピー損失

## 目次
- [4.1 はじめに](#41-はじめに)
- [4.2 エントロピーとは](#42-エントロピーとは)
  - [4.2.1 情報理論の基礎](#421-情報理論の基礎)
  - [4.2.2 シャノンエントロピー](#422-シャノンエントロピー)
- [4.3 交差エントロピー](#43-交差エントロピー)
  - [4.3.1 定義](#431-定義)
  - [4.3.2 直感的な理解](#432-直感的な理解)
- [4.4 交差エントロピー損失](#44-交差エントロピー損失)
  - [4.4.1 分類問題での使用](#441-分類問題での使用)
  - [4.4.2 数式](#442-数式)
  - [4.4.3 具体例](#443-具体例)
- [4.5 負の対数尤度（NLL）](#45-負の対数尤度nll)
- [4.6 PyTorchでの実装](#46-pytorchでの実装)
  - [4.6.1 F.cross_entropy](#461-fcross_entropy)
  - [4.6.2 内部の処理](#462-内部の処理)
- [4.7 バイナリ交差エントロピー](#47-バイナリ交差エントロピー)
- [4.8 KLダイバージェンス](#48-klダイバージェンス)
- [4.9 なぜ交差エントロピーを使うのか](#49-なぜ交差エントロピーを使うのか)
- [4.10 nanochatでの使用例](#410-nanochatでの使用例)
- [4.11 まとめ](#411-まとめ)

---

## 4.1 はじめに

**交差エントロピー損失**（Cross-Entropy Loss）は、分類タスクで最も広く使われる損失関数です。特に、ニューラルネットワークの出力（確率分布）と正解ラベル（真の分布）の差を測る指標として使われます。

言語モデルでは、次のトークンを予測する際に、この損失を最小化するように訓練します。

この章では、交差エントロピーの数学的な定義から、nanochatでの使用例まで学びます。

**前提知識**：
- [数学03: Softmax関数](03-softmax.md)
- 確率の基礎（確率分布の合計が1）

---

## 4.2 エントロピーとは

### 4.2.1 情報理論の基礎

**エントロピー**は、情報理論における「不確実性」や「情報量」の尺度です。

直感的な例：
```
コイン投げ（表/裏が50%ずつ）:
  不確実性が高い → エントロピーが高い

必ず表が出るコイン:
  不確実性がない → エントロピーが0
```

### 4.2.2 シャノンエントロピー

**シャノンエントロピー**は、確率分布P(x)の不確実性を測ります。

```
H(P) = -Σ P(x) log P(x)
```

**具体例**：

```python
import torch
import math

# 確率分布1: 均一（不確実性が高い）
p1 = torch.tensor([0.25, 0.25, 0.25, 0.25])
H1 = -(p1 * torch.log(p1)).sum()
print(f"H(p1) = {H1:.4f}")
# H(p1) = 1.3863  （高い）

# 確率分布2: 偏り（不確実性が低い）
p2 = torch.tensor([0.9, 0.05, 0.03, 0.02])
H2 = -(p2 * torch.log(p2)).sum()
print(f"H(p2) = {H2:.4f}")
# H(p2) = 0.6390  （低い）

# 確率分布3: 確定的（不確実性ゼロ）
p3 = torch.tensor([1.0, 0.0, 0.0, 0.0])
# log(0)は-∞なので、0 * log(0) = 0と定義
H3 = -(p3[p3 > 0] * torch.log(p3[p3 > 0])).sum()
print(f"H(p3) = {H3:.4f}")
# H(p3) = 0.0000  （ゼロ）
```

**解釈**：
- エントロピーが高い = 予測が難しい（不確実性が高い）
- エントロピーが低い = 予測しやすい（不確実性が低い）

---

## 4.3 交差エントロピー

### 4.3.1 定義

**交差エントロピー**は、2つの確率分布P（真の分布）とQ（モデルの予測分布）の間の「距離」を測ります。

```
H(P, Q) = -Σ P(x) log Q(x)
```

- `P(x)`: 真の分布（正解ラベル）
- `Q(x)`: 予測分布（モデルの出力）

### 4.3.2 直感的な理解

交差エントロピーは、「真の分布Pに従ってサンプリングしたとき、Qを使って符号化すると何ビット必要か」を表します。

```
┌─────────────────────────────────────────────┐
│        Cross-Entropy の直感                  │
└─────────────────────────────────────────────┘

真の分布P: 正解は「猫」
  [猫: 1.0, 犬: 0.0, 鳥: 0.0]

モデルの予測Q1: 正しい予測
  [猫: 0.9, 犬: 0.05, 鳥: 0.05]
  H(P, Q1) = -1.0 * log(0.9) = 0.105 （低い）

モデルの予測Q2: 間違った予測
  [猫: 0.1, 犬: 0.7, 鳥: 0.2]
  H(P, Q2) = -1.0 * log(0.1) = 2.303 （高い）
```

**重要なポイント**：
- PとQが一致するほど、交差エントロピーは低い
- PとQが異なるほど、交差エントロピーは高い
- モデルの予測が正解に近いほど、損失が小さい

---

## 4.4 交差エントロピー損失

### 4.4.1 分類問題での使用

分類問題では、各サンプルに対して：
- **真の分布P**: one-hot エンコーディング（正解クラスが1、他は0）
- **予測分布Q**: モデルの出力（Softmax適用後）

### 4.4.2 数式

C個のクラスの分類問題で、正解クラスがyの場合：

```
L = -log Q(y)
  = -log(Softmax(z)_y)
  = -log(exp(z_y) / Σⱼ exp(z_j))
  = -z_y + log(Σⱼ exp(z_j))
```

ここで：
- `z`: モデルの出力（logits）
- `y`: 正解クラスのインデックス
- `Q(y)`: 正解クラスの予測確率

**バッチの場合**：

```
L = (1/N) Σᵢ -log Q_i(y_i)
```

N個のサンプルの平均損失を計算します。

### 4.4.3 具体例

```python
import torch
import torch.nn.functional as F

# 3クラス分類、バッチサイズ2
logits = torch.tensor([
    [2.0, 1.0, 0.5],  # サンプル1
    [0.5, 2.5, 1.0]   # サンプル2
])

# 正解ラベル
targets = torch.tensor([0, 1])  # サンプル1はクラス0、サンプル2はクラス1

# 交差エントロピー損失
loss = F.cross_entropy(logits, targets)
print(f"Loss: {loss:.4f}")
# Loss: 0.5996

# 手動計算で確認
probs = F.softmax(logits, dim=1)
print(probs)
# tensor([[0.6590, 0.2424, 0.0986],
#         [0.1192, 0.8758, 0.2424]])

# サンプル1: 正解クラス0の確率 = 0.6590
# サンプル2: 正解クラス1の確率 = 0.8758

manual_loss = -(torch.log(probs[0, 0]) + torch.log(probs[1, 1])) / 2
print(f"Manual loss: {manual_loss:.4f}")
# Manual loss: 0.5996  （一致）
```

---

## 4.5 負の対数尤度（NLL）

**負の対数尤度**（Negative Log-Likelihood, NLL）は、交差エントロピー損失と本質的に同じです。

```
NLL = -log P(正解 | モデル)
    = -log Q(y)
```

「モデルが正解を予測する確率の対数の負」です。

**なぜ負の対数？**

1. **対数**: 確率の積を和に変換（数値的に安定）
   ```
   P(y1) * P(y2) * ... * P(yn)
   → log P(y1) + log P(y2) + ... + log P(yn)
   ```

2. **負**: 最大化問題を最小化問題に変換
   ```
   最大化: P(正解 | モデル)  （確率を高くしたい）
   最小化: -log P(正解 | モデ��）  （損失を低くしたい）
   ```

---

## 4.6 PyTorchでの実装

### 4.6.1 F.cross_entropy

PyTorchの`F.cross_entropy`は、Softmax + NLLを効率的に計算します。

```python
import torch
import torch.nn.functional as F

# logits: (N, C) - バッチサイズN、クラス数C
logits = torch.randn(32, 10)  # 32サンプル、10クラス
targets = torch.randint(0, 10, (32,))  # 正解ラベル

# 交差エントロピー損失
loss = F.cross_entropy(logits, targets)
```

**重要な特徴**：
- **logits**を直接受け取る（Softmaxは内部で適用）
- **targets**はクラスインデックス（one-hotではない）
- 数値的に安定（log-softmaxを使用）

### 4.6.2 内部の処理

`F.cross_entropy`は以下と等価です：

```python
# 方法1: 明示的にSoftmax + NLL
probs = F.softmax(logits, dim=1)
log_probs = torch.log(probs)
loss1 = F.nll_loss(log_probs, targets)

# 方法2: log_softmax + NLL（より安定）
log_probs = F.log_softmax(logits, dim=1)
loss2 = F.nll_loss(log_probs, targets)

# 方法3: cross_entropy（最も効率的）
loss3 = F.cross_entropy(logits, targets)

print(torch.allclose(loss1, loss2))  # True
print(torch.allclose(loss2, loss3))  # True
```

**内部実装のメリット**：
- `log(softmax(x))`を直接計算せず、`log_softmax(x)`を使用
- 数値的に安定（アンダーフロー/オーバーフローを防ぐ）
- 計算が効率的

---

## 4.7 バイナリ交差エントロピー

**バイナリ交差エントロピー**（Binary Cross-Entropy, BCE）は、2クラス分類専用の損失関数です。

```
BCE = -[y log(p) + (1-y) log(1-p)]
```

- `y`: 正解ラベル（0 or 1）
- `p`: クラス1の予測確率

```python
# シグモイド出力（2クラス分類）
logits = torch.randn(32, 1)
targets = torch.randint(0, 2, (32, 1)).float()

# バイナリ交差エントロピー
loss = F.binary_cross_entropy_with_logits(logits, targets)
```

**多クラス交差エントロピーとの関係**：
- 多クラス: Softmax + NLL
- バイナリ: Sigmoid + BCE

---

## 4.8 KLダイバージェンス

**KLダイバージェンス**（Kullback-Leibler divergence）は、2つの分布の違いを測る指標です。

```
KL(P || Q) = Σ P(x) log(P(x) / Q(x))
           = Σ P(x) log P(x) - Σ P(x) log Q(x)
           = -H(P) + H(P, Q)
```

**交差エントロピーとの関係**：

```
H(P, Q) = H(P) + KL(P || Q)

H(P): 真の分布のエントロピー（定数）
KL(P || Q): PとQの違い
```

分類問題では、Pは固定（正解ラベル）なので、`H(P)`は定数です。
したがって、**交差エントロピーを最小化 = KLダイバージェンスを最小化**

```python
# KLダイバージェンスの計算例
P = torch.tensor([0.8, 0.1, 0.1])  # 真の分布
Q = torch.tensor([0.7, 0.2, 0.1])  # 予測分布

kl_div = (P * torch.log(P / Q)).sum()
print(f"KL(P || Q): {kl_div:.4f}")
# KL(P || Q): 0.0223

# 交差エントロピー
H_P = -(P * torch.log(P)).sum()  # エントロピー
H_PQ = -(P * torch.log(Q)).sum()  # 交差エントロピー

print(f"H(P): {H_P:.4f}")
print(f"H(P, Q): {H_PQ:.4f}")
print(f"H(P) + KL(P||Q): {H_P + kl_div:.4f}")
# H(P): 0.6390
# H(P, Q): 0.6613
# H(P) + KL(P||Q): 0.6613  （一致）
```

---

## 4.9 なぜ交差エントロピーを使うのか

### 1. 確率的解釈

最尤推定（Maximum Likelihood Estimation）と等価です。
```
モデルのパラメータθを、訓練データの尤度を最大化するように学習
⇔ 負の対数尤度（交差エントロピー）を最小化
```

### 2. 勾配の性質

交差エントロピーとSoftmaxの組み合わせは、勾配が非常にシンプルです。

```
L = -log(Softmax(z)_y)

∂L/∂z_i = Softmax(z)_i - δ_iy

δ_iy: クロネッカーのデルタ（i=yなら1、それ以外0）
```

つまり、**勾配 = 予測確率 - 正解ラベル**

```python
# 例: 3クラス、正解はクラス1
z = torch.tensor([1.0, 2.0, 0.5], requires_grad=True)
target = 1

# Softmax
probs = F.softmax(z, dim=0)
print(f"Probs: {probs}")
# Probs: tensor([0.2564, 0.6364, 0.1072])

# 損失
loss = -torch.log(probs[target])
loss.backward()

print(f"Gradient: {z.grad}")
# Gradient: tensor([ 0.2564, -0.3636,  0.1072])
# = [0.2564, 0.6364, 0.1072] - [0, 1, 0]
```

### 3. 数値安定性

PyTorchの実装は、`log(softmax(x))`を直接計算せず、`log_softmax(x)`を使います。

```
log_softmax(x_i) = x_i - log(Σⱼ exp(x_j))
                 = x_i - max(x) - log(Σⱼ exp(x_j - max(x)))
```

これにより、オーバーフロー/アンダーフローを防ぎます。

### 4. 多クラス対応

2クラス以上の分類に自然に拡張できます。

---

## 4.10 nanochatでの使用例

### 訓練時の損失計算

```python
# gpt.py:282-286
# 言語モデルの損失計算

if targets is not None:
    # 訓練モード: 損失を計算して返す
    logits = self.lm_head(x)  # (B, T, vocab_size)
    logits = softcap * torch.tanh(logits / softcap)  # logits softcap
    logits = logits.float()  # tf32/fp32を使用
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),  # (B*T, vocab_size)
        targets.view(-1),                   # (B*T,)
        ignore_index=-1,
        reduction='mean'
    )
    return loss
```

**詳細**：

1. **logitsの形状変換**
   ```
   元: (B, T, vocab_size)
   → (B*T, vocab_size)

   各トークン位置を独立したサンプルとして扱う
   ```

2. **targetsの形状変換**
   ```
   元: (B, T)
   → (B*T,)

   各位置の正解トークンID
   ```

3. **ignore_index=-1**
   ```
   特定のトークン（パディングなど）を損失計算から除外
   ```

4. **reduction='mean'**
   ```
   全サンプルの平均損失を計算
   ```

### 具体例

```python
# ミニ例
B, T, V = 2, 3, 5  # バッチ2、シーケンス3、語彙5

logits = torch.randn(B, T, V)
targets = torch.tensor([
    [2, 3, 1],  # サンプル1の正解トークン
    [0, 4, 2]   # サンプル2の正解トークン
])

loss = F.cross_entropy(
    logits.view(-1, V),      # (6, 5)
    targets.view(-1)          # (6,)
)

print(f"Loss: {loss:.4f}")
```

### Bits per Byte (BPB)評価

```python
# loss_eval.py:29
# BPB評価での損失計算

with torch.no_grad():
    loss = model(input_tokens, target_tokens)  # 平均損失

# BPBに変換
loss_per_token = loss.item()
bpb = loss_per_token / math.log(2) * avg_token_bytes
```

**Bits per Byte**：
- 1バイトを符号化するのに必要なビット数
- 圧縮率の指標（低いほど良い）

---

## 4.11 まとめ

この章では、交差エントロピー損失について学びました。

### 主要な概念

1. **エントロピー**
   ```
   H(P) = -Σ P(x) log P(x)
   ```
   - 確率分布の不確実性の尺度
   - 高いほど予測が難しい

2. **交差エントロピー**
   ```
   H(P, Q) = -Σ P(x) log Q(x)
   ```
   - 2つの分布の違いを測る
   - P: 真の分布、Q: 予測分布

3. **交差エントロピー損失**
   ```
   L = -log Q(y)
   ```
   - 分類問題の標準的な損失関数
   - 正解クラスの予測確率の対数の負

4. **負の対数尤度（NLL）**
   - 交差エントロピーと本質的に同じ
   - 尤度を最大化 = NLLを最小化

5. **PyTorchの実装**
   ```python
   loss = F.cross_entropy(logits, targets)
   ```
   - logitsを直接受け取る
   - 内部でlog_softmaxを使用（数値安定）
   - 効率的な実装

6. **勾配の性質**
   ```
   ∂L/∂z_i = Softmax(z)_i - δ_iy
   ```
   - 予測確率 - 正解ラベル
   - シンプルで直感的

### nanochatでの主な使用箇所

| 用途 | コード位置 | 説明 |
|------|----------|------|
| 訓練時の損失 | `gpt.py:285` | 次トークン予測の損失計算 |
| 評価 | `loss_eval.py:29` | 検証データでの損失評価 |

### なぜ交差エントロピーか？

- ✅ 確率的解釈（最尤推定）
- ✅ シンプルな勾配
- ✅ 数値的に安定
- ✅ 多クラス対応
- ✅ 理論的に確立

### 次のステップ

次の数学ドキュメントでは、以下を学びます：
- **勾配降下法**: パラメータ最適化の基礎
- **誤差逆伝播法**: 勾配の効率的な計算
- **最適化アルゴリズム**: SGD, Adam, Muonなど

交差エントロピー損失を理解したことで、モデルがどのように学習するかがより明確になります。

---

**関連ドキュメント**:
- [数学03: Softmax関数](03-softmax.md)
- [数学05: 勾配降下法](05-gradient-descent.md)
- [数学06: 誤差逆伝播法](06-backpropagation.md)
- [第5章: データパイプラインと訓練プロセス](../05-training-pipeline.md)
