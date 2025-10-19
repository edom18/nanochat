# 数学03: Softmax関数

## 目次
- [3.1 はじめに](#31-はじめに)
- [3.2 Softmax関数とは](#32-softmax関数とは)
  - [3.2.1 定義](#321-定義)
  - [3.2.2 直感的な理解](#322-直感的な理解)
  - [3.2.3 具体例](#323-具体例)
- [3.3 なぜSoftmaxが必要か](#33-なぜsoftmaxが必要か)
- [3.4 数値安定性の問題](#34-数値安定性の問題)
  - [3.4.1 オーバーフローの問題](#341-オーバーフローの問題)
  - [3.4.2 安定版Softmax](#342-安定版softmax)
- [3.5 温度パラメータ](#35-温度パラメータ)
  - [3.5.1 温度スケーリング](#351-温度スケーリング)
  - [3.5.2 温度の効果](#352-温度の効果)
- [3.6 Softmaxの性質](#36-softmaxの性質)
- [3.7 LogSoftmax](#37-logsoftmax)
- [3.8 nanochatでの使用例](#38-nanochatでの使用例)
- [3.9 まとめ](#39-まとめ)

---

## 3.1 はじめに

**Softmax関数**は、ニューラルネットワークで最も頻繁に使われる関数の1つです。特に、分類問題や言語モデルにおいて、モデルの出力（logits）を**確率分布**に変換するために使用されます。

この章では、Softmax関数の仕組みと、nanochatでの使用例を学びます。

---

## 3.2 Softmax関数とは

### 3.2.1 定義

Softmax関数は、実数ベクトルを確率分布に変換します。

```
入力: z = [z₁, z₂, ..., zₙ] （任意の実数）
出力: y = [y₁, y₂, ..., yₙ] （確率、合計=1）

yᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)
```

各要素に対して：
1. 指数関数を適用: `exp(zᵢ)`
2. 全体の和で正規化: `Σⱼ exp(zⱼ)`

### 3.2.2 直感的な理解

```
┌────────────────────────────────────────┐
│           Softmax の働き                │
└────────────────────────────────────────┘

入力（logits）: [-2, 0, 3]
              任意の実数、範囲制限なし
              ↓ exp()
              [0.135, 1.0, 20.086]
              全て正の数に
              ↓ 正規化（合計で割る）
出力（確率）: [0.006, 0.047, 0.947]
              全て正、合計=1
```

**特徴**：
- 全ての出力が 0 〜 1 の範囲
- 全ての出力の合計が 1（確率分布）
- 大きな入力値ほど大きな確率
- 小さな入力値でも0にならない（常に正）

### 3.2.3 具体例

```python
import torch
import torch.nn.functional as F

# 入力: モデルの出力（logits）
logits = torch.tensor([1.0, 2.0, 3.0])

# Softmax
probs = F.softmax(logits, dim=0)
print(probs)
# tensor([0.0900, 0.2447, 0.6652])

# 合計は1
print(probs.sum())
# tensor(1.0000)
```

**計算過程**：
```
logits = [1.0, 2.0, 3.0]

1. exp()を適用:
   exp(1.0) = 2.718
   exp(2.0) = 7.389
   exp(3.0) = 20.086
   合計 = 30.193

2. 正規化:
   probs[0] = 2.718 / 30.193 = 0.0900
   probs[1] = 7.389 / 30.193 = 0.2447
   probs[2] = 20.086 / 30.193 = 0.6652
```

---

## 3.3 なぜSoftmaxが必要か

### 1. 確率解釈

ニューラルネットワークの出力（logits）は任意の実数です。これを確率として解釈するには、以下の条件が必要です：

```
- 各値が 0 ≤ p ≤ 1
- 全ての値の合計が 1
```

Softmaxはこれを満たします。

### 2. 微分可能

訓練には**勾配**が必要です。Softmaxは微分可能なため、誤差逆伝播法で学習できます。

### 3. 指数による強調

大きな値を指数的に強調し、小さな値を抑制します。

```
例:
logits = [1, 2, 3]    → probs = [0.09, 0.24, 0.67]
logits = [1, 2, 10]   → probs = [0.0001, 0.0002, 0.9997]

10が圧倒的に大きいと、確率もほぼ1.0になる
```

### 4. 言語モデルでの使用

```
┌────────────────────────────────────────┐
│      言語モデルのトークン予測            │
└────────────────────────────────────────┘

入力: "The capital of France is"
        ↓ モデル
logits: [2.1  ("Paris"),
         0.5  ("London"),
         -1.3 ("Berlin"),
         ...
         -5.2 ("Tokyo")]

        ↓ Softmax

probs:  [0.72 ("Paris"),
         0.15 ("London"),
         0.02 ("Berlin"),
         ...
         0.0001 ("Tokyo")]

→ "Paris"が最も高い確率
```

---

## 3.4 数値安定性の問題

### 3.4.1 オーバーフローの問題

素朴にSoftmaxを計算すると、`exp()`が非常に大きな値を生成し、**オーバーフロー**が発生します。

```python
import torch

# 大きな値
logits = torch.tensor([1000.0, 1001.0, 1002.0])

# 素朴なSoftmax（失敗する）
exp_logits = torch.exp(logits)
print(exp_logits)
# tensor([inf, inf, inf])  ← オーバーフロー

probs = exp_logits / exp_logits.sum()
print(probs)
# tensor([nan, nan, nan])  ← 計算不能
```

### 3.4.2 安定版Softmax

**解決策**：入力から最大値を引いてから計算

```
yᵢ = exp(zᵢ - max(z)) / Σⱼ exp(zⱼ - max(z))
```

**数学的に等価**：
```
exp(zᵢ - c) / Σⱼ exp(zⱼ - c)
= [exp(zᵢ) / exp(c)] / [Σⱼ exp(zⱼ) / exp(c)]
= exp(zᵢ) / Σⱼ exp(zⱼ)
```

cに何を使っても結果は同じですが、`c = max(z)`とすると数値的に安定します。

```python
# 安定版Softmax
logits = torch.tensor([1000.0, 1001.0, 1002.0])

# 最大値を引く
logits_stable = logits - logits.max()
print(logits_stable)
# tensor([-2., -1.,  0.])  ← 扱いやすい範囲

exp_logits = torch.exp(logits_stable)
probs = exp_logits / exp_logits.sum()
print(probs)
# tensor([0.0900, 0.2447, 0.6652])  ← 正しく計算できた
```

**PyTorchのF.softmax()は自動的に安定版を使用**します。

---

## 3.5 温度パラメータ

### 3.5.1 温度スケーリング

**温度**（Temperature）`T` を使って、確率分布の鋭さを制御できます。

```
yᵢ = exp(zᵢ / T) / Σⱼ exp(zⱼ / T)
```

- `T = 1`: 通常のSoftmax
- `T < 1`: 分布が鋭くなる（高確率のトークンに集中）
- `T > 1`: 分布が平坦になる（多様性が増す）

### 3.5.2 温度の効果

```python
logits = torch.tensor([1.0, 2.0, 3.0])

# T = 1（通常）
probs_t1 = F.softmax(logits / 1.0, dim=0)
print(probs_t1)
# tensor([0.0900, 0.2447, 0.6652])

# T = 0.5（低温：鋭い分布）
probs_t05 = F.softmax(logits / 0.5, dim=0)
print(probs_t05)
# tensor([0.0158, 0.1173, 0.8668])
# → 最大値（3.0）に集中

# T = 2.0（高温：平坦な分布）
probs_t2 = F.softmax(logits / 2.0, dim=0)
print(probs_t2)
# tensor([0.1863, 0.3072, 0.5065])
# → より均等に分散
```

**視覚化**：
```
T = 0.5 (低温):   ▁▃█  鋭い
T = 1.0 (標準):   ▃▅█  標準
T = 2.0 (高温):   ▅▆█  平坦
T → ∞:            ███  完全に均一
T → 0:            ▁▁█  ほぼone-hot
```

### nanochatでの使用

```python
# engine.py:137-138, 142-143
# 推論時の温度スケーリング

if temperature > 0:
    logits = logits / temperature  # 温度で割る
    probs = F.softmax(logits, dim=-1)
```

**温度の使い分け**（第7章参照）：
- `T = 0.0`: 決定的（最大値を選択）
- `T = 0.3-0.7`: 事実的な回答
- `T = 0.8-1.0`: バランス
- `T = 1.2-1.5`: 創造的

---

## 3.6 Softmaxの性質

### 1. 不変性

入力に定数を加えても、結果は変わりません。

```python
logits = torch.tensor([1.0, 2.0, 3.0])

probs1 = F.softmax(logits, dim=0)
probs2 = F.softmax(logits + 100, dim=0)

print(torch.allclose(probs1, probs2))  # True
```

### 2. 順序保存

logitsの順序と確率の順序は一致します。

```
z₁ > z₂  ⇔  y₁ > y₂
```

### 3. 微分

Softmaxの微分（ヤコビアン）は：

```
∂yᵢ/∂zⱼ = yᵢ(δᵢⱼ - yⱼ)

δᵢⱼ: クロネッカーのデルタ（i=jなら1、それ以外0）
```

```python
# 自動微分で確認
logits = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
probs = F.softmax(logits, dim=0)

# probs[0]に対するlogitsの勾配
probs[0].backward(retain_graph=True)
print(logits.grad)
# tensor([0.0819, -0.0220, -0.0598])
```

### 4. 最大値近似

温度が0に近づくと、最大値の位置が1、他が0に近づきます（one-hot）。

```python
logits = torch.tensor([1.0, 2.0, 3.0])

for T in [1.0, 0.1, 0.01]:
    probs = F.softmax(logits / T, dim=0)
    print(f"T={T}: {probs}")

# T=1.0: tensor([0.0900, 0.2447, 0.6652])
# T=0.1: tensor([0.0000, 0.0047, 0.9953])
# T=0.01: tensor([0.0000, 0.0000, 1.0000])
```

---

## 3.7 LogSoftmax

**LogSoftmax**は、Softmaxの対数です。

```
log_softmax(zᵢ) = log(yᵢ) = log(exp(zᵢ) / Σⱼ exp(zⱼ))
                = zᵢ - log(Σⱼ exp(zⱼ))
```

### なぜLog-Softmaxか？

1. **数値安定性**
   - Softmaxの確率が非常に小さい場合、`log(0)`に近づく
   - LogSoftmaxは直接計算するため安定

2. **計算効率**
   - 交差エントロピー損失との組み合わせで高速

3. **勾配の安定性**
   - 対数空間での計算が安定

```python
logits = torch.tensor([1.0, 2.0, 3.0])

# Softmax → log
probs = F.softmax(logits, dim=0)
log_probs1 = torch.log(probs)

# LogSoftmax（直接）
log_probs2 = F.log_softmax(logits, dim=0)

print(log_probs1)
# tensor([-2.4076, -1.4076, -0.4076])
print(log_probs2)
# tensor([-2.4076, -1.4076, -0.4076])

print(torch.allclose(log_probs1, log_probs2))  # True
```

### nanochatでの使用

```python
# gpt.py:285
# 損失計算にはPyTorchの交差エントロピーを使用
# 内部でlog_softmaxが効率的に計算される

loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                       targets.view(-1),
                       ignore_index=-1)
```

---

## 3.8 nanochatでの使用例

### 1. Attentionでのソフトマックス

```python
# gpt.py:119-120
# Attention重みの計算

# attn: (B, H, T, T) - 各トークンペアのスコア
attn = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)

# Softmax: 各クエリに対して、全キーの重みを確率分布に
attn = F.softmax(attn, dim=-1)  # 最後の次元でSoftmax

# 各行が確率分布（合計=1）になる
```

**意味**：
```
各クエリトークンが、どのキートークンに注目するかの確率分布

例: T=3の場合
attn[0, 0] =
  [[0.7, 0.2, 0.1],   ← トークン0は自分(0)に70%注目
   [0.3, 0.6, 0.1],   ← トークン1は自分(1)に60%注目
   [0.2, 0.3, 0.5]]   ← トークン2は自分(2)に50%注目

各行の合計 = 1.0
```

### 2. 推論時のサンプリング

```python
# engine.py:137-144
# トークンのサンプリング

if top_k is not None:
    # Top-kフィルタリング
    vals, idx = torch.topk(logits, k, dim=-1)
    vals = vals / temperature  # 温度スケーリング
    probs = F.softmax(vals, dim=-1)  # Softmax
    choice = torch.multinomial(probs, num_samples=1)
    return idx.gather(1, choice)
else:
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

### 3. 訓練時の損失計算

```python
# gpt.py:284-285
# 交差エントロピー損失（内部でlog_softmaxを使用）

logits = self.lm_head(x)  # (B, T, vocab_size)
logits = logits.float()

loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),  # (B*T, vocab_size)
    targets.view(-1),                   # (B*T,)
    ignore_index=-1
)
```

**内部の処理**：
```
1. log_softmax(logits) で対数確率を計算
2. 正解トークンの対数確率を抽出
3. 負の対数尤度（Negative Log-Likelihood）を計算
```

---

## 3.9 まとめ

この章では、Softmax関数について学びました。

### 主要な概念

1. **Softmax関数の定義**
   ```
   yᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)
   ```
   - 任意の実数ベクトルを確率分布に変換
   - 全ての出力が正、合計が1

2. **役割**
   - ニューラルネットワークの出力を確率解釈
   - 微分可能（訓練可能）
   - 大きな値を指数的に強調

3. **数値安定性**
   - 素朴な実装はオーバーフロー
   - 安定版: 最大値を引いてから計算
   - PyTorchは自動的に安定版を使用

4. **温度パラメータ**
   ```
   yᵢ = exp(zᵢ / T) / Σⱼ exp(zⱼ / T)
   ```
   - T < 1: 鋭い分布（確信度高）
   - T > 1: 平坦な分布（多様性高）
   - T → 0: one-hot（決定的）

5. **Log-Softmax**
   - Softmaxの対数
   - 数値的に安定
   - 交差エントロピー損失と組み合わせて効率的

### nanochatでの主な使用箇所

| 用途 | コード位置 | 説明 |
|------|----------|------|
| Attention重み | `gpt.py:120` | Attentionスコアを確率分布に |
| トークンサンプリング | `engine.py:138, 143` | logitsから次のトークンを確率的に選択 |
| 損失計算 | `gpt.py:285` | 交差エントロピー（内部でlog_softmax） |

### 次のステップ

次の数学ドキュメントでは、以下を学びます：
- **交差エントロピー損失**: 分類タスクの損失関数
- **勾配降下法**: パラメータ最適化の基礎
- **Attention機構の数式**: Q, K, Vの詳細な計算

Softmax関数を理解したことで、これらの概念がより明確になります。

---

**関連ドキュメント**:
- [数学04: 交差エントロピー損失](04-cross-entropy.md)
- [数学07: Attention機構の数式](07-attention-math.md)
- [数学12: 確率的サンプリング](12-probability-sampling.md)
- [第7章: 推論エンジンとサンプリング](../07-inference.md)
