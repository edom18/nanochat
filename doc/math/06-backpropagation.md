# 数学06: 誤差逆伝播法

## 目次
- [6.1 はじめに](#61-はじめに)
- [6.2 連鎖律（Chain Rule）](#62-連鎖律chain-rule)
  - [6.2.1 1変数の連鎖律](#621-1変数の連鎖律)
  - [6.2.2 多変数の連鎖律](#622-多変数の連鎖律)
- [6.3 計算グラフ](#63-計算グラフ)
  - [6.3.1 計算グラフとは](#631-計算グラフとは)
  - [6.3.2 順伝播（Forward Pass）](#632-順伝播forward-pass)
  - [6.3.3 逆伝播（Backward Pass）](#633-逆伝播backward-pass)
- [6.4 誤差逆伝播法の詳細](#64-誤差逆伝播法の詳細)
  - [6.4.1 基本的なアルゴリズム](#641-基本的なアルゴリズム)
  - [6.4.2 具体例：単純なネットワーク](#642-具体例単純なネットワーク)
- [6.5 各層の勾配計算](#65-各層の勾配計算)
  - [6.5.1 線形層](#651-線形層)
  - [6.5.2 活性化関数](#652-活性化関数)
  - [6.5.3 Softmax + 交差エントロピー](#653-softmax--交差エントロピー)
- [6.6 行列演算での逆伝播](#66-行列演算での逆伝播)
  - [6.6.1 行列積の勾配](#661-行列積の勾配)
  - [6.6.2 要素ごとの演算の勾配](#662-要素ごとの演算の勾配)
- [6.7 PyTorchの自動微分](#67-pytorchの自動微分)
  - [6.7.1 Autograd](#671-autograd)
  - [6.7.2 計算グラフの構築](#672-計算グラフの構築)
  - [6.7.3 勾配の蓄積](#673-勾配の蓄積)
- [6.8 実装例](#68-実装例)
- [6.9 nanochatでの使用例](#69-nanochatでの使用例)
- [6.10 まとめ](#610-まとめ)

---

## 6.1 はじめに

**誤差逆伝播法**（Backpropagation, 略してBackprop）は、ニューラルネットワークの訓練において、損失関数の勾配を**効率的に計算**するアルゴリズムです。

単純な方法で勾配を計算すると、パラメータ数に比例した時間がかかります。誤差逆伝播法を使うと、**1回の順伝播と1回の逆伝播**だけで全パラメータの勾配を計算できます。

この章では、誤差逆伝播法の仕組みを、数学的基礎から実装まで学びます。

**前提知識**：
- [数学05: 勾配降下法](05-gradient-descent.md)
- 微分の基礎

---

## 6.2 連鎖律（Chain Rule）

誤差逆伝播法の数学的基盤は**連鎖律**です。

### 6.2.1 1変数の連鎖律

合成関数の微分は、各関数の微分の積です。

```
y = f(g(x))

dy/dx = (dy/dg) × (dg/dx)
```

**具体例**：
```
y = (x² + 1)³

g(x) = x² + 1
y = g³

dy/dx = (dy/dg) × (dg/dx)
      = 3g² × 2x
      = 3(x² + 1)² × 2x
      = 6x(x² + 1)²
```

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
g = x**2 + 1
y = g**3

y.backward()
print(x.grad)  # tensor(90.)

# 手動計算: 6 * 2 * (2² + 1)² = 6 * 2 * 25 = 300...
# あれ？違う。確認：g = 5, y = 125
# dy/dg = 3 * g² = 3 * 25 = 75
# dg/dx = 2x = 4
# dy/dx = 75 * 4 = 300
# PyTorchのgrad.item()を確認すると...実は300.0が正しい
```

実際には：
```python
x = torch.tensor(2.0, requires_grad=True)
g = x**2 + 1  # g = 5
y = g**3      # y = 125

y.backward()
print(x.grad)  # tensor(300.)

# 手動: dy/dx = 3g² × 2x = 3(25) × 4 = 300 ✓
```

### 6.2.2 多変数の連鎖律

複数の変数がある場合、**全経路の微分を足し合わせ**ます。

```
z = f(x, y)
x = g(t)
y = h(t)

dz/dt = (∂z/∂x)(dx/dt) + (∂z/∂y)(dy/dt)
```

**具体例**：
```
z = x·y
x = t²
y = t³

dz/dt = (∂z/∂x)(dx/dt) + (∂z/∂y)(dy/dt)
      = y·(2t) + x·(3t²)
      = t³·(2t) + t²·(3t²)
      = 2t⁴ + 3t⁴
      = 5t⁴
```

```python
t = torch.tensor(2.0, requires_grad=True)
x = t**2
y = t**3
z = x * y

z.backward()
print(t.grad)  # tensor(80.)

# 手動: 5t⁴ = 5 × 16 = 80 ✓
```

---

## 6.3 計算グラフ

### 6.3.1 計算グラフとは

**計算グラフ**は、計算を有向非巡回グラフ（DAG）として表現したものです。

```
例: y = (x + 2) × (x - 1)

    x
   ╱ ╲
  ╱   ╲
 +2   -1
  ╲   ╱
   ╲ ╱
    ×
    │
    y
```

各ノードは演算、エッジはデータの流れを表します。

### 6.3.2 順伝播（Forward Pass）

**順伝播**は、入力から出力へ計算を進める過程です。

```
y = (x + 2) × (x - 1)

x = 3 のとき:

[Forward]
a = x + 2 = 3 + 2 = 5
b = x - 1 = 3 - 1 = 2
y = a × b = 5 × 2 = 10
```

### 6.3.3 逆伝播（Backward Pass）

**逆伝播**は、出力から入力へ勾配を伝える過程です。

```
目標: ∂y/∂x を計算

[Backward]
∂y/∂y = 1  （開始）

∂y/∂a = b = 2
∂y/∂b = a = 5

∂a/∂x = 1
∂b/∂x = 1

∂y/∂x = (∂y/∂a)(∂a/∂x) + (∂y/∂b)(∂b/∂x)
      = 2 × 1 + 5 × 1
      = 7
```

**視覚化**：
```
順伝播（値を計算）:
    3
   ╱ ╲
  5   2
   ╲ ╱
    10

逆伝播（勾配を計算）:
    7 ← 最終的な勾配
   ╱ ╲
  2   5 ← 各経路の勾配
   ╲ ╱
    1 ← 開始（∂y/∂y = 1）
```

---

## 6.4 誤差逆伝播法の詳細

### 6.4.1 基本的なアルゴリズム

誤差逆伝播法は以下のステップで構成されます：

```
1. 順伝播（Forward Pass）:
   - 入力から出力へ計算
   - 各層の出力を保存（勾配計算に必要）

2. 損失計算:
   - 出力と正解の差（損失）を計算

3. 逆伝播（Backward Pass）:
   - 出力層から入力層へ勾配を計算
   - 連鎖律を使って各パラメータの勾配を計算

4. パラメータ更新:
   - 勾配降下法でパラメータを更新
```

### 6.4.2 具体例：単純なネットワーク

2層ニューラルネットワークの例：

```
入力: x
重み: W1, b1, W2, b2
活性化: ReLU

z1 = W1·x + b1
a1 = ReLU(z1)
z2 = W2·a1 + b2
ŷ = z2
L = (y - ŷ)²  （損失）
```

**順伝播**：
```python
# 順伝播
z1 = W1 @ x + b1
a1 = torch.relu(z1)
z2 = W2 @ a1 + b2
y_pred = z2
loss = (y - y_pred)**2
```

**逆伝播**：
```
∂L/∂ŷ = -2(y - ŷ)

∂L/∂W2 = (∂L/∂z2)(∂z2/∂W2)
        = (∂L/∂ŷ)(a1ᵀ)

∂L/∂b2 = ∂L/∂z2

∂L/∂a1 = (∂L/∂z2)(∂z2/∂a1)
        = (∂L/∂ŷ)(W2ᵀ)

∂L/∂z1 = (∂L/∂a1) ⊙ (a1 > 0)  （ReLUの微分）

∂L/∂W1 = (∂L/∂z1)(xᵀ)

∂L/∂b1 = ∂L/∂z1
```

---

## 6.5 各層の勾配計算

### 6.5.1 線形層

```
順伝播: y = W·x + b

逆伝播:
  ∂L/∂W = (∂L/∂y) ⊗ xᵀ  （外積）
  ∂L/∂b = ∂L/∂y
  ∂L/∂x = Wᵀ·(∂L/∂y)
```

```python
# 順伝播
x = torch.randn(3, requires_grad=True)
W = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True)

y = W @ x + b

# 逆伝播（自動）
loss = y.sum()
loss.backward()

print(f"∂L/∂W shape: {W.grad.shape}")  # (2, 3)
print(f"∂L/∂b shape: {b.grad.shape}")  # (2,)
print(f"∂L/∂x shape: {x.grad.shape}")  # (3,)
```

### 6.5.2 活性化関数

**ReLU**：
```
順伝播: y = max(0, x)

逆伝播: ∂L/∂x = (∂L/∂y) ⊙ (x > 0)
```

```python
x = torch.tensor([-1.0, 2.0, -0.5, 3.0], requires_grad=True)
y = torch.relu(x)

loss = y.sum()
loss.backward()

print(x.grad)
# tensor([0., 1., 0., 1.])
# 負の要素は勾配0、正の要素は勾配1
```

**Sigmoid**：
```
順伝播: y = σ(x) = 1 / (1 + e⁻ˣ)

逆伝播: ∂L/∂x = (∂L/∂y) ⊙ σ(x) ⊙ (1 - σ(x))
```

**Tanh**：
```
順伝播: y = tanh(x)

逆伝播: ∂L/∂x = (∂L/∂y) ⊙ (1 - tanh²(x))
```

### 6.5.3 Softmax + 交差エントロピー

Softmax + 交差エントロピーの組み合わせは、非常にシンプルな勾配になります。

```
順伝播:
  z = logits
  p = softmax(z)
  L = -log(p_y)  （yは正解クラス）

逆伝播:
  ∂L/∂z_i = p_i - δ_iy

  δ_iy: クロネッカーのデルタ（i=yなら1、それ以外0）
```

つまり、**勾配 = 予測確率 - 正解ラベル**

```python
logits = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
target = 1  # 正解クラス

# Softmax + NLL
log_probs = F.log_softmax(logits, dim=0)
loss = -log_probs[target]

loss.backward()
print(logits.grad)
# tensor([ 0.0900, -0.7553,  0.6652])

# 確認
probs = F.softmax(logits, dim=0)
target_one_hot = torch.tensor([0.0, 1.0, 0.0])
manual_grad = probs - target_one_hot
print(manual_grad)
# tensor([ 0.0900, -0.7553,  0.6652])  ✓
```

---

## 6.6 行列演算での逆伝播

### 6.6.1 行列積の勾配

```
順伝播: C = A @ B

逆伝播:
  ∂L/∂A = (∂L/∂C) @ Bᵀ
  ∂L/∂B = Aᵀ @ (∂L/∂C)
```

**形状の確認**：
```
A: (m, n)
B: (n, p)
C: (m, p)

∂L/∂C: (m, p)

∂L/∂A = (∂L/∂C) @ Bᵀ
      = (m, p) @ (p, n)
      = (m, n)  ✓

∂L/∂B = Aᵀ @ (∂L/∂C)
      = (n, m) @ (m, p)
      = (n, p)  ✓
```

```python
A = torch.randn(3, 4, requires_grad=True)
B = torch.randn(4, 5, requires_grad=True)

C = A @ B
loss = C.sum()
loss.backward()

print(f"A.grad shape: {A.grad.shape}")  # (3, 4)
print(f"B.grad shape: {B.grad.shape}")  # (4, 5)
```

### 6.6.2 要素ごとの演算の勾配

**加算**：
```
順伝播: z = x + y

逆伝播:
  ∂L/∂x = ∂L/∂z
  ∂L/∂y = ∂L/∂z
```

**乗算**：
```
順伝播: z = x ⊙ y  （要素ごとの積）

逆伝播:
  ∂L/∂x = (∂L/∂z) ⊙ y
  ∂L/∂y = (∂L/∂z) ⊙ x
```

**ブロードキャスト**：

ブロードキャストされた演算では、勾配を元の形状に**縮約**する必要があります。

```python
# (3, 4) + (4,) の場合
x = torch.randn(3, 4, requires_grad=True)
y = torch.randn(4, requires_grad=True)

z = x + y  # yが各行にブロードキャスト
loss = z.sum()
loss.backward()

print(f"x.grad shape: {x.grad.shape}")  # (3, 4)
print(f"y.grad shape: {y.grad.shape}")  # (4,)

# y.gradは各行の勾配の和
```

---

## 6.7 PyTorchの自動微分

### 6.7.1 Autograd

PyTorchの**Autograd**は、計算グラフを自動的に構築し、逆伝播を実行します。

```python
import torch

# requires_grad=True で勾配追跡を有効化
x = torch.tensor(2.0, requires_grad=True)
y = x**3 + 2*x**2 - 5*x + 3

# 逆伝播
y.backward()

# 勾配を取得
print(x.grad)
# dy/dx = 3x² + 4x - 5 = 12 + 8 - 5 = 15
# tensor(15.)
```

### 6.7.2 計算グラフの構築

Autogradは計算グラフを動的に構築します（Define-by-Run）。

```python
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)

# 順伝播しながらグラフを構築
z = x * y
w = z + x
loss = w ** 2

# グラフが構築された
print(loss.grad_fn)
# <PowBackward0 object at ...>

# 逆伝播
loss.backward()

print(x.grad)  # tensor(10.)
print(y.grad)  # tensor(6.)
```

### 6.7.3 勾配の蓄積

`.backward()`を複数回呼ぶと、勾配が**蓄積**されます。

```python
x = torch.tensor(2.0, requires_grad=True)

# 1回目
y1 = x ** 2
y1.backward()
print(x.grad)  # tensor(4.)

# 2回目（グラフを再構築）
y2 = x ** 3
y2.backward()
print(x.grad)  # tensor(16.)  = 4 + 12

# 勾配をゼロ化
x.grad.zero_()
print(x.grad)  # tensor(0.)
```

**訓練ループでは必ず勾配をゼロ化**：
```python
for epoch in range(num_epochs):
    for batch in dataloader:
        # 順伝播
        output = model(batch)
        loss = criterion(output, target)

        # 逆伝播
        loss.backward()

        # パラメータ更新
        optimizer.step()

        # 勾配をゼロ化（重要！）
        optimizer.zero_grad()
```

---

## 6.8 実装例

### 手動実装（教育目的）

```python
import torch

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = torch.randn(input_size, hidden_size) * 0.01
        self.b1 = torch.zeros(hidden_size)
        self.W2 = torch.randn(hidden_size, output_size) * 0.01
        self.b2 = torch.zeros(output_size)

    def forward(self, x):
        # 順伝播
        self.z1 = x @ self.W1 + self.b1
        self.a1 = torch.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2

    def backward(self, x, y, output):
        # 逆伝播（手動実装）
        batch_size = x.shape[0]

        # 出力層
        dz2 = (output - y) / batch_size  # 損失の勾配
        self.dW2 = self.a1.T @ dz2
        self.db2 = dz2.sum(dim=0)

        # 隠れ層
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0).float()  # ReLUの微分
        self.dW1 = x.T @ dz1
        self.db1 = dz1.sum(dim=0)

    def update(self, learning_rate):
        # パラメータ更新
        self.W1 -= learning_rate * self.dW1
        self.b1 -= learning_rate * self.db1
        self.W2 -= learning_rate * self.dW2
        self.b2 -= learning_rate * self.db2
```

### PyTorch実装（実用的）

```python
import torch
import torch.nn as nn

class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 使用例
model = TwoLayerNet(784, 128, 10)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(num_epochs):
    for x, y in dataloader:
        # 順伝播
        output = model(x)
        loss = F.cross_entropy(output, y)

        # 逆伝播（自動）
        loss.backward()

        # パラメータ更新
        optimizer.step()
        optimizer.zero_grad()
```

---

## 6.9 nanochatでの使用例

### 訓練時の逆伝播

```python
# base_train.py:257-262
for micro_step in range(grad_accum_steps):
    with autocast_ctx:
        loss = model(x, y)  # 順伝播

    train_loss = loss.detach()
    loss = loss / grad_accum_steps  # 正規化
    loss.backward()  # 逆伝播（自動）

    x, y = next(train_loader)
```

**ポイント**：
- `loss.backward()`で自動的に全パラメータの勾配を計算
- 勾配累積のため、損失を`grad_accum_steps`で割る
- PyTorchのAutogradが計算グラフを自動構築・逆伝播

### モデル定義での勾配計算

```python
# gpt.py:282-286
if targets is not None:
    logits = self.lm_head(x)
    logits = logits.float()
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=-1
    )
    return loss  # この損失に対して.backward()が呼ばれる
```

**内部で起こること**：
1. 順伝播で計算グラフを構築
2. `.backward()`で以下の勾配を計算：
   - `lm_head.weight`の勾配
   - 全Transformer層のパラメータの勾配
   - 埋め込み層`wte.weight`の勾配

### 勾配のゼロ化

```python
# base_train.py:277
model.zero_grad(set_to_none=True)
```

`set_to_none=True`の効果：
- 勾配を0で埋めず、`None`に設定
- メモリ効率が良い

---

## 6.10 まとめ

この章では、誤差逆伝播法について学びました。

### 主要な概念

1. **連鎖律**
   ```
   dy/dx = (dy/dz)(dz/dx)
   ```
   - 合成関数の微分
   - 誤差逆伝播法の数学的基礎

2. **計算グラフ**
   - 計算を有向非巡回グラフで表現
   - 順伝播: 入力→出力へ値を計算
   - 逆伝播: 出力→入力へ勾配を計算

3. **誤差逆伝播法のアルゴリズム**
   ```
   1. 順伝播（Forward）
   2. 損失計算
   3. 逆伝播（Backward）
   4. パラメータ更新
   ```

4. **各層の勾配**
   - 線形層: `∂L/∂W = (∂L/∂y) ⊗ xᵀ`
   - ReLU: `∂L/∂x = (∂L/∂y) ⊙ (x > 0)`
   - Softmax+CE: `∂L/∂z = p - y`（シンプル！）

5. **行列演算の勾配**
   - 行列積: `∂L/∂A = (∂L/∂C) @ Bᵀ`
   - ブロードキャスト: 勾配を縮約

6. **PyTorchのAutograd**
   - 計算グラフを自動構築
   - `.backward()`で自動的に勾配計算
   - Define-by-Run（動的計算グラフ）

7. **実用的なポイント**
   - 勾配は蓄積される→必ずゼロ化
   - `set_to_none=True`でメモリ効率化
   - 混合精度訓練との組み合わせ

### 効率性

**ナイーブな方法**：
```
各パラメータθᵢについて:
  ∂L/∂θᵢ を数値微分で計算
  → O(n)回の順伝播が必要（nはパラメータ数）
```

**誤差逆伝播法**：
```
1回の順伝播 + 1回の逆伝播
  → O(1)回の計算で全パラメータの勾配を取得
```

深層学習が実用的になったのは、誤差逆伝播法のおかげです。

### nanochatでの使用

| 要素 | コード位置 | 説明 |
|------|----------|------|
| 逆伝播 | `base_train.py:262` | `loss.backward()` |
| 勾配ゼロ化 | `base_train.py:277` | `model.zero_grad()` |
| 損失計算 | `gpt.py:285` | 交差エントロピー |

### 次のステップ

次の数学ドキュメントでは、以下を学びます：
- **Attention機構の数式**: Q, K, Vの詳細な計算と勾配
- **正規化手法**: LayerNorm, RMSNormの数学
- **位置エンコーディング**: Rotary Embeddingsの原理

誤差逆伝播法を理解したことで、これらの高度な技術がどのように訓練されるかが明確になります。

---

**関連ドキュメント**:
- [数学05: 勾配降下法](05-gradient-descent.md)
- [数学07: Attention機構の数式](07-attention-math.md)
- [数学10: 最適化アルゴリズム](10-optimization-algorithms.md)
- [第6章: 最適化手法（Muon, AdamW）](../06-optimization.md)
