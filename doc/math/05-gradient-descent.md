# 数学05: 勾配降下法

## 目次
- [5.1 はじめに](#51-はじめに)
- [5.2 最適化問題とは](#52-最適化問題とは)
- [5.3 勾配とは](#53-勾配とは)
  - [5.3.1 1変数関数の微分](#531-1変数関数の微分)
  - [5.3.2 多変数関数の勾配](#532-多変数関数の勾配)
  - [5.3.3 勾配の意味](#533-勾配の意味)
- [5.4 勾配降下法の基本](#54-勾配降下法の基本)
  - [5.4.1 アルゴリズム](#541-アルゴリズム)
  - [5.4.2 学習率](#542-学習率)
  - [5.4.3 収束条件](#543-収束条件)
- [5.5 勾配降下法の種類](#55-勾配降下法の種類)
  - [5.5.1 バッチ勾配降下法](#551-バッチ勾配降下法)
  - [5.5.2 確率的勾配降下法（SGD）](#552-確率的勾配降下法sgd)
  - [5.5.3 ミニバッチ勾配降下法](#553-ミニバッチ勾配降下法)
- [5.6 学習率スケジューリング](#56-学習率スケジューリング)
  - [5.6.1 固定学習率](#561-固定学習率)
  - [5.6.2 学習率減衰](#562-学習率減衰)
  - [5.6.3 ウォームアップとウォームダウン](#563-ウォームアップとウォームダウン)
- [5.7 モメンタム](#57-モメンタム)
  - [5.7.1 基本的なモメンタム](#571-基本的なモメンタム)
  - [5.7.2 Nesterovモメンタム](#572-nesterovモメンタム)
- [5.8 勾配クリッピング](#58-勾配クリッピング)
- [5.9 実装例](#59-実装例)
- [5.10 nanochatでの使用例](#510-nanochatでの使用例)
- [5.11 まとめ](#511-まとめ)

---

## 5.1 はじめに

**勾配降下法**（Gradient Descent）は、機械学習における最も基本的な最適化アルゴリズムです。ニューラルネットワークの訓練では、損失関数を最小化するためにこの手法を使います。

この章では、勾配降下法の基礎から、実際の深層学習での応用まで学びます。

**前提知識**：
- [数学04: 交差エントロピー損失](04-cross-entropy.md)
- 微分の基礎（高校数学レベル）

---

## 5.2 最適化問題とは

機械学習の訓練は、**最適化問題**として定式化されます。

```
目的: 損失関数 L(θ) を最小化するパラメータ θ を見つける

min L(θ)
 θ
```

**例：線形回帰**
```
データ: (x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)
モデル: ŷ = w·x + b
損失: L(w, b) = (1/n) Σᵢ (yᵢ - ŷᵢ)²

最適化: L(w, b) を最小化する w, b を見つける
```

**例：ニューラルネットワーク**
```
データ: 訓練サンプル (x, y)
モデル: f(x; θ)  （θは全パラメータ）
損失: L(θ) = 交差エントロピー損失

最適化: L(θ) を最小化する θ を見つける
```

---

## 5.3 勾配とは

### 5.3.1 1変数関数の微分

**微分**は、関数の変化率を表します。

```
f(x) = x²

df/dx = 2x

x = 3 のとき、df/dx = 6
→ xが少し増えると、f(x)は6倍の速さで増える
```

```python
import torch

x = torch.tensor(3.0, requires_grad=True)
f = x ** 2

f.backward()  # 微分を計算
print(x.grad)  # tensor(6.)
```

### 5.3.2 多変数関数の勾配

**勾配**（Gradient）は、多変数関数の各変数に対する偏微分をまとめたベクトルです。

```
f(x, y) = x² + 2y²

∇f = [∂f/∂x, ∂f/∂y] = [2x, 4y]

(x, y) = (3, 2) のとき、∇f = [6, 8]
```

```python
x = torch.tensor(3.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)

f = x**2 + 2*y**2
f.backward()

print(f"∂f/∂x = {x.grad}")  # tensor(6.)
print(f"∂f/∂y = {y.grad}")  # tensor(8.)
```

### 5.3.3 勾配の意味

勾配は、**関数が最も急激に増加する方向**を指します。

```
┌─────────────────────────────────────────┐
│         Gradient の幾何学的意味           │
└─────────────────────────────────────────┘

        ↑ ∇f (勾配ベクトル)
        │
    ●───┘  現在位置
   /│\
  / │ \
 /  │  \
────────── 等高線

- 勾配の方向: 最も急な登り坂
- 勾配の大きさ: 傾きの急さ
- -勾配の方向: 最も急な下り坂（最小化の方向）
```

**重要な性質**：
- 勾配が0の点 = 極値（極小・極大・鞍点）
- 負の勾配方向に進む = 関数値が減少

---

## 5.4 勾配降下法の基本

### 5.4.1 アルゴリズム

**勾配降下法**は、勾配の逆方向（負の勾配方向）に少しずつパラメータを更新します。

```
繰り返し:
  1. 現在のパラメータ θ で損失 L(θ) を計算
  2. 勾配 ∇L(θ) を計算
  3. パラメータを更新: θ ← θ - η·∇L(θ)
```

ここで：
- `η` (イータ): **学習率**（Learning Rate）
- `∇L(θ)`: 損失関数の勾配

**視覚的イメージ**：
```
損失関数の地形

     高
      ↑
      │     ╱╲
      │    ╱  ╲
      │   ╱    ╲    ┌─ 初期位置
      │  ╱  ●   ╲   │
      │ ╱   │    ╲  │ 勾配降下で
      │╱    ↓     ╲ │ 徐々に下る
      ○─────●──────○ ← 最小値（目標）
     低

各ステップで勾配の逆方向に移動
```

### 5.4.2 学習率

**学習率** `η` は、1ステップでどれだけ移動するかを制御します。

```python
# 学習率の違い
theta = 10.0
grad = 4.0

# 学習率が小さい（η = 0.01）
theta_new1 = theta - 0.01 * grad
print(theta_new1)  # 9.96  - 少しずつ移動

# 学習率が大きい（η = 0.5）
theta_new2 = theta - 0.5 * grad
print(theta_new2)  # 8.0  - 大きく移動
```

**学習率の選び方**：

```
η が小さすぎる:
  ✗ 収束が遅い
  ✗ 局所最小値に陥りやすい

η が大きすぎる:
  ✗ 振動する
  ✗ 発散する（最小値を飛び越す）

η が適切:
  ✓ スムーズに収束
```

```
┌────────────────────────────────────────┐
│      学習率の影響（1次元の例）           │
└────────────────────────────────────────┘

損失

  │     小さすぎる（η=0.01）
  │     ●→●→●→●→●→●→●→ ゆっくり
  │
  │     適切（η=0.1）
  │     ●→→●→→●→○  スムーズに収束
  │
  │     大きすぎる（η=0.5）
  │     ●→→→→←←←→→  振動・発散
  │
  └──────────────────────────→ パラメータ
         ↑
       最小値
```

### 5.4.3 収束条件

勾配降下法の終了条件：

1. **最大イテレーション数**
   ```python
   for epoch in range(max_epochs):
       # 訓練
   ```

2. **勾配のノルムが小さい**
   ```python
   if grad.norm() < threshold:
       break
   ```

3. **損失の変化が小さい**
   ```python
   if abs(loss - prev_loss) < threshold:
       break
   ```

---

## 5.5 勾配降下法の種類

### 5.5.1 バッチ勾配降下法

**全訓練データ**を使って勾配を計算します。

```python
# 疑似コード
for epoch in range(num_epochs):
    # 全データで勾配を計算
    total_grad = 0
    for x, y in all_training_data:
        loss = compute_loss(model(x), y)
        total_grad += compute_gradient(loss)

    # 平均勾配で更新
    avg_grad = total_grad / len(all_training_data)
    theta = theta - learning_rate * avg_grad
```

**特徴**：
- ✓ 安定した収束
- ✓ 勾配の分散が小さい
- ✗ 遅い（全データの処理が必要）
- ✗ メモリを大量に消費
- ✗ オンライン学習ができない

### 5.5.2 確率的勾配降下法（SGD）

**1つのサンプル**だけで勾配を計算します。

```python
# 疑似コード
for epoch in range(num_epochs):
    # データをシャッフル
    shuffle(training_data)

    # 1サンプルずつ更新
    for x, y in training_data:
        loss = compute_loss(model(x), y)
        grad = compute_gradient(loss)
        theta = theta - learning_rate * grad
```

**特徴**：
- ✓ 速い（1サンプルのみ処理）
- ✓ オンライン学習が可能
- ✓ 局所最小値から脱出しやすい
- ✗ 勾配のノイズが大きい
- ✗ 収束が不安定

### 5.5.3 ミニバッチ勾配降下法

**小さなバッチ**で勾配を計算します（バッチとSGDの中間）。

```python
# 疑似コード
batch_size = 32

for epoch in range(num_epochs):
    shuffle(training_data)

    # ミニバッチごとに更新
    for batch in get_batches(training_data, batch_size):
        total_grad = 0
        for x, y in batch:
            loss = compute_loss(model(x), y)
            total_grad += compute_gradient(loss)

        avg_grad = total_grad / batch_size
        theta = theta - learning_rate * avg_grad
```

**特徴**：
- ✓ バッチGDとSGDの良いとこ取り
- ✓ GPU並列化が効率的
- ✓ 適度なノイズで局所最小値を回避
- ✓ 実用的で最も広く使われる

**現代の深層学習では、ほぼ全てミニバッチSGDです。**

---

## 5.6 学習率スケジューリング

### 5.6.1 固定学習率

最もシンプルな方法：

```python
learning_rate = 0.01

for epoch in range(num_epochs):
    # 常に同じ学習率
    theta = theta - learning_rate * grad
```

### 5.6.2 学習率減衰

訓練が進むにつれて学習率を**減少**させます。

**ステップ減衰**：
```python
initial_lr = 0.1
decay_rate = 0.5
decay_steps = 1000

for step in range(total_steps):
    lr = initial_lr * (decay_rate ** (step // decay_steps))
    theta = theta - lr * grad
```

**指数減衰**：
```python
lr = initial_lr * exp(-decay_rate * step)
```

**逆時間減衰**：
```python
lr = initial_lr / (1 + decay_rate * step)
```

### 5.6.3 ウォームアップとウォームダウン

nanochatで使用されている方法です。

```python
# base_train.py:148-157
warmup_ratio = 0.0    # ウォームアップなし
warmdown_ratio = 0.2  # 最後の20%でLRを減衰
final_lr_frac = 0.0   # 最終LRは0

def get_lr_multiplier(it):
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)

    if it < warmup_iters:
        # ウォームアップ: 線形増加
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        # 定常フェーズ: LR = 1.0
        return 1.0
    else:
        # ウォームダウン: 線形減衰
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac
```

**効果**：
```
LR
1.0 ┤──────────────────────────┐
    │                          ╲
    │                           ╲
    │                            ╲
0.0 ┤                             ╲──
    └─────────────────────────────────→ Iteration
    0%                  80%       100%
           定常              ウォームダウン
```

**利点**：
- 初期は大きな学習率で速く学習
- 後期は小さな学習率で fine-tuning

---

## 5.7 モメンタム

### 5.7.1 基本的なモメンタム

**モメンタム**は、過去の勾配の情報を利用して更新を加速します。

```
v_t = β·v_{t-1} + ∇L(θ_t)
θ_{t+1} = θ_t - η·v_t
```

- `v`: 速度（velocity）
- `β`: モメンタム係数（通常0.9）

**物理的アナロジー**：
```
ボールが坂を転がる:
  - 勾配: 坂の傾き
  - モメンタム: ボールの慣性
  → 過去の方向を保ちつつ加速
```

```python
# モメンタムSGD
v = 0
beta = 0.9

for step in range(num_steps):
    grad = compute_gradient()

    # 速度の更新
    v = beta * v + grad

    # パラメータの更新
    theta = theta - learning_rate * v
```

**効果**：
- 谷間を素早く横切る
- 振動を抑える
- 局所最小値を脱出しやすい

### 5.7.2 Nesterovモメンタム

**Nesterov加速勾配**（NAG）は、モメンタムの改良版です。

```
v_t = β·v_{t-1} + ∇L(θ_t - η·β·v_{t-1})
θ_{t+1} = θ_t - η·v_t
```

「次に行く場所」の勾配を使います。

```python
# Nesterov モメンタム
v = 0
beta = 0.9

for step in range(num_steps):
    # 「先読み」の位置で勾配を計算
    theta_lookahead = theta - learning_rate * beta * v
    grad = compute_gradient(theta_lookahead)

    # 速度の更新
    v = beta * v + grad

    # パラメータの更新
    theta = theta - learning_rate * v
```

**nanochatでの使用**：
```python
# muon.py:80-81
buf.lerp_(g, 1 - group["momentum"])
g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
```

---

## 5.8 勾配クリッピング

**勾配クリッピング**は、勾配が大きくなりすぎるのを防ぎます。

```python
# ノルムによるクリッピング
max_norm = 1.0
grad_norm = grad.norm()

if grad_norm > max_norm:
    grad = grad * (max_norm / grad_norm)

theta = theta - learning_rate * grad
```

**nanochatでの使用**：
```python
# base_train.py:265-266
grad_clip = 1.0

if grad_clip > 0.0:
    torch.nn.utils.clip_grad_norm_(orig_model.parameters(), grad_clip)
```

**効果**：
- 勾配爆発を防ぐ
- 訓練を安定化
- 特にRNNで重要

---

## 5.9 実装例

### シンプルなSGD

```python
import torch

# モデルパラメータ
theta = torch.randn(10, requires_grad=True)

# ハイパーパラメータ
learning_rate = 0.01

# 訓練ループ
for step in range(1000):
    # 順伝播
    loss = compute_loss(theta)

    # 逆伝播
    loss.backward()

    # パラメータ更新
    with torch.no_grad():
        theta -= learning_rate * theta.grad

    # 勾配をゼロ化
    theta.grad.zero_()
```

### PyTorchのOptimizer

```python
import torch.optim as optim

# モデル
model = MyModel()

# オプティマイザ
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 訓練ループ
for epoch in range(num_epochs):
    for batch in dataloader:
        # 順伝播
        loss = model(batch)

        # 逆伝播
        loss.backward()

        # パラメータ更新
        optimizer.step()

        # 勾配をゼロ化
        optimizer.zero_grad()
```

---

## 5.10 nanochatでの使用例

### 訓練ループ

```python
# base_train.py:252-281
for step in range(num_iterations + 1):
    # [1] 勾配累積ループ
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)

        loss = loss / grad_accum_steps
        loss.backward()  # 勾配計算

        x, y = next(train_loader)

    # [2] 勾配クリッピング
    if grad_clip > 0.0:
        torch.nn.utils.clip_grad_norm_(orig_model.parameters(), grad_clip)

    # [3] 学習率の更新
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm

    # [4] Muonモメンタムの更新
    muon_momentum = get_muon_momentum(step)
    for group in muon_optimizer.param_groups:
        group["momentum"] = muon_momentum

    # [5] オプティマイザーのステップ
    for opt in optimizers:
        opt.step()

    # [6] 勾配のゼロ化
    model.zero_grad(set_to_none=True)
```

### Muonモメンタムスケジューリング

```python
# base_train.py:160-163
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum
```

最初の300ステップで0.85から0.95に増加：
- 初期: 小さなモメンタム（探索重視）
- 後期: 大きなモメンタム（安定収束）

---

## 5.11 まとめ

この章では、勾配降下法について学びました。

### 主要な概念

1. **勾配**
   ```
   ∇L(θ) = [∂L/∂θ₁, ∂L/∂θ₂, ..., ∂L/∂θₙ]
   ```
   - 関数が最も急激に増加する方向
   - 負の勾配方向に進むと関数値が減少

2. **勾配降下法**
   ```
   θ ← θ - η·∇L(θ)
   ```
   - 勾配の逆方向にパラメータを更新
   - 学習率 η で更新幅を制御

3. **種類**
   - **バッチGD**: 全データで勾配計算（安定だが遅い）
   - **SGD**: 1サンプルで勾配計算（速いがノイズ大）
   - **ミニバッチGD**: 小バッチで勾配計算（実用的）

4. **学習率スケジューリング**
   - 固定学習率
   - 学習率減衰
   - ウォームアップ/ウォームダウン

5. **モメンタム**
   ```
   v_t = β·v_{t-1} + ∇L(θ_t)
   θ_{t+1} = θ_t - η·v_t
   ```
   - 過去の勾配情報を利用
   - 収束を加速

6. **勾配クリッピング**
   - 勾配のノルムを制限
   - 訓練の安定化

### nanochatでの主な使用箇所

| 要素 | コード位置 | 説明 |
|------|----------|------|
| 訓練ループ | `base_train.py:252-281` | 勾配計算と更新 |
| LRスケジューリング | `base_train.py:148-157` | ウォームダウン |
| モメンタムスケジューリング | `base_train.py:160-163` | Muonモメンタム |
| 勾配クリッピング | `base_train.py:265-266` | ノルムクリッピング |

### 実践的なヒント

- **学習率**: 最も重要なハイパーパラメータ
- **ミニバッチ**: GPUを効率的に使うため適切なサイズを選ぶ
- **モメンタム**: ほぼ常に使うべき（β=0.9が標準）
- **クリッピング**: RNNや大きなモデルで重要

### 次のステップ

次の数学ドキュメントでは、以下を学びます：
- **誤差逆伝播法**: 勾配の効率的な計算方法
- **最適化アルゴリズム**: Adam, AdamW, Muonなどの高度な手法

勾配降下法を理解したことで、これらの高度な最適化手法の基礎ができました。

---

**関連ドキュメント**:
- [数学04: 交差エントロピー損失](04-cross-entropy.md)
- [数学06: 誤差逆伝播法](06-backpropagation.md)
- [数学10: 最適化アルゴリズム](10-optimization-algorithms.md)
- [第6章: 最適化手法（Muon, AdamW）](../06-optimization.md)
