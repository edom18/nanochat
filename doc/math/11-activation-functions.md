# 第11章: 活性化関数

## 目次
- [活性化関数とは何か](#活性化関数とは何か)
- [古典的な活性化関数](#古典的な活性化関数)
  - [Sigmoid](#sigmoid)
  - [Tanh](#tanh)
- [ReLUファミリー](#reluファミリー)
  - [ReLU](#relu)
  - [Leaky ReLU](#leaky-relu)
  - [ELU](#elu)
  - [GELU](#gelu)
- [現代的な活性化関数](#現代的な活性化関数)
  - [Swish/SiLU](#swishsilu)
  - [Squared ReLU](#squared-relu)
- [比較とまとめ](#比較とまとめ)
- [nanochatでの実装](#nanochatでの実装)
- [練習問題](#練習問題)

---

## 活性化関数とは何か

**活性化関数（Activation Function）** は、ニューラルネットワークの各層に**非線形性**を導入する関数です。

### なぜ非線形性が必要か？

線形変換のみでは、どれだけ層を深くしても**単一の線形変換と等価**になります。

#### 証明

```
2層の線形ネットワーク:
  h = W_1 x
  y = W_2 h = W_2 (W_1 x) = (W_2 W_1) x = W x

W = W_2 W_1 という単一の行列と等価！

深さが無意味になる
```

#### 非線形性の導入

```
活性化関数 f を挿入:
  h = f(W_1 x)
  y = f(W_2 h)

f が非線形なら、単一の線形変換に還元できない
→ 表現力が向上
```

### 可視化

```
線形のみ:
  入力 → W_1 → W_2 → W_3 → 出力
         ↓
  入力 → W → 出力  （等価）

非線形あり:
  入力 → W_1 → f → W_2 → f → W_3 → 出力
         ↓
  複雑な関数を表現可能（XOR、曲線など）
```

### 活性化関数の役割

1. **非線形性の導入**: 複雑な関数を表現可能に
2. **勾配の制御**: 逆伝播時の勾配の流れを調整
3. **出力の範囲制御**: 活性化値を適切な範囲に制限

---

## 古典的な活性化関数

### Sigmoid

**Sigmoid** は、出力を `(0, 1)` の範囲に圧縮します。

#### 数式

```
σ(x) = 1 / (1 + e^(-x))
```

#### グラフ

```
 1.0 ┤         ╭─────
     │       ╭─╯
 0.5 ┤    ╭──╯
     │  ╭─╯
 0.0 ┤──╯
     └───────────────→ x
    -5   0   5
```

#### 特性

```
範囲: (0, 1)
σ(0) = 0.5
σ(+∞) = 1
σ(-∞) = 0

対称性:
  σ(-x) = 1 - σ(x)
```

#### 微分

```
σ'(x) = σ(x) (1 - σ(x))

x=0 で最大: σ'(0) = 0.25
|x| が大きいと σ'(x) ≈ 0  ← 勾配消失
```

#### 問題点

1. **勾配消失**: `|x|` が大きいと勾配がほぼゼロ
2. **出力が正のみ**: 平均が0.5（ゼロ中心でない）
3. **計算コスト**: 指数関数の計算が必要

#### PyTorchでの実装

```python
import torch.nn.functional as F

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
y = torch.sigmoid(x)
# tensor([0.1192, 0.2689, 0.5000, 0.7311, 0.8808])
```

### Tanh

**Tanh（双曲線正接）** は、出力を `(-1, 1)` の範囲に圧縮します。

#### 数式

```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
        = 2σ(2x) - 1  （Sigmoidとの関係）
```

#### グラフ

```
 1.0 ┤         ╭─────
     │       ╭─╯
 0.0 ┤    ╭──╯
     │  ╭─╯
-1.0 ┤──╯
     └───────────────→ x
    -5   0   5
```

#### 特性

```
範囲: (-1, 1)
tanh(0) = 0  ← ゼロ中心（Sigmoidより良い）
tanh(+∞) = 1
tanh(-∞) = -1

奇関数:
  tanh(-x) = -tanh(x)
```

#### 微分

```
tanh'(x) = 1 - tanh²(x)

x=0 で最大: tanh'(0) = 1
|x| が大きいと tanh'(x) ≈ 0  ← 勾配消失
```

#### Sigmoidとの比較

**利点**:
- ゼロ中心（平均が0に近い）

**欠点**:
- 依然として勾配消失問題がある

---

## ReLUファミリー

### ReLU

**ReLU（Rectified Linear Unit）** は、最もシンプルで広く使われる活性化関数です。

#### 数式

```
ReLU(x) = max(0, x) = {
  x  if x > 0
  0  if x ≤ 0
}
```

#### グラフ

```
     │
     │    ╱
     │  ╱
     │╱
─────┼─────→ x
     0
```

#### 微分

```
ReLU'(x) = {
  1  if x > 0
  0  if x < 0
  未定義  if x = 0（実装では通常0または1）
}
```

#### 利点

1. **勾配消失の軽減**: `x > 0` で勾配が常に1
2. **計算効率**: 単純な比較とマスク
3. **疎な活性化**: 約半分のニューロンがゼロ（効率的）

#### 問題点: Dying ReLU

```
問題:
  x < 0 の領域で勾配が常にゼロ
  → 一度負の領域に入ると、更新されなくなる

例:
  初期化で大きな負の重みを持つと、
  活性化が常にゼロ → 勾配がゼロ → 学習が停止
```

#### PyTorchでの実装

```python
import torch.nn.functional as F

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
y = F.relu(x)
# tensor([0.0, 0.0, 0.0, 1.0, 2.0])
```

### Leaky ReLU

**Leaky ReLU** は、負の領域に小さな勾配を持たせてDying ReLUを緩和します。

#### 数式

```
LeakyReLU(x) = max(αx, x) = {
  x   if x > 0
  αx  if x ≤ 0
}

通常 α = 0.01
```

#### グラフ

```
     │
     │    ╱
     │  ╱
    ╱│╱
  ╱  ┼─────→ x
     0
    ↑
   傾き α
```

#### 微分

```
LeakyReLU'(x) = {
  1  if x > 0
  α  if x < 0
}

負の領域でも勾配がα（小さいが非ゼロ）
```

#### PyTorchでの実装

```python
import torch.nn as nn

leaky_relu = nn.LeakyReLU(negative_slope=0.01)
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
y = leaky_relu(x)
# tensor([-0.02, -0.01, 0.0, 1.0, 2.0])
```

### ELU

**ELU（Exponential Linear Unit）** は、負の領域を指数関数で滑らかにします。

#### 数式

```
ELU(x) = {
  x                if x > 0
  α(e^x - 1)       if x ≤ 0
}

通常 α = 1.0
```

#### 特徴

```
x → -∞ のとき ELU(x) → -α
x = 0 で連続かつ微分可能
平均出力がゼロに近い
```

#### 微分

```
ELU'(x) = {
  1           if x > 0
  α e^x       if x ≤ 0
}

x = 0 で連続（ELU'(0) = 1）
```

#### 利点と欠点

**利点**:
- 負の値を持つため、平均がゼロに近い
- 滑らかな曲線（勾配が連続）

**欠点**:
- 指数関数の計算コスト

### GELU

**GELU（Gaussian Error Linear Unit）** は、Transformerで人気の活性化関数です。

#### 数式

```
GELU(x) = x Φ(x)

ここで:
  Φ(x) = P(X ≤ x) where X ~ N(0, 1)
       = 標準正規分布の累積分布関数（CDF）
```

近似版：
```
GELU(x) ≈ 0.5 x (1 + tanh(√(2/π) (x + 0.044715 x³)))

または:
GELU(x) ≈ x σ(1.702 x)  （Sigmoidによる近似）
```

#### 直感的理解

```
GELU(x) = x * P(X ≤ x)

x が大きい → Φ(x) ≈ 1 → GELU(x) ≈ x
x が小さい → Φ(x) ≈ 0 → GELU(x) ≈ 0

ReLUの「硬い」閾値を、確率的に「柔らかく」したもの
```

#### PyTorchでの実装

```python
import torch.nn.functional as F

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
y = F.gelu(x)
# tensor([-0.0454, -0.1588, 0.0000, 0.8412, 1.9546])

# 近似版
y_approx = F.gelu(x, approximate='tanh')
```

---

## 現代的な活性化関数

### Swish/SiLU

**Swish（またはSiLU: Sigmoid Linear Unit）** は、滑らかで自己ゲート型の活性化関数です。

#### 数式

```
Swish(x) = x σ(βx) = x / (1 + e^(-βx))

通常 β = 1（この場合 SiLU と同じ）
```

#### グラフ

```
 2.0 ┤           ╱
     │         ╱
 1.0 ┤       ╱
     │     ╱
 0.0 ┤   ╱╯
     │ ╱╯
-0.5 ┤╯
     └───────────────→ x
    -5   0   5
```

#### 微分

```
Swish'(x) = Swish(x) + σ(x) (1 - Swish(x))

（連鎖律により導出）
```

#### 特徴

1. **自己ゲート**: 入力 `x` 自身でゲートを制御
2. **滑らか**: 無限回微分可能
3. **有界でない**: 上限がない（ReLUと同様）
4. **非単調**: `x < 0` で僅かに負の値

#### 利点

```
実験的に、深いネットワークでReLUより良い性能を示すことが多い
```

#### PyTorchでの実装

```python
import torch.nn.functional as F

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
y = F.silu(x)  # Swish (β=1)
# tensor([-0.2384, -0.2689, 0.0000, 0.7311, 1.7616])
```

### Squared ReLU

**Squared ReLU（ReLU²）** は、ReLUの出力を二乗します（**nanochatで使用**）。

#### 数式

```
ReLU²(x) = (max(0, x))² = {
  x²  if x > 0
  0   if x ≤ 0
}
```

#### グラフ

```
     │
 4.0 ┤        ╱
     │      ╱
 1.0 ┤    ╱
     │  ╱
 0.0 ┤─╯
     └───────────────→ x
    -2  0  2
```

#### 微分

```
(ReLU²)'(x) = {
  2x  if x > 0
  0   if x ≤ 0
}

= 2 * ReLU(x)
```

#### 特性

1. **非線形性の強化**: 二乗により非線形性が増加
2. **スケーリング**: 大きな値がさらに大きく、小さな値はさらに小さく
3. **疎性**: ReLUと同様に負の領域でゼロ

#### なぜSquared ReLUか？

いくつかの理論的・実験的な利点があります：

```
1. より強い非線形性:
   ReLU: y = x
   ReLU²: y = x²
   → より複雑な関数を少ない層で表現可能

2. 勾配の適応性:
   微分が 2x → 入力に比例した勾配
   大きな入力は大きな勾配、小さな入力は小さな勾配

3. 実験的な成功:
   一部のTransformerアーキテクチャで良好な性能
```

#### 懸念点

```
勾配爆発のリスク:
  大きな x で勾配が 2x → 爆発的に増加する可能性

対策:
  - 適切な初期化
  - 勾配クリッピング
  - 正規化（LayerNorm/RMSNorm）
```

#### PyTorchでの実装

```python
import torch.nn.functional as F

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
y = F.relu(x).square()
# tensor([0.0, 0.0, 0.0, 1.0, 4.0])

# または
y = torch.square(F.relu(x))
```

---

## 比較とまとめ

### 活性化関数の系譜

```
古典的:
  ├─ Sigmoid (1980s)
  └─ Tanh (1990s)

ReLU革命 (2010s):
  ├─ ReLU
  ├─ Leaky ReLU
  ├─ ELU
  └─ PReLU

現代的 (2015-):
  ├─ GELU
  ├─ Swish/SiLU
  └─ Squared ReLU
```

### 特性比較

| 関数 | 範囲 | 連続微分 | ゼロ中心 | 計算コスト | 勾配消失 |
|------|------|---------|---------|-----------|---------|
| Sigmoid | (0,1) | ✓ | ✗ | 高 | ✓ 問題あり |
| Tanh | (-1,1) | ✓ | ✓ | 高 | ✓ 問題あり |
| ReLU | [0,∞) | ✗ | ✗ | 低 | ✗ 軽減 |
| Leaky ReLU | (-∞,∞) | ✗ | ✗ | 低 | ✗ 軽減 |
| ELU | (-α,∞) | ✓ | ✗ | 中 | ✗ 軽減 |
| GELU | (-∞,∞) | ✓ | ✗ | 中 | ✗ 軽減 |
| Swish/SiLU | (-∞,∞) | ✓ | ✗ | 中 | ✗ 軽減 |
| ReLU² | [0,∞) | ✗ | ✗ | 低 | ✗ 軽減 |

### 用途別の推奨

```
CNN（画像）:
  ReLU ← シンプルで効果的、デファクトスタンダード

Transformer（NLP）:
  GELU ← BERT、GPT-2/3で使用
  Swish/SiLU ← 一部の現代的なモデル
  ReLU² ← nanochatなど

RNN:
  Tanh ← 内部ゲート（LSTM, GRU）
  ReLU ← 出力層の代替として

出力層:
  Sigmoid ← 二値分類（確率出力）
  Softmax ← 多クラス分類
  線形（なし）← 回帰
```

### 選択のガイドライン

```
デフォルトの選択:
  ReLU または GELU
  理由: 広く使われ、安定、実績がある

実験的に試す:
  Swish/SiLU, ReLU²
  理由: 特定のタスクでより良い性能の可能性

避けるべき:
  Sigmoid/Tanh（隠れ層で）
  理由: 勾配消失問題
```

---

## nanochatでの実装

### Squared ReLUの使用（gpt.py:135-139）

```python
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()  # ← Squared ReLU
        x = self.c_proj(x)
        return x
```

### なぜSquared ReLUを選択したか？

nanochatのコードベースから推測される理由：

1. **シンプルさ**: 実装が簡単（ReLU + square）
2. **計算効率**: 指数関数不要（GELUやSwishより高速）
3. **非線形性**: 強い非線形性で表現力向上
4. **実験的な成功**: 最近の研究で良好な結果

### 実行例

```python
import torch
import torch.nn.functional as F

# サンプル入力
x = torch.randn(2, 512)  # (バッチ, 埋め込み次元)

# MLP
mlp = MLP(config)
out = mlp(x)

# 中間の活性化を見る
x_fc = mlp.c_fc(x)  # (2, 2048) 4倍に拡大
print(f"After linear: min={x_fc.min():.2f}, max={x_fc.max():.2f}")

x_act = F.relu(x_fc).square()  # Squared ReLU
print(f"After activation: min={x_act.min():.2f}, max={x_act.max():.2f}")

# 疎性を確認
sparsity = (x_act == 0).float().mean()
print(f"Sparsity: {sparsity:.2%}")
```

出力例：
```
After linear: min=-2.45, max=3.12
After activation: min=0.00, max=9.73
Sparsity: 48.23%
```

約半分がゼロ（ReLUの特性を継承）し、正の値は二乗されて拡大されます。

---

## 練習問題

### 問題1: Sigmoidの計算

`x = [-2, 0, 2]` に対してSigmoidを計算してください（`e ≈ 2.718`）。

<details>
<summary>解答</summary>

```
σ(x) = 1 / (1 + e^(-x))

x = -2:
  σ(-2) = 1 / (1 + e^2) = 1 / (1 + 7.389) = 1 / 8.389 ≈ 0.119

x = 0:
  σ(0) = 1 / (1 + e^0) = 1 / 2 = 0.5

x = 2:
  σ(2) = 1 / (1 + e^(-2)) = 1 / (1 + 0.135) = 1 / 1.135 ≈ 0.881
```
</details>

### 問題2: ReLUとLeaky ReLUの比較

`x = [-2.0, -1.0, 0.0, 1.0, 2.0]` に対して、ReLUとLeaky ReLU（α=0.01）を適用し、違いを示してください。

<details>
<summary>解答</summary>

```
ReLU(x) = max(0, x):
  [-2.0, -1.0, 0.0, 1.0, 2.0]
  → [0.0, 0.0, 0.0, 1.0, 2.0]

Leaky ReLU(x) = max(0.01x, x):
  [-2.0, -1.0, 0.0, 1.0, 2.0]
  → [-0.02, -0.01, 0.0, 1.0, 2.0]

違い:
  負の領域で、Leaky ReLUは小さな値を保持
  → Dying ReLU問題を軽減
```
</details>

### 問題3: Squared ReLUの微分

Squared ReLU `f(x) = (max(0, x))²` の微分を求めてください。

<details>
<summary>解答</summary>

```
f(x) = (max(0, x))² = (ReLU(x))²

連鎖律:
  f'(x) = 2 * ReLU(x) * ReLU'(x)

ReLU'(x) = {
  1 if x > 0
  0 if x ≤ 0
}

したがって:
  f'(x) = {
    2 * x * 1 = 2x  if x > 0
    2 * 0 * 0 = 0   if x ≤ 0
  }

つまり:
  (ReLU²)'(x) = 2 * ReLU(x)
```
</details>

### 問題4: 活性化関数の選択

以下の状況で、どの活性化関数を選ぶべきか理由とともに答えてください。

1. 深い畳み込みニューラルネットワーク（100層）
2. Transformerモデルの隠れ層
3. 二値分類の出力層

<details>
<summary>解答</summary>

```
1. 深いCNN（100層）:
   選択: ReLU（またはLeaky ReLU、ELU）
   理由:
     - 勾配消失を軽減（Sigmoid/Tanhは不適）
     - 計算効率的
     - 深いネットワークで実績あり（ResNetなど）

2. Transformerの隠れ層:
   選択: GELU または Squared ReLU
   理由:
     - GELU: BERT、GPTで標準的
     - Squared ReLU: 計算効率的で非線形性が強い（nanochat）

3. 二値分類の出力層:
   選択: Sigmoid
   理由:
     - 出力を (0, 1) の確率に変換
     - 損失関数（Binary Cross Entropy）と相性が良い
```
</details>

---

## まとめ

### 活性化関数の核心

```
目的: 非線形性の導入

線形のみ:
  どれだけ深くしても単一の線形変換と等価

非線形あり:
  複雑な関数を表現可能
```

### 歴史的な変遷

```
1980s-1990s: Sigmoid/Tanh
  問題: 勾配消失

2010s: ReLU革命
  解決: 勾配消失の軽減
  問題: Dying ReLU

2015-現在: GELU/Swish/ReLU²
  改善: より滑らかな勾配、より強い非線形性
```

### nanochatの選択

```python
活性化関数: Squared ReLU
  f(x) = (max(0, x))²

理由:
  - 計算効率的
  - 強い非線形性
  - ReLUの疎性を維持
```

### 実用的なアドバイス

```
デフォルト:
  ReLU（シンプルで効果的）

Transformer:
  GELU（BERT、GPTの標準）

実験:
  Swish/SiLU、Squared ReLU

避ける（隠れ層で）:
  Sigmoid/Tanh（勾配消失）
```

### 次のステップ

- [数学06: 誤差逆伝播法](06-backpropagation.md) - 活性化関数の微分
- [第4章: モデルの詳細実装](../04-model-implementation.md) - MLPでの使用
- [数学12: 確率的サンプリング](12-sampling.md) - 出力層での確率変換
