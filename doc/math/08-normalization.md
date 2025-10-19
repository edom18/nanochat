# 第8章: 正規化手法

## 目次
- [正規化とは何か](#正規化とは何か)
- [なぜ正規化が必要か](#なぜ正規化が必要か)
- [Batch Normalization](#batch-normalization)
- [Layer Normalization](#layer-normalization)
- [RMS Normalization](#rms-normalization)
- [各手法の比較](#各手法の比較)
- [Pre-Norm vs Post-Norm](#pre-norm-vs-post-norm)
- [nanochatでの実装](#nanochatでの実装)
- [練習問題](#練習問題)

---

## 正規化とは何か

**正規化（Normalization）** は、ニューラルネットワークの中間層の出力を特定の統計的性質（平均や分散）に調整する手法です。

### 基本的な考え方

データを**標準的なスケール**に揃えることで、学習を安定化・高速化します。

```
正規化の一般的な形式:

y = (x - μ) / σ

ここで:
  x: 入力
  μ: 平均
  σ: 標準偏差
  y: 正規化された出力（平均0、標準偏差1）
```

### 簡単な例

```python
import torch

x = torch.tensor([10.0, 20.0, 30.0, 40.0])

# 平均と標準偏差
mean = x.mean()  # 25.0
std = x.std()    # 11.18

# 正規化
y = (x - mean) / std
# tensor([-1.3416, -0.4472,  0.4472,  1.3416])

print(y.mean())  # 0.0（ほぼ）
print(y.std())   # 1.0
```

正規化後、データは**平均0、標準偏差1の分布**になります。

---

## なぜ正規化が必要か

### 問題1: 内部共変量シフト（Internal Covariate Shift）

ニューラルネットワークの学習中、各層の入力分布が変化し続けます。

```
訓練ステップ1:
  Layer 1 出力: [0.1, 0.2, 0.3, ...]  平均≈0.2

  ↓ パラメータ更新

訓練ステップ2:
  Layer 1 出力: [5.2, 6.1, 4.8, ...]  平均≈5.4
                 ↑
                 大きく変化！

Layer 2は、入力分布が安定しないため学習が困難
```

**正規化により、各層の入力分布を安定化**できます。

### 問題2: 勾配の消失・爆発

活性化値が極端になると、勾配が消失または爆発します。

```
活性化値が大きすぎる場合:
  x = [100, 200, 300]
  Sigmoid(x) ≈ [1.0, 1.0, 1.0]  ← 飽和
  勾配 ≈ 0  ← 勾配消失

活性化値が小さすぎる場合:
  x = [-100, -200, -300]
  Sigmoid(x) ≈ [0.0, 0.0, 0.0]  ← 飽和
  勾配 ≈ 0  ← 勾配消失
```

**正規化により、活性化値を適切な範囲に保つ**ことができます。

### 問題3: 学習率の調整が難しい

層ごとに最適な学習率が異なる場合、全体的な学習率の設定が困難です。

```
最適な学習率:
  Layer 1: 0.001
  Layer 2: 0.1    ← 100倍も違う
  Layer 3: 0.01

正規化により、各層のスケールが揃うため、
単一の学習率でも効率的に学習可能
```

---

## Batch Normalization

**Batch Normalization (BatchNorm)** は、**ミニバッチ内のサンプル間**で正規化を行います。

### 数式

```
μ_B = (1/m) Σ_i x_i          (バッチ平均)
σ²_B = (1/m) Σ_i (x_i - μ_B)²  (バッチ分散)

x_norm = (x - μ_B) / √(σ²_B + ε)

y = γ x_norm + β

ここで:
  m: バッチサイズ
  ε: 数値安定性のための小さな定数（例: 1e-5）
  γ, β: 学習可能なパラメータ（スケールとシフト）
```

### 可視化

```
バッチ（4サンプル、3特徴量）:

サンプル0: [1.0, 2.0, 3.0]
サンプル1: [4.0, 5.0, 6.0]
サンプル2: [7.0, 8.0, 9.0]
サンプル3: [10., 11., 12.]

特徴量ごとに正規化:

特徴量0: [1.0, 4.0, 7.0, 10.0]
         平均=5.5, 標準偏差=3.35
         正規化→ [-1.34, -0.45, 0.45, 1.34]

特徴量1: [2.0, 5.0, 8.0, 11.0]
         平均=6.5, 標準偏差=3.35
         正規化→ [-1.34, -0.45, 0.45, 1.34]

特徴量2: [3.0, 6.0, 9.0, 12.0]
         平均=7.5, 標準偏差=3.35
         正規化→ [-1.34, -0.45, 0.45, 1.34]

バッチ次元で統計量を計算
```

### PyTorchでの実装

```python
import torch.nn as nn

# CNNでの使用例
bn = nn.BatchNorm2d(num_features=64)

# 入力: (バッチ, チャネル, 高さ, 幅)
x = torch.randn(32, 64, 28, 28)
y = bn(x)

# 各チャネルごとに、バッチ×高さ×幅で統計量を計算
```

### 問題点

BatchNormは**バッチサイズに依存**します：

```
問題1: 小さいバッチサイズ
  バッチサイズ=2の場合、統計量が不安定
  μ, σの推定精度が低い

問題2: 可変長シーケンス
  自然言語処理では、シーケンス長がサンプルごとに異なる
  バッチ統計が計算しにくい

問題3: 推論時の不一致
  訓練時: バッチ統計を使用
  推論時: 移動平均した統計を使用
  → 訓練と推論の挙動が異なる
```

これらの問題から、**Transformerでは使用されない**ことが多いです。

---

## Layer Normalization

**Layer Normalization (LayerNorm)** は、**各サンプルの特徴量次元内**で正規化を行います。

### 数式

```
μ = (1/D) Σ_i x_i          (特徴量の平均)
σ² = (1/D) Σ_i (x_i - μ)²  (特徴量の分散)

x_norm = (x - μ) / √(σ² + ε)

y = γ x_norm + β

ここで:
  D: 特徴量の次元数
  γ, β: 学習可能なパラメータ（各次元ごと）
```

### BatchNormとの違い

```
BatchNorm: バッチ次元で統計量を計算
  形状: (B, D)
  平均: 各列（特徴量）ごとに計算（Bサンプルの平均）

LayerNorm: 特徴量次元で統計量を計算
  形状: (B, D)
  平均: 各行（サンプル）ごとに計算（D特徴量の平均）
```

可視化：
```
入力 (3サンプル × 4特徴量):

サンプル0: [1.0, 2.0, 3.0, 4.0]  → 平均=2.5
サンプル1: [5.0, 6.0, 7.0, 8.0]  → 平均=6.5
サンプル2: [9.0, 10., 11., 12.]  → 平均=10.5

各サンプル内で正規化（行方向）:

サンプル0: [-1.34, -0.45, 0.45, 1.34]
サンプル1: [-1.34, -0.45, 0.45, 1.34]
サンプル2: [-1.34, -0.45, 0.45, 1.34]

バッチサイズに依存しない！
```

### PyTorchでの実装

```python
import torch.nn as nn

# LayerNorm
ln = nn.LayerNorm(normalized_shape=512)

# 入力: (バッチ, シーケンス長, 特徴量)
x = torch.randn(32, 128, 512)
y = ln(x)

# 各サンプル・各位置の512次元で正規化
# y[i, j] は x[i, j, :] の512次元で平均0、標準偏差1
```

### 利点

1. **バッチサイズに依存しない**: バッチサイズ=1でも動作
2. **訓練と推論で一貫**: 同じ計算方法
3. **可変長シーケンスに対応**: NLPに最適

### Transformerでの使用

```
Transformer Block:

  入力 x
    ↓
  LayerNorm
    ↓
  Attention
    ↓
  残差接続 (x +)
    ↓
  LayerNorm
    ↓
  MLP
    ↓
  残差接続 (x +)
    ↓
  出力
```

---

## RMS Normalization

**RMS Normalization (RMSNorm)** は、LayerNormの**簡略版**です。**平均を引かず、RMS（二乗平均平方根）のみで正規化**します。

### 数式

```
RMS(x) = √((1/D) Σ_i x_i²)

x_norm = x / RMS(x)

y = γ x_norm

ここで:
  D: 特徴量の次元数
  γ: 学習可能なスケールパラメータ（バイアスβは無し）
```

### LayerNormとの違い

```
LayerNorm:
  1. 平均を計算: μ = (1/D) Σ x_i
  2. 平均を引く: x' = x - μ
  3. 標準偏差で割る: x_norm = x' / σ
  4. スケール・シフト: y = γ x_norm + β

RMSNorm:
  1. RMSを計算: rms = √((1/D) Σ x_i²)
  2. RMSで割る: x_norm = x / rms
  3. スケールのみ: y = γ x_norm
```

### なぜ簡略化できるのか？

**Re-centering（平均を引く）は必須ではない**ことが実験的に示されています：

1. **残差接続との組み合わせ**: Transformerは残差接続を使うため、バイアスが自然に調整される
2. **計算効率**: 平均計算とシフト操作を省略できる
3. **実験的な性能**: LayerNormと同等の性能を維持

### 計算量の比較

```
LayerNorm:
  1. 平均計算: D回の加算 + 1回の除算
  2. 中心化: D回の減算
  3. 分散計算: D回の二乗 + D回の加算 + 1回の除算
  4. 正規化: D回の除算
  5. アフィン変換: D回の乗算 + D回の加算
  合計: 約 5D 回の演算

RMSNorm:
  1. 二乗和: D回の二乗 + D回の加算
  2. RMS: 1回の平方根
  3. 正規化: D回の除算
  4. スケール: D回の乗算
  合計: 約 3D 回の演算

約40%の演算削減！
```

### 具体例

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0, 4.0])

# LayerNorm
mean = x.mean()  # 2.5
std = x.std(unbiased=False)  # 1.118
x_ln = (x - mean) / std
# tensor([-1.3416, -0.4472,  0.4472,  1.3416])

# RMSNorm
rms = torch.sqrt((x ** 2).mean())  # 2.7386
x_rms = x / rms
# tensor([0.3651, 0.7303, 1.0954, 1.4606])

# RMSNormは平均を引かないため、出力の平均は0にならない
print(x_ln.mean())   # 0.0
print(x_rms.mean())  # 0.9129（非ゼロ）
```

### PyTorchでの実装

```python
# PyTorch 2.4+ の組み込み関数
import torch.nn.functional as F

x = torch.randn(32, 128, 512)
y = F.rms_norm(x, normalized_shape=(512,))

# 手動実装
def rms_norm(x, eps=1e-6):
    """
    x: (*, D) 任意の形状
    """
    # 最後の次元でRMSを計算
    rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + eps)
    return x / rms

# 学習可能なスケールパラメータ付き
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return self.weight * x_norm
```

---

## 各手法の比較

### 正規化軸の違い

```
入力形状: (B=2, T=3, D=4)  (バッチ, シーケンス, 特徴量)

データ:
  [[[1, 2, 3, 4],      サンプル0, 位置0
    [5, 6, 7, 8],      サンプル0, 位置1
    [9, 10, 11, 12]],  サンプル0, 位置2
   [[13, 14, 15, 16],  サンプル1, 位置0
    [17, 18, 19, 20],  サンプル1, 位置1
    [21, 22, 23, 24]]] サンプル1, 位置2

BatchNorm (特徴量ごと):
  特徴量0: [1, 5, 9, 13, 17, 21] → 平均=11
  特徴量1: [2, 6, 10, 14, 18, 22] → 平均=12
  ...
  (B×T) 個の値で統計量を計算

LayerNorm / RMSNorm (サンプル・位置ごと):
  サンプル0, 位置0: [1, 2, 3, 4] → 平均=2.5
  サンプル0, 位置1: [5, 6, 7, 8] → 平均=6.5
  ...
  D 個の値で統計量を計算
```

### 特性の比較

| 手法 | 正規化軸 | バッチ依存 | 訓練/推論一貫性 | 計算量 | Transformer適用 |
|------|---------|----------|---------------|--------|----------------|
| BatchNorm | バッチ×位置 | 依存 | 不一致 | 中 | ✗ 不適 |
| LayerNorm | 特徴量 | 独立 | 一貫 | 大 | ✓ 適用可能 |
| RMSNorm | 特徴量 | 独立 | 一貫 | 小 | ✓ 推奨 |

### 選択のガイドライン

```
CNN（画像）:
  BatchNorm ← バッチサイズが大きい場合
  GroupNorm ← バッチサイズが小さい場合

Transformer（NLP）:
  LayerNorm ← 標準的
  RMSNorm ← 高速化が必要な場合（nanochatなど）

RNN:
  LayerNorm ← シーケンス処理に適合
```

---

## Pre-Norm vs Post-Norm

Transformerでは、正規化を適用する**位置**が重要です。

### Post-Norm（初期のTransformer）

```
x → Attention → 残差 → LayerNorm → x'
                  ↑        ↓
                  └────────┘

x' → MLP → 残差 → LayerNorm → 出力
            ↑        ↓
            └────────┘
```

数式：
```
x' = LayerNorm(x + Attention(x))
x'' = LayerNorm(x' + MLP(x'))
```

**問題点**: 深いネットワークで勾配が不安定になりやすい

### Pre-Norm（現代的なTransformer）

```
x → LayerNorm → Attention → 残差 → x'
                              ↑      ↓
                              └──────┘

x' → LayerNorm → MLP → 残差 → 出力
                         ↑      ↓
                         └──────┘
```

数式：
```
x' = x + Attention(LayerNorm(x))
x'' = x' + MLP(LayerNorm(x'))
```

**利点**:
- **勾配の流れが良好**: 残差接続が正規化前にあるため
- **深いネットワークで安定**: 100層以上でも学習可能
- **学習率の許容範囲が広い**

### 可視化

```
Post-Norm:
  入力 → [サブレイヤー] → 加算 → [正規化] → 出力
          ↑___________________________|
          勾配がサブレイヤーを通る必要がある

Pre-Norm:
  入力 → [正規化] → [サブレイヤー] → 加算 → 出力
          ↑__________________________________|
          勾配が直接入力に流れる（残差パス）
```

**nanochatはPre-Normを採用**しています（gpt.py:149-150）。

---

## nanochatでの実装

### 1. RMSNorm関数（gpt.py:36-38）

```python
def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))
```

**特徴**:
- **学習可能なパラメータなし**（γも省略）
- PyTorchの組み込み関数 `F.rms_norm` を使用
- 最後の次元で正規化

### 2. Blockでの使用（gpt.py:148-151）

```python
class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache):
        # Pre-Norm: AttentionとMLPの前に正規化
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        x = x + self.mlp(norm(x))
        return x
```

**パターン**:
```
x → norm → Attention → 残差 (+x) → x'
x' → norm → MLP → 残差 (+x') → 出力
```

これは**Pre-Norm**のパターンです。

### 3. Attention内での正規化（gpt.py:90）

```python
def forward(self, x, cos_sin, kv_cache):
    # ... QKV生成 ...

    # Rotary Embeddings適用
    q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)

    # QK正規化
    q, k = norm(q), norm(k)  # ← ここ！

    # Attention計算
    # ...
```

**QK正規化**:
- Query と Key を個別に正規化
- スケーリング係数 `√d_k` の代わりになる効果
- Attention計算の安定性向上

### 4. 最終層の正規化（gpt.py:186-190）

```python
def forward(self, idx, targets=None, kv_cache=None):
    # ... Embedding + Blocks ...

    for block in self.transformer.h:
        x = block(x, cos_sin, kv_cache)

    # 最終層の正規化
    x = norm(x)

    # Logits計算
    logits = self.lm_head(x)
    # ...
```

**最終正規化**:
- 全てのBlockを通過後、最後に正規化
- 出力の安定性を確保

### nanochatの正規化戦略まとめ

```
埋め込み
  ↓
┌─────────────────────────┐
│ Block 1                 │
│  x → norm → Attention → (+x) │
│  x → norm → MLP → (+x)       │
└─────────────────────────┘
  ↓
┌─────────────────────────┐
│ Block 2                 │
│  x → norm → Attention → (+x) │
│    (内部: norm(q), norm(k))  │
│  x → norm → MLP → (+x)       │
└─────────────────────────┘
  ↓
  ... (繰り返し)
  ↓
norm (最終正規化)
  ↓
LM Head (Logits)
```

**特徴**:
1. **Pre-Norm**: 各サブレイヤー前に正規化
2. **RMSNorm**: 計算効率重視（学習可能パラメータなし）
3. **QK正規化**: Attention安定性向上
4. **最終正規化**: 出力安定性確保

---

## 練習問題

### 問題1: RMSNormの計算

以下のベクトルに対してRMSNormを計算してください。

```
x = [3.0, 4.0, 0.0, 0.0]
```

<details>
<summary>解答</summary>

```
1. 二乗和の平均:
   mean(x²) = (3² + 4² + 0² + 0²) / 4
            = (9 + 16 + 0 + 0) / 4
            = 25 / 4
            = 6.25

2. RMS:
   RMS = √6.25 = 2.5

3. 正規化:
   x_norm = x / RMS
          = [3.0, 4.0, 0.0, 0.0] / 2.5
          = [1.2, 1.6, 0.0, 0.0]
```
</details>

### 問題2: LayerNorm vs RMSNormの出力平均

LayerNormとRMSNormで正規化した後、出力の平均はどうなるか？

```python
x = torch.tensor([1.0, 2.0, 3.0, 4.0])

# LayerNorm (γ=1, β=0と仮定)
x_ln = (x - x.mean()) / x.std(unbiased=False)

# RMSNorm
x_rms = x / torch.sqrt((x ** 2).mean())

print(x_ln.mean())
print(x_rms.mean())
```

<details>
<summary>解答</summary>

```
LayerNorm:
  平均を引くため、出力の平均は常に 0

RMSNorm:
  平均を引かないため、出力の平均は一般に非ゼロ

実際の計算:
  x = [1.0, 2.0, 3.0, 4.0]

  LayerNorm:
    mean = 2.5, std = 1.118
    x_ln = [-1.34, -0.45, 0.45, 1.34]
    x_ln.mean() = 0.0

  RMSNorm:
    rms = 2.738
    x_rms = [0.365, 0.730, 1.095, 1.461]
    x_rms.mean() = 0.913（非ゼロ）
```
</details>

### 問題3: 正規化軸の理解

形状 `(2, 3, 4)` のテンソル（バッチ=2, シーケンス=3, 特徴量=4）について：
1. BatchNormは何個の要素で統計量を計算するか？
2. LayerNormは何個の要素で統計量を計算するか？
3. それぞれ何回正規化が行われるか？

<details>
<summary>解答</summary>

```
テンソル形状: (B=2, T=3, D=4)
総要素数: 2 × 3 × 4 = 24

BatchNorm（特徴量ごとに正規化）:
  1. 統計量を計算する要素数: B × T = 2 × 3 = 6個
  2. 正規化の回数: D = 4回（各特徴量ごと）

LayerNorm（サンプル・位置ごとに正規化）:
  1. 統計量を計算する要素数: D = 4個
  2. 正規化の回数: B × T = 2 × 3 = 6回（各サンプル・各位置ごと）

可視化:
  BatchNorm: 縦方向に集約（特徴量軸を保持）
    |||||||... (6要素) → 統計量
    ↓
    各特徴量ごとに正規化（4回）

  LayerNorm: 横方向に集約（バッチ・時間軸を保持）
    ──── (4要素) → 統計量
    ↓
    各サンプル・各位置ごとに正規化（6回）
```
</details>

### 問題4: Pre-Norm vs Post-Normの勾配フロー

以下の疑似コードで、勾配が入力 `x` に直接流れるのはどちらか？

```python
# Post-Norm
def post_norm_block(x):
    return layer_norm(x + sublayer(x))

# Pre-Norm
def pre_norm_block(x):
    return x + sublayer(layer_norm(x))
```

<details>
<summary>解答</summary>

```
Pre-Normの方が勾配が直接流れます。

逆伝播時の勾配フロー:

Post-Norm:
  ∂L/∂x = ∂L/∂output × ∂layer_norm/∂(x + sublayer(x)) × ∂(x + sublayer(x))/∂x
        = grad_output × (1/σ) × (1 + ∂sublayer/∂x)
          ↑
          layer_normを通る必要がある

Pre-Norm:
  ∂L/∂x = ∂L/∂output × ∂(x + sublayer(...))/∂x
        = grad_output × 1
          ↑
          残差接続により直接流れる（+ ∂sublayer/∂xの項もある）

Pre-Normでは、残差接続が正規化の外にあるため、
勾配が直接入力に到達する経路がある（恒等写像）。
これにより深いネットワークでも勾配が消失しにくい。
```
</details>

---

## まとめ

### 正規化手法の核心

1. **目的**: 層の出力を標準的なスケールに揃える
2. **効果**: 学習の安定化・高速化、勾配の流れ改善
3. **種類**: BatchNorm（バッチ軸）、LayerNorm（特徴量軸）、RMSNorm（簡略版）

### Transformerでの正規化

```
選択: LayerNorm または RMSNorm
理由:
  - バッチサイズに依存しない
  - 訓練と推論で一貫
  - 可変長シーケンスに対応

配置: Pre-Norm
理由:
  - 勾配の流れが良好
  - 深いネットワークで安定
```

### nanochatの戦略

```python
# RMSNorm（学習可能パラメータなし）
def norm(x):
    return F.rms_norm(x, (x.size(-1),))

# Pre-Norm配置
x = x + self.attn(norm(x), ...)
x = x + self.mlp(norm(x))

# QK正規化
q, k = norm(q), norm(k)

# 最終正規化
x = norm(x)
```

**特徴**: 計算効率とシンプルさを重視

### 次のステップ

- [数学09: 位置エンコーディング](09-positional-encoding.md) - Rotary Embeddings
- [数学07: Attention機構の数式](07-attention.md) - QK正規化の文脈
- [第4章: モデルの詳細実装](../04-model-implementation.md) - Blockの全体像
