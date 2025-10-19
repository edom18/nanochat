# 第9章: 位置エンコーディング

## 目次
- [なぜ位置情報が必要か](#なぜ位置情報が必要か)
- [絶対位置エンコーディング](#絶対位置エンコーディング)
  - [Sinusoidal位置エンコーディング](#sinusoidal位置エンコーディング)
  - [学習可能な位置エンコーディング](#学習可能な位置エンコーディング)
- [相対位置エンコーディング](#相対位置エンコーディング)
- [Rotary Positional Embeddings (RoPE)](#rotary-positional-embeddings-rope)
  - [基本的なアイデア](#基本的なアイデア)
  - [数学的定義](#数学的定義)
  - [2次元回転の直感](#2次元回転の直感)
  - [実装の詳細](#実装の詳細)
- [nanochatでの実装](#nanochatでの実装)
- [練習問題](#練習問題)

---

## なぜ位置情報が必要か

**Transformerの Attention機構は、位置に対して順序不変（permutation invariant）** です。

### 順序不変とは？

シーケンスの順序を入れ替えても、Attention計算の結果は変わりません（位置情報がない場合）。

```
例: "cat ate fish" vs "fish ate cat"

位置情報なしの場合:
  単語埋め込み:
    cat  → [0.1, 0.2, 0.3, ...]
    ate  → [0.4, 0.5, 0.6, ...]
    fish → [0.7, 0.8, 0.9, ...]

  Attention計算:
    各単語の埋め込みベクトルのみを使用
    → 順序情報が失われる

  "cat ate fish" と "fish ate cat" が同じ表現に！
```

### 問題の深刻さ

```
文1: "The cat chased the mouse."
文2: "The mouse chased the cat."

意味は正反対だが、位置情報がないと区別できない
```

### 解決策

**各トークンに位置情報を追加**することで、モデルが順序を認識できるようにします。

```
位置情報付き:
  cat (位置0) → [0.1, 0.2, 0.3, ...] + [pos_0の情報]
  ate (位置1) → [0.4, 0.5, 0.6, ...] + [pos_1の情報]
  fish(位置2) → [0.7, 0.8, 0.9, ...] + [pos_2の情報]

異なる位置 → 異なる埋め込み
```

---

## 絶対位置エンコーディング

**絶対位置エンコーディング**は、各位置に固有のベクトルを割り当てます。

### Sinusoidal位置エンコーディング

元祖Transformer（Vaswani et al. 2017）で使用された手法です。

#### 数式

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

ここで:
  pos: 位置（0, 1, 2, ...）
  i: 次元のインデックス（0, 1, 2, ..., d_model/2 - 1）
  d_model: モデルの埋め込み次元数
```

#### 特徴

1. **決定論的**: 学習不要、固定のパターン
2. **異なる周波数**: 次元ごとに異なる周波数の正弦波
3. **外挿可能**: 訓練時より長いシーケンスにも対応可能（理論上）

#### 可視化

```
d_model = 8, 位置0〜5の場合:

次元0,1 (低周波): sin/cos(pos / 10000^0) = sin/cos(pos)
  位置0: sin(0)=0.00,   cos(0)=1.00
  位置1: sin(1)=0.84,   cos(1)=0.54
  位置2: sin(2)=0.91,   cos(2)=-0.42
  ...

次元2,3 (中周波): sin/cos(pos / 10000^(2/8)) = sin/cos(pos / 5.62)
  位置0: sin(0)=0.00,   cos(0)=1.00
  位置1: sin(0.18)=0.18, cos(0.18)=0.98
  ...

次元6,7 (高周波): sin/cos(pos / 10000^(6/8)) = sin/cos(pos / 316)
  位置0〜5: ほぼ変化なし（周期が長い）
```

低次元は短い周期、高次元は長い周期を持ちます。

#### PyTorchでの実装

```python
import torch
import math

def sinusoidal_positional_encoding(seq_len, d_model):
    """
    seq_len: シーケンス長
    d_model: 埋め込み次元（偶数である必要がある）
    """
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe

# 使用例
pe = sinusoidal_positional_encoding(seq_len=100, d_model=512)
print(pe.shape)  # torch.Size([100, 512])
```

#### 使用方法

```python
# 入力埋め込みに加算
x = token_embeddings + positional_encodings
```

### 学習可能な位置エンコーディング

**学習可能な位置エンコーディング**は、位置ごとに学習可能なベクトルを用意します。

#### 実装

```python
import torch.nn as nn

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        self.pe = nn.Embedding(max_seq_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        return x + self.pe(positions)

# 使用例
pe_layer = LearnedPositionalEncoding(max_seq_len=512, d_model=768)
x = torch.randn(32, 128, 768)
x_with_pos = pe_layer(x)
```

#### 利点と欠点

**利点**:
- タスクに特化した位置情報を学習可能
- GPT-2、BERTなどで使用され、実証済み

**欠点**:
- 最大シーケンス長を超えられない（外挿不可）
- パラメータ数が増える（max_seq_len × d_model）

### 絶対位置エンコーディングの問題点

1. **長さの制限**: 学習可能な位置エンコーディングは最大長を超えられない
2. **外挿の困難**: 訓練時より長いシーケンスで性能劣化
3. **相対位置の学習困難**: 「AはBの3つ前」のような相対関係を直接表現できない

これらの問題を解決するため、**相対位置エンコーディング**が登場しました。

---

## 相対位置エンコーディング

**相対位置エンコーディング**は、**絶対位置**ではなく**トークン間の相対的な距離**を重視します。

### 動機

```
例: "The cat sat on the mat."

絶対位置:
  cat = 位置1
  sat = 位置2

別の文: "Yesterday, the cat sat on the mat."

絶対位置:
  cat = 位置2  ← 位置が変わった！
  sat = 位置3

相対位置:
  sat は常に cat の1つ後
  → 位置が変わっても関係性は同じ
```

### 利点

1. **汎化性**: 異なる文脈でも相対関係は一貫
2. **外挿性**: 訓練時より長いシーケンスでも機能
3. **文脈依存**: 各トークンペアの関係を個別に学習

### 実装のバリエーション

複数のアプローチがあります：

1. **T5の相対位置バイアス**: Attentionスコアにバイアスを加算
2. **ALiBi**: Attentionスコアに線形バイアス
3. **Rotary Embeddings (RoPE)**: Query/Keyを回転（nanochatで使用）

---

## Rotary Positional Embeddings (RoPE)

**Rotary Positional Embeddings (RoPE)** は、相対位置情報を**回転行列**で表現する手法です。

### 基本的なアイデア

**Query と Key を位置に応じて回転**させることで、内積（Attentionスコア）に相対位置情報を埋め込みます。

```
位置 m のQuery: q_m
位置 n のKey:   k_n

回転適用後:
  q'_m = R(m) q_m
  k'_n = R(n) k_n

Attentionスコア:
  q'_m · k'_n = (R(m) q_m) · (R(n) k_n)
              = q_m · R(m-n) k_n

内積が相対位置 (m-n) のみに依存！
```

### 2次元回転の直感

まず、2次元空間での回転を考えます。

#### 回転行列

```
角度θの回転行列:
R(θ) = [cos(θ)  -sin(θ)]
       [sin(θ)   cos(θ)]

ベクトル [x, y] を回転:
[x']   [cos(θ)  -sin(θ)] [x]
[y'] = [sin(θ)   cos(θ)] [y]

   = [x*cos(θ) - y*sin(θ)]
     [x*sin(θ) + y*cos(θ)]
```

#### 可視化

```
元のベクトル (x, y)
         ↑ y
         |  /
         | /  ← ベクトル
         |/
    -----+----→ x

回転後 (x', y')
         ↑
        /|
       / |  ← θ度回転
      /  |
    -----+----→
```

#### 回転の性質

```
2つのベクトルを同じ角度回転:
  v1 → R(θ) v1
  v2 → R(θ) v2

内積は不変:
  (R(θ) v1) · (R(θ) v2) = v1 · v2

2つのベクトルを異なる角度回転:
  v1 → R(θ1) v1
  v2 → R(θ2) v2

内積は相対角度に依存:
  (R(θ1) v1) · (R(θ2) v2) = v1 · R(θ1 - θ2) v2
```

この性質を位置エンコーディングに応用します。

### 数学的定義

#### 1. 位置ごとの回転角度

```
位置 m における次元 2i, 2i+1 の回転角度:
θ_i = m / (10000^(2i / d))

ここで:
  m: 位置インデックス
  i: 次元ペアのインデックス (i = 0, 1, ..., d/2 - 1)
  d: ヘッド次元数
```

**低次元**ほど高周波（大きな角度変化）、**高次元**ほど低周波（小さな角度変化）。

#### 2. ベクトルの回転

d次元ベクトルを d/2 個の2次元ペアに分割し、各ペアを独立に回転します。

```
元のベクトル: x = [x_0, x_1, x_2, x_3, ..., x_{d-1}]

ペア分割:
  ペア0: [x_0, x_1]
  ペア1: [x_2, x_3]
  ...
  ペア_{d/2-1}: [x_{d-2}, x_{d-1}]

各ペアを回転:
  ペアi を角度 θ_i で回転

回転後: x' = [x'_0, x'_1, x'_2, x'_3, ..., x'_{d-1}]
```

#### 3. 回転の数式

```
ペア i (次元 2i, 2i+1) の回転:

[x'_{2i}  ]   [cos(θ_i)  -sin(θ_i)] [x_{2i}  ]
[x'_{2i+1}] = [sin(θ_i)   cos(θ_i)] [x_{2i+1}]

展開すると:
  x'_{2i}   = x_{2i} * cos(θ_i) - x_{2i+1} * sin(θ_i)
  x'_{2i+1} = x_{2i} * sin(θ_i) + x_{2i+1} * cos(θ_i)
```

#### 4. 相対位置の性質

位置 m のQuery q_m と位置 n のKey k_n のAttentionスコア：

```
スコア = q'_m · k'_n

各次元ペア i について:
  スコア_i = (q_{2i} * cos(mθ_i) - q_{2i+1} * sin(mθ_i)) *
             (k_{2i} * cos(nθ_i) - k_{2i+1} * sin(nθ_i))
           + (q_{2i} * sin(mθ_i) + q_{2i+1} * cos(mθ_i)) *
             (k_{2i} * sin(nθ_i) + k_{2i+1} * cos(nθ_i))

三角関数の加法定理を使うと:
  スコア_i = q_{2i}*k_{2i}*cos((m-n)θ_i) + ...

相対位置 (m-n) のみに依存！
```

### 実装の詳細

#### ステップ1: 回転角度の事前計算

```python
def precompute_freqs(seq_len, head_dim, base=10000):
    """
    seq_len: 最大シーケンス長
    head_dim: ヘッド次元数
    base: 基底（10000が標準）
    """
    # 次元インデックス: 0, 2, 4, ..., head_dim-2
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)

    # 逆周波数: 1 / (base^(2i/d))
    inv_freq = 1.0 / (base ** (channel_range / head_dim))

    # 位置: 0, 1, 2, ..., seq_len-1
    t = torch.arange(seq_len, dtype=torch.float32)

    # 各(位置, 次元)ペアの角度
    freqs = torch.outer(t, inv_freq)  # (seq_len, head_dim/2)

    # cos と sin を事前計算
    cos = freqs.cos()  # (seq_len, head_dim/2)
    sin = freqs.sin()  # (seq_len, head_dim/2)

    return cos, sin
```

#### ステップ2: 回転の適用

```python
def apply_rotary_emb(x, cos, sin):
    """
    x: (batch, seq_len, n_head, head_dim)
    cos: (1, seq_len, 1, head_dim/2)
    sin: (1, seq_len, 1, head_dim/2)
    """
    # ベクトルを2つに分割
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]

    # 回転適用
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos

    # 再結合
    return torch.cat([y1, y2], dim=-1)
```

### 利点

1. **相対位置の自然な表現**: 内積が相対位置のみに依存
2. **外挿性**: 訓練時より長いシーケンスでも性能維持
3. **計算効率**: 要素ごとの乗算のみ（行列乗算不要）
4. **学習不要**: パラメータ追加なし

### 例: 4次元ベクトルの回転

```
位置 m=2, ヘッド次元 d=4 の場合:

次元ペア0 (i=0): θ_0 = 2 / 10000^0 = 2.0
次元ペア1 (i=1): θ_1 = 2 / 10000^(2/4) = 2 / 100 = 0.02

元のQuery: q = [1.0, 0.0, 1.0, 0.0]

ペア0を回転:
  [q_0']   [cos(2.0)  -sin(2.0)] [1.0]   [-0.416]
  [q_1'] = [sin(2.0)   cos(2.0)] [0.0] = [ 0.909]

ペア1を回転:
  [q_2']   [cos(0.02)  -sin(0.02)] [1.0]   [ 0.9998]
  [q_3'] = [sin(0.02)   cos(0.02)] [0.0] = [ 0.0200]

回転後: q' = [-0.416, 0.909, 0.9998, 0.0200]
```

---

## nanochatでの実装

### 1. 回転角度の事前計算（gpt.py:201-215）

```python
def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
    # デバイス自動検出
    if device is None:
        device = self.transformer.wte.weight.device

    # 次元インデックス: [0, 2, 4, ..., head_dim-2]
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)

    # 逆周波数: 1 / (10000^(2i/d))
    inv_freq = 1.0 / (base ** (channel_range / head_dim))

    # 位置: [0, 1, 2, ..., seq_len-1]
    t = torch.arange(seq_len, dtype=torch.float32, device=device)

    # 角度行列: (seq_len, head_dim/2)
    freqs = torch.outer(t, inv_freq)

    # cos/sin を事前計算
    cos, sin = freqs.cos(), freqs.sin()
    cos, sin = cos.bfloat16(), sin.bfloat16()  # bf16に変換

    # ブロードキャスト用に形状調整: (1, seq_len, 1, head_dim/2)
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]

    return cos, sin
```

**ポイント**:
- モデル初期化時に1回だけ計算
- `base=10000` は標準値（より大きくすると長距離依存に対応）
- `seq_len * 10` 分を事前計算（余裕を持たせる）

### 2. モデル初期化時の登録（gpt.py:169-171）

```python
self.rotary_seq_len = config.sequence_len * 10  # 10倍のバッファ
head_dim = config.n_embd // config.n_head
cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
self.register_buffer("cos", cos, persistent=False)
self.register_buffer("sin", sin, persistent=False)
```

**`register_buffer`**:
- モデルの一部として登録（デバイス移動時に自動で移動）
- `persistent=False`: チェックポイントに保存しない（再計算可能なため）

### 3. 回転の適用（gpt.py:41-49）

```python
def apply_rotary_emb(x, cos, sin):
    """
    x: (B, T, H, D) または (B, H, T, D)
    cos, sin: (1, T, 1, D/2)
    """
    assert x.ndim == 4  # multi-head attention

    # ベクトルを2つに分割
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]

    # 回転適用
    y1 = x1 * cos + x2 * sin     # nanochatの実装（符号注意）
    y2 = x1 * (-sin) + x2 * cos

    # 再結合
    out = torch.cat([y1, y2], 3)
    out = out.to(x.dtype)  # 入出力の型を一致

    return out
```

**注意**: nanochatの実装は標準的なRoPEと少し異なる符号を使用していますが、数学的には等価です。

### 4. Attentionでの使用（gpt.py:87-90）

```python
def forward(self, x, cos_sin, kv_cache):
    # QKV生成
    q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
    k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
    v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

    # Rotary Embeddings適用
    cos, sin = cos_sin
    q = apply_rotary_emb(q, cos, sin)
    k = apply_rotary_emb(k, cos, sin)
    # Valueには適用しない！

    # QK正規化
    q, k = norm(q), norm(k)

    # Attention計算
    # ...
```

**重要**: Rotary Embeddingsは **Query と Key のみ**に適用されます。Valueには適用しません。

### 5. KVキャッシュ使用時のオフセット（gpt.py:267-268）

```python
# KVキャッシュがある場合、現在位置にオフセット
T0 = 0 if kv_cache is None else kv_cache.get_pos()
cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]
```

推論時、すでにキャッシュされた部分をスキップするため、現在位置から開始します。

### 実行フロー

```
訓練時:
  入力: "The cat sat"
  位置: [0, 1, 2]
  cos_sin: self.cos[:, 0:3], self.sin[:, 0:3]

推論時（KVキャッシュあり）:
  ステップ1: "The" → 位置0
    cos_sin: self.cos[:, 0:1], self.sin[:, 0:1]

  ステップ2: "cat" → 位置1（キャッシュ位置=1）
    cos_sin: self.cos[:, 1:2], self.sin[:, 1:2]

  ステップ3: "sat" → 位置2（キャッシュ位置=2）
    cos_sin: self.cos[:, 2:3], self.sin[:, 2:3]
```

---

## 練習問題

### 問題1: Sinusoidal位置エンコーディングの計算

`d_model=4`, `pos=0` と `pos=1` の位置エンコーディングを計算してください。

```
次元0,1: sin/cos(pos / 10000^(0/4)) = sin/cos(pos)
次元2,3: sin/cos(pos / 10000^(2/4)) = sin/cos(pos / 100)
```

<details>
<summary>解答</summary>

```
位置0:
  次元0: sin(0) = 0.0
  次元1: cos(0) = 1.0
  次元2: sin(0) = 0.0
  次元3: cos(0) = 1.0
  PE(0) = [0.0, 1.0, 0.0, 1.0]

位置1:
  次元0: sin(1) ≈ 0.841
  次元1: cos(1) ≈ 0.540
  次元2: sin(1/100) = sin(0.01) ≈ 0.010
  次元3: cos(1/100) = cos(0.01) ≈ 1.000
  PE(1) = [0.841, 0.540, 0.010, 1.000]
```
</details>

### 問題2: 2次元回転の計算

ベクトル `[1, 0]` を角度 `π/4`（45度）回転させてください。

```
回転行列:
R(π/4) = [cos(π/4)  -sin(π/4)]
         [sin(π/4)   cos(π/4)]
```

<details>
<summary>解答</summary>

```
cos(π/4) = √2/2 ≈ 0.707
sin(π/4) = √2/2 ≈ 0.707

[x']   [0.707  -0.707] [1]   [0.707]
[y'] = [0.707   0.707] [0] = [0.707]

回転後: [0.707, 0.707]

可視化:
  元: (1, 0) → x軸方向
  回転後: (0.707, 0.707) → 対角線方向（45度）
```
</details>

### 問題3: RoPEの相対位置依存性

位置 m=3 と n=1 のQuery/Keyペアについて、Attentionスコアは相対位置 `m-n=2` のみに依存することを確認してください（概念的に）。

<details>
<summary>解答</summary>

```
位置3のQuery: q_3 を角度 3θ で回転 → q'_3
位置1のKey:   k_1 を角度 1θ で回転 → k'_1

Attentionスコア = q'_3 · k'_1

各次元ペアについて（2次元の場合）:
  q'_3 = R(3θ) q_3
  k'_1 = R(1θ) k_1

内積:
  q'_3 · k'_1 = (R(3θ) q_3) · (R(1θ) k_1)

回転の性質:
  (R(θ1) v1) · (R(θ2) v2) = v1 · R(θ1 - θ2) v2

したがって:
  q'_3 · k'_1 = q_3 · R(3θ - 1θ) k_1
              = q_3 · R(2θ) k_1

相対位置 (m-n) = 2 のみに依存！
```
</details>

### 問題4: RoPE vs 絶対位置エンコーディング

RoPEと絶対位置エンコーディングの主な違いを3つ挙げてください。

<details>
<summary>解答</summary>

```
1. 情報の種類:
   - 絶対位置: 各トークンの絶対的な位置
   - RoPE: トークン間の相対的な距離

2. 外挿性:
   - 絶対位置: 訓練時より長いシーケンスで性能劣化
   - RoPE: 外挿が比較的容易（相対関係が保持される）

3. 適用方法:
   - 絶対位置: 埋め込みに加算
   - RoPE: Query/Keyを回転変換
```
</details>

---

## まとめ

### 位置エンコーディングの進化

```
絶対位置エンコーディング:
  - Sinusoidal (元祖Transformer)
  - 学習可能 (BERT, GPT-2)
  問題: 外挿困難、相対位置の表現が間接的

相対位置エンコーディング:
  - T5バイアス
  - ALiBi
  - RoPE (nanochatで使用)
  利点: 外挿可能、汎化性向上
```

### RoPEの核心

```
基本アイデア:
  Query/Keyを位置に応じて回転
  → 内積が相対位置のみに依存

数式:
  位置 m: q'_m = R(mθ) q_m
  位置 n: k'_n = R(nθ) k_n
  スコア: q'_m · k'_n ∝ R((m-n)θ)

実装:
  1. 事前計算: cos/sin(pos / 10000^(2i/d))
  2. 分割: [x_0, x_1, x_2, x_3] → [x_0, x_1], [x_2, x_3]
  3. 回転: 各ペアを独立に回転
  4. 再結合
```

### nanochatの戦略

```python
# 事前計算（初期化時）
cos, sin = precompute_rotary_embeddings(seq_len, head_dim)

# 適用（forward時）
q = apply_rotary_emb(q, cos, sin)
k = apply_rotary_emb(k, cos, sin)
# Valueには適用しない

# KVキャッシュ使用時はオフセット
cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]
```

**特徴**:
- 学習不要（パラメータ0）
- 計算効率的（要素ごとの乗算のみ）
- 外挿可能（訓練より長いシーケンスでも動作）

### 次のステップ

- [数学07: Attention機構の数式](07-attention.md) - RoPEが使われる文脈
- [数学10: 最適化アルゴリズム](10-optimization.md) - 訓練の最適化
- [第4章: モデルの詳細実装](../04-model-implementation.md) - 全体アーキテクチャ
