# 第7章: Attention機構の数式

## 目次
- [Attentionとは何か](#attentionとは何か)
- [Query、Key、Valueの概念](#querykeyvalueの概念)
- [Scaled Dot-Product Attention](#scaled-dot-product-attention)
  - [スコア計算](#スコア計算)
  - [スケーリング係数の意義](#スケーリング係数の意義)
  - [Softmaxによる正規化](#softmaxによる正規化)
  - [重み付き和の計算](#重み付き和の計算)
- [Causal Masking（因果マスク）](#causal-masking因果マスク)
- [Multi-Head Attention](#multi-head-attention)
- [Grouped Query Attention (GQA/MQA)](#grouped-query-attention-gqamqa)
- [nanochatでの実装](#nanochatでの実装)
- [練習問題](#練習問題)

---

## Attentionとは何か

**Attention（注意機構）** は、Transformerモデルの中核となる仕組みです。入力シーケンス内の各位置が、他のどの位置の情報に「注目（attend）」すべきかを動的に決定します。

### なぜAttentionが必要か？

従来のRNN（再帰型ニューラルネットワーク）は、シーケンスを順番に処理するため：
- **長距離依存関係の学習が困難**（情報が薄れていく）
- **並列化が難しい**（1ステップずつ順番に処理）

Attentionは、これらの問題を解決します：
- **任意の距離の関係を直接モデル化**（遠くの単語も直接参照）
- **完全に並列化可能**（すべての位置を同時に計算）

### Attentionの直感的理解

文「The cat sat on the mat because it was tired.」を考えます。

「it」が何を指すか理解するには、「cat」に注目する必要があります。Attentionは、この「注目すべき場所」を自動的に学習します。

```
単語:    The   cat   sat   on   the   mat   because   it   was   tired
注目度:  0.05  0.70  0.10  0.03  0.02  0.05    0.03  1.0   0.02   0.00
         ↑     ↑                                      ↑
         わずか 強く注目                              クエリ
```

「it」（クエリ）が「cat」（キー）に強く注目（高いattentionスコア）し、「cat」の情報（バリュー）を取得します。

---

## Query、Key、Valueの概念

Attentionは、**Query（質問）**、**Key（鍵）**、**Value（値）** という3つの役割に基づいて動作します。

### 比喩：図書館での情報検索

```
図書館の例:
  - Query  = あなたの検索クエリ「Transformerについて」
  - Key    = 各本の索引カード（タイトル、キーワード）
  - Value  = 本の実際の内容

プロセス:
  1. あなたのクエリと各本の索引カードを比較
  2. 関連度の高い本を特定（スコア計算）
  3. 関連度に応じて重み付けして内容を取得
```

### 数学的定義

入力ベクトル `x` から、3つの行列を生成します：

```
Q = x W_Q    (クエリ行列)
K = x W_K    (キー行列)
V = x W_V    (バリュー行列)

ここで:
  x: (バッチサイズ, シーケンス長, 埋め込み次元)
  W_Q, W_K, W_V: 学習可能な重み行列
```

重要な点：
- **同じ入力xから3つの異なる表現を作成**
- **各役割に特化した変換**（異なる重み行列を使用）

### 各要素の役割

| 要素 | 役割 | サイズ |
|------|------|--------|
| Query | 「何を探しているか」を表現 | (B, n_head, T_q, d_k) |
| Key | 「この位置には何があるか」を表現 | (B, n_head, T_k, d_k) |
| Value | 「実際に取り出す情報」 | (B, n_head, T_v, d_v) |

通常、`d_k = d_v = d_model / n_head`（ヘッド次元）です。

---

## Scaled Dot-Product Attention

Attentionの計算は、以下の4ステップで行われます。

### 全体の数式

```
Attention(Q, K, V) = softmax(Q K^T / √d_k) V

ステップ分解:
  1. スコア = Q K^T              (内積で類似度計算)
  2. スケーリング = スコア / √d_k  (分散を安定化)
  3. 重み = softmax(スケーリング)   (確率分布に変換)
  4. 出力 = 重み V               (重み付き和)
```

### ステップ1: スコア計算

**Query** と **Key** の内積で、各位置間の関連性を計算します。

```
スコア = Q K^T

形状の変化:
  Q: (B, H, T_q, d_k)
  K: (B, H, T_k, d_k)
  K^T: (B, H, d_k, T_k)
  スコア: (B, H, T_q, T_k)
```

**スコア[i, j]** は、クエリ位置 `i` がキー位置 `j` にどれだけ注目すべきかを示します。

#### 内積が類似度を表す理由

2つのベクトル `q` と `k` の内積：

```
q · k = |q| |k| cos(θ)

ここで θ はベクトル間の角度

- θ = 0° (同じ方向)  → cos(0°) = 1   → 内積が大きい (高い類似度)
- θ = 90° (直交)     → cos(90°) = 0  → 内積が0 (無関係)
- θ = 180° (反対)    → cos(180°) = -1 → 内積が負 (反対の意味)
```

可視化：
```
ベクトルq
    ↑ θ=30°
    |  /
    | / ← ベクトルk (類似方向 → 内積大)
    |/
    +------→

ベクトルq
    ↑
    |
    | ← ベクトルk (直交 → 内積≈0)
    ⟋
    +------→
```

### ステップ2: スケーリング係数の意義

内積の結果を `√d_k` で割ります。

```
スケールされたスコア = スコア / √d_k
```

#### なぜスケーリングが必要か？

**問題**: 次元 `d_k` が大きくなると、内積の値が大きくなりすぎる

2つのランダムなベクトル `q`, `k`（各要素が平均0、分散1）の内積を考えます：

```
q · k = q_1*k_1 + q_2*k_2 + ... + q_{d_k}*k_{d_k}

期待値: E[q · k] = 0
分散: Var[q · k] = d_k
```

つまり、**内積の分散は次元数に比例**します。

#### 具体例

```python
import torch

d_k = 64
q = torch.randn(d_k)
k = torch.randn(d_k)

score = q @ k
print(f"スコア: {score:.2f}")  # 例: 8.5 (大きな値)

# d_k = 512 の場合
d_k = 512
q = torch.randn(d_k)
k = torch.randn(d_k)

score = q @ k
print(f"スコア: {score:.2f}")  # 例: 24.3 (さらに大きい)
```

スコアが大きすぎると、Softmaxの入力が極端になり：

```
softmax([100, 2, 1]) ≈ [1.0, 0.0, 0.0]  # ほぼワンホット

→ 勾配消失（ほとんどの要素の勾配がゼロに）
```

#### スケーリングの効果

```
スケールされたスコア = スコア / √d_k

期待値: E[スケールされたスコア] = 0
分散: Var[スケールされたスコア] = 1  ← 安定！
```

スケーリングにより、**次元数に関わらず分散が1に正規化**されます。

```python
scaled_score = score / (d_k ** 0.5)
print(f"スケールされたスコア: {scaled_score:.2f}")  # 例: 1.06 (適度な値)
```

### ステップ3: Softmaxによる正規化

スケールされたスコアをSoftmaxに通して、確率分布に変換します。

```
attention_weights = softmax(スケールされたスコア)

各行の和が1になる確率分布:
  Σ_j attention_weights[i, j] = 1
```

詳細は [数学03: Softmax関数](03-softmax.md) を参照。

#### 例

```
位置:           0      1      2      3
スコア:       [2.1,   4.5,   1.8,   3.2]
            ↓ softmax
重み:        [0.10,  0.63,  0.07,  0.20]
            ↑
            和 = 1.0
```

### ステップ4: 重み付き和の計算

Attention重みを使って、Valueの重み付き和を計算します。

```
出力 = attention_weights V

形状:
  attention_weights: (B, H, T_q, T_k)
  V: (B, H, T_k, d_v)
  出力: (B, H, T_q, d_v)
```

#### 直感的理解

各クエリ位置について、**関連するValue情報を適切な比率で混ぜ合わせる**：

```
出力[i] = Σ_j attention_weights[i, j] * V[j]

例:
  位置i の出力 = 0.10*V[0] + 0.63*V[1] + 0.07*V[2] + 0.20*V[3]
                 ↑           ↑ 最も重要    ↑          ↑
                 わずか       (63%)        わずか      やや重要
```

可視化：
```
クエリ位置 i について:

Value[0] ──→ ×0.10 ──┐
Value[1] ──→ ×0.63 ──┼──→ 出力[i]
Value[2] ──→ ×0.07 ──┤
Value[3] ──→ ×0.20 ──┘

Value[1] の情報が最も強く反映される
```

### 完全な計算例

具体的な数値で追ってみましょう。

```
入力:
  Q = [[1, 0],    K = [[1, 0],    V = [[10, 20],
       [0, 1]]         [0, 1]]         [30, 40]]

  d_k = 2

ステップ1: スコア計算
  スコア = Q K^T
        = [[1, 0],  [[1, 0],  ^T
           [0, 1]]   [0, 1]]

        = [[1, 0],  [[1, 0],
           [0, 1]]   [0, 1]]

        = [[1*1+0*0, 1*0+0*1],
           [0*1+1*0, 0*0+1*1]]

        = [[1, 0],
           [0, 1]]

ステップ2: スケーリング
  スケールされたスコア = スコア / √2
                      = [[0.707, 0],
                         [0, 0.707]]

ステップ3: Softmax
  重み = softmax([[0.707, 0],
                  [0, 0.707]])
       ≈ [[0.67, 0.33],
          [0.33, 0.67]]

  (各行の和が1)

ステップ4: 重み付き和
  出力 = 重み V
       = [[0.67, 0.33],  [[10, 20],
          [0.33, 0.67]]   [30, 40]]

       = [[0.67*10+0.33*30, 0.67*20+0.33*40],
          [0.33*10+0.67*30, 0.33*20+0.67*40]]

       = [[16.6, 26.6],
          [23.4, 33.4]]
```

---

## Causal Masking（因果マスク）

**自己回帰言語モデル**（GPTなど）では、**未来の情報を見てはいけない**という制約があります。

### なぜマスクが必要か？

訓練時、シーケンス全体が同時に入力されますが：
- 位置 `i` は、位置 `i+1` 以降を見てはいけない
- そうしないと、**カンニング**になる（次の単語を見ながら予測）

### Causal Maskの仕組み

Softmax前のスコアに、**未来の位置に `-∞` を設定**します。

```
マスク前のスコア:
  [[2.1,  4.5,  1.8,  3.2],   位置0は全てを参照可能
   [1.2,  3.4,  2.8,  1.9],   位置1は0,1のみ参照
   [0.8,  2.1,  4.0,  2.5],   位置2は0,1,2のみ参照
   [1.5,  2.9,  1.3,  3.7]]   位置3は全てを参照可能

Causal Mask (下三角行列):
  [[1,  0,  0,  0],
   [1,  1,  0,  0],
   [1,  1,  1,  0],
   [1,  1,  1,  1]]

マスク適用 (0の位置に-∞を設定):
  [[2.1,  -∞,  -∞,  -∞],
   [1.2, 3.4,  -∞,  -∞],
   [0.8, 2.1, 4.0,  -∞],
   [1.5, 2.9, 1.3, 3.7]]

Softmax適用後:
  [[1.0,  0.0,  0.0,  0.0],   位置0は自分のみ
   [0.1,  0.9,  0.0,  0.0],   位置1は0,1の混合
   [0.05, 0.15, 0.8, 0.0],    位置2は主に自分
   [0.08, 0.29, 0.06, 0.57]]  位置3は全体の混合
```

**softmax(-∞) = 0** なので、未来の位置の重みが完全にゼロになります。

### 可視化

```
位置0: "The"
  参照可能: [The]
  ●

位置1: "cat"
  参照可能: [The, cat]
  ● ●

位置2: "sat"
  参照可能: [The, cat, sat]
  ● ● ●

位置3: "down"
  参照可能: [The, cat, sat, down]
  ● ● ● ●

下三角パターン（因果的）
```

### PyTorchでの実装

```python
# PyTorch組み込み関数を使用
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

# 手動実装
def causal_attention(q, k, v):
    d_k = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / (d_k ** 0.5)

    # Causal maskの作成
    T = scores.size(-1)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))

    # マスク適用
    scores = scores.masked_fill(~mask, float('-inf'))

    # Softmax + 重み付き和
    weights = F.softmax(scores, dim=-1)
    return weights @ v
```

---

## Multi-Head Attention

**Multi-Head Attention** は、複数の異なる「注目パターン」を並列に学習します。

### 動機

単一のAttentionでは、1つの関係性しか捉えられません：
- 「it」→「cat」（主語を参照）
- 「sat」→「on」（前置詞との関係）
- 「tired」→「cat」（形容詞と名詞の関係）

これらを**同時に**学習するため、複数のヘッドを使います。

### 数式

```
MultiHead(Q, K, V) = Concat(head_1, head_2, ..., head_h) W_O

ここで:
  head_i = Attention(Q W_Q^i, K W_K^i, V W_V^i)

各ヘッドは独立した重み行列を持つ
```

### ステップバイステップ

```
1. 入力を h 個のヘッドに分割
   d_model = 512, h = 8 の場合
   各ヘッドの次元 = 512 / 8 = 64

2. 各ヘッドで独立にAttentionを計算
   head_1: (B, T, 64)
   head_2: (B, T, 64)
   ...
   head_8: (B, T, 64)

3. 結果を連結
   Concat: (B, T, 512)

4. 出力射影
   出力 = Concat W_O
```

### 可視化

```
入力 (B, T, 512)
    |
    | 分割
    ├────────┬────────┬────────┬─────→ (8ヘッド)
    ↓        ↓        ↓        ↓
  Head 1   Head 2   Head 3  ... Head 8
  (64次元) (64次元) (64次元)    (64次元)
    |        |        |           |
  Attn 1  Attn 2  Attn 3  ...  Attn 8
    |        |        |           |
    ├────────┴────────┴───────────┘
    | Concat
    ↓
  (B, T, 512)
    |
  W_O (出力射影)
    ↓
  出力 (B, T, 512)
```

### なぜ効果的か？

**異なるヘッドが異なるパターンを学習**：

```
例: "The cat sat on the mat."

Head 1: 主語-動詞の関係
  cat → sat (0.8)

Head 2: 前置詞の関係
  sat → on (0.9)
  on → mat (0.7)

Head 3: 冠詞-名詞の関係
  The → cat (0.6)
  the → mat (0.5)

Head 4: 長距離依存
  mat → cat (0.4)
```

各ヘッドが**異なる言語的関係を専門化**します。

### PyTorchでの実装

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.head_dim = d_model // n_head

        # QKV射影
        self.c_q = nn.Linear(d_model, d_model)
        self.c_k = nn.Linear(d_model, d_model)
        self.c_v = nn.Linear(d_model, d_model)

        # 出力射影
        self.c_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.size()

        # QKV計算
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)

        # ヘッドをバッチ次元に
        q = q.transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention計算
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # ヘッドを連結
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 出力射影
        y = self.c_proj(y)
        return y
```

---

## Grouped Query Attention (GQA/MQA)

**Grouped Query Attention (GQA)** と **Multi-Query Attention (MQA)** は、Multi-Head Attentionの効率的な変種です。

### 問題: KVキャッシュのメモリ使用量

推論時、KVキャッシュが大量のメモリを消費します：

```
標準的なMulti-Head Attention:
  - 8ヘッド、各ヘッド64次元
  - KVキャッシュ: 8 × 2 (K, V) × 64 = 1024次元分

シーケンス長2048、バッチサイズ32の場合:
  メモリ: 32 × 2048 × 1024 × 4バイト ≈ 256 MB (1層あたり)
```

### GQA/MQAの解決策

**Key と Value のヘッド数を減らす**：

```
標準 MHA:
  Query: 8ヘッド
  Key:   8ヘッド
  Value: 8ヘッド

MQA (Multi-Query Attention):
  Query: 8ヘッド
  Key:   1ヘッド  ← 共有
  Value: 1ヘッド  ← 共有

GQA (Grouped Query Attention):
  Query: 8ヘッド
  Key:   2ヘッド  ← グループ化
  Value: 2ヘッド  ← グループ化

  グループ分け:
    Query [0,1,2,3] → KV head 0
    Query [4,5,6,7] → KV head 1
```

### メモリ削減効果

```
MQA (1 KVヘッド):
  KVキャッシュ: 1 × 2 × 64 = 128次元分
  削減率: 1024 / 128 = 8倍削減！

GQA (2 KVヘッド):
  KVキャッシュ: 2 × 2 × 64 = 256次元分
  削減率: 1024 / 256 = 4倍削減
```

### 実装: KVヘッドの複製

```python
# nanochatの実装から (gpt.py:100-101)
nrep = self.n_head // self.n_kv_head  # 8 // 2 = 4
k, v = repeat_kv(k, nrep), repeat_kv(v, nrep)

def repeat_kv(x, nrep):
    """KVヘッドを複製してQヘッド数に合わせる"""
    if nrep == 1:
        return x
    B, n_kv_head, T, head_dim = x.shape
    # 各KVヘッドを nrep 回複製
    x = x[:, :, None, :, :].expand(B, n_kv_head, nrep, T, head_dim)
    return x.reshape(B, n_kv_head * nrep, T, head_dim)
```

可視化：
```
GQA (2 KVヘッド、8 Qヘッド):

KV Head 0 ─┬→ Q Head 0
           ├→ Q Head 1
           ├→ Q Head 2
           └→ Q Head 3

KV Head 1 ─┬→ Q Head 4
           ├→ Q Head 5
           ├→ Q Head 6
           └→ Q Head 7

各KVヘッドが4つのQヘッドで共有される
```

### 性能トレードオフ

| 方式 | KVキャッシュ | 表現力 | 用途 |
|------|------------|--------|------|
| MHA | 大 | 高 | 訓練時、小モデル |
| GQA | 中 | 中 | バランス型（推奨） |
| MQA | 小 | 低 | 推論特化、大モデル |

nanochatは **GQA** を採用：
```python
# nanochat/gpt.py:69-73
self.n_head = config.n_head        # 8 (Qヘッド数)
self.n_kv_head = config.n_kv_head  # 2 (KVヘッド数)
```

---

## nanochatでの実装

### 完全な実装（gpt.py:65-126）

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head        # 8
        self.n_kv_head = config.n_kv_head  # 2 (GQA)
        self.n_embd = config.n_embd        # 512
        self.head_dim = self.n_embd // self.n_head  # 64

        # QKV射影行列
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)

        # 出力射影
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()

        # 1. QKV生成
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # 2. Rotary Embeddings (位置情報を付与)
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)

        # 3. QK正規化
        q, k = norm(q), norm(k)

        # 4. ヘッドをバッチ次元に
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        # (B, T, H, D) -> (B, H, T, D)

        # 5. KVキャッシュに挿入
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)

        Tq = q.size(2)  # クエリ数
        Tk = k.size(2)  # キー数（キャッシュ含む）

        # 6. GQA: KVヘッドを複製
        nrep = self.n_head // self.n_kv_head
        k, v = repeat_kv(k, nrep), repeat_kv(v, nrep)

        # 7. Attention計算（ケース分け）
        if kv_cache is None or Tq == Tk:
            # 訓練時: シンプルなCausal Attention
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif Tq == 1:
            # 推論時（単一トークン）: キャッシュ全体を参照
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            # 推論時（複数トークン）: カスタムマスク
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device)
            prefix_len = Tk - Tq
            if prefix_len > 0:
                attn_mask[:, :prefix_len] = True  # プレフィックス全体を参照
            # チャンク内はCausal
            attn_mask[:, prefix_len:] = torch.tril(
                torch.ones((Tq, Tq), dtype=torch.bool, device=q.device)
            )
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        # 8. ヘッドを連結
        y = y.transpose(1, 2).contiguous().view(B, T, -1)

        # 9. 出力射影
        y = self.c_proj(y)
        return y
```

### 重要なポイント

1. **GQA**: `n_head=8`, `n_kv_head=2` でメモリ効率化
2. **Rotary Embeddings**: 位置情報を直接QKに埋め込む
3. **QK Normalization**: 安定性向上
4. **KVキャッシュ**: 推論時の計算量削減
5. **ケース分けAttention**: 訓練/推論で最適化

### F.scaled_dot_product_attention の内部

PyTorchの組み込み関数は、以下を自動で行います：

```python
# 擬似コード
def scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False):
    # 1. スコア計算
    scores = q @ k.transpose(-2, -1)

    # 2. スケーリング
    scores = scores / (q.size(-1) ** 0.5)

    # 3. マスク適用
    if is_causal:
        mask = torch.tril(torch.ones(...))
        scores = scores.masked_fill(~mask, float('-inf'))
    elif attn_mask is not None:
        scores = scores.masked_fill(~attn_mask, float('-inf'))

    # 4. Softmax
    weights = F.softmax(scores, dim=-1)

    # 5. 重み付き和
    return weights @ v
```

実際には、**FlashAttention** など高度な最適化が使われます（メモリ効率とスピードを改善）。

---

## 練習問題

### 問題1: 基本的なAttention計算

以下のQuery、Key、ValueでAttentionを計算してください（`d_k=2`）。

```
Q = [[2, 0]]    K = [[1, 0],    V = [[5],
                     [0, 1]]         [10]]
```

ステップ：
1. スコアを計算
2. `√d_k` でスケーリング
3. Softmaxを適用
4. Valueの重み付き和を計算

<details>
<summary>解答</summary>

```
1. スコア = Q K^T
         = [[2, 0]] [[1, 0],^T
                     [0, 1]]
         = [[2, 0]] [[1, 0],
                     [0, 1]]
         = [[2*1+0*0, 2*0+0*1]]
         = [[2, 0]]

2. スケーリング = [2, 0] / √2 = [1.414, 0]

3. Softmax = exp([1.414, 0]) / sum(exp([1.414, 0]))
           = [4.113, 1.0] / 5.113
           ≈ [0.804, 0.196]

4. 出力 = [0.804, 0.196] [[5],
                          [10]]
        = 0.804 * 5 + 0.196 * 10
        = 4.02 + 1.96
        = 5.98
```
</details>

### 問題2: Causal Maskの効果

以下のスコア行列にCausal Maskを適用し、Softmax後の重みを計算してください。

```
スコア = [[1.0, 2.0, 3.0],
          [0.5, 1.5, 2.5],
          [1.2, 0.8, 2.0]]
```

<details>
<summary>解答</summary>

```
Causal Mask適用:
  [[1.0,  -∞,  -∞],
   [0.5, 1.5,  -∞],
   [1.2, 0.8, 2.0]]

Softmax (各行):
  行0: softmax([1.0, -∞, -∞]) = [1.0, 0.0, 0.0]
  行1: softmax([0.5, 1.5, -∞]) ≈ [0.27, 0.73, 0.0]
  行2: softmax([1.2, 0.8, 2.0]) ≈ [0.24, 0.16, 0.60]

結果:
  [[1.0,  0.0,  0.0],
   [0.27, 0.73, 0.0],
   [0.24, 0.16, 0.60]]
```

位置0は自分のみ、位置1は0と1、位置2は全てを参照できる。
</details>

### 問題3: Multi-Head Attention

`d_model=128`, `n_head=4` のMulti-Head Attentionで：
1. 各ヘッドの次元数は？
2. 8ヘッドに変更した場合の各ヘッド次元は？
3. GQAで `n_kv_head=2` にした場合、各Qヘッドに対応するKVヘッド数は？

<details>
<summary>解答</summary>

```
1. 各ヘッドの次元 = d_model / n_head = 128 / 4 = 32

2. 8ヘッドの場合 = 128 / 8 = 16

3. GQA設定:
   - Qヘッド数: 4
   - KVヘッド数: 2
   - 複製数 nrep = 4 / 2 = 2

   各KVヘッドが2つのQヘッドで共有される:
     KV Head 0 → Q Head 0, 1
     KV Head 1 → Q Head 2, 3
```
</details>

### 問題4: スケーリング係数の必要性

`d_k=256` の場合と `d_k=64` の場合で、ランダムベクトルの内積の分散を比較してください（概算）。スケーリング後の分散も計算してください。

<details>
<summary>解答</summary>

```
各要素が標準正規分布 N(0, 1) のランダムベクトル q, k の内積:

d_k=256 の場合:
  Var(q · k) = d_k = 256
  標準偏差 = √256 = 16

d_k=64 の場合:
  Var(q · k) = d_k = 64
  標準偏差 = √64 = 8

比率: 256 / 64 = 4倍の分散差！

スケーリング後:
  d_k=256: Var((q · k) / √256) = 256 / 256 = 1
  d_k=64:  Var((q · k) / √64) = 64 / 64 = 1

両方とも分散1に正規化される！
```
</details>

---

## まとめ

### Attentionの核心

1. **Query-Key-Value構造**: 検索のアナロジー
2. **Scaled Dot-Product**: 内積 + スケーリング + Softmax + 重み付き和
3. **Causal Mask**: 自己回帰モデルで未来を見ない
4. **Multi-Head**: 複数の関係性を並列学習
5. **GQA/MQA**: KVキャッシュのメモリ削減

### 数式のまとめ

```
基本Attention:
  Attention(Q, K, V) = softmax(Q K^T / √d_k) V

Multi-Head Attention:
  MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O
  head_i = Attention(Q W_Q^i, K W_K^i, V W_V^i)

Causal Attention:
  mask[i, j] = (i >= j)  # 下三角行列
  Attention = softmax((Q K^T / √d_k) + mask) V
```

### nanochatの実装ポイント

- **GQA**: メモリ効率（`n_head=8`, `n_kv_head=2`）
- **Rotary Embeddings**: 位置情報の埋め込み
- **KVキャッシュ**: 推論高速化
- **PyTorchの最適化**: `F.scaled_dot_product_attention` (FlashAttention)

Attentionは、Transformerの**心臓部**です。この機構により、LLMは長距離の依存関係を効率的に学習できます。

### 次のステップ

- [数学08: 正規化手法](08-normalization.md) - LayerNorm、RMSNorm
- [数学09: 位置エンコーディング](09-positional-encoding.md) - Rotary Embeddings詳細
- [第4章: モデルの詳細実装](../04-model-implementation.md) - Transformerブロック全体
