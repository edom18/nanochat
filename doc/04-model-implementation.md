# 第4章: モデルの詳細実装

## 目次
1. [GPTConfigの設計](#gptconfigの設計)
2. [RMSNorm: パラメータなし正規化](#rmsnormパラメータなし正規化)
3. [Rotary Embeddings（RoPE）](#rotary-embeddingsrope)
4. [CausalSelfAttention: 自己注意機構](#causalselfattention自己注意機構)
5. [MLP: フィードフォワードネットワーク](#mlpフィードフォワードネットワーク)
6. [Block: Transformerブロック](#blocktransformerブロック)
7. [GPT: メインモデルクラス](#gptメインモデルクラス)
8. [重みの初期化](#重みの初期化)
9. [オプティマイザーのセットアップ](#オプティマイザーのセットアップ)
10. [次章への導入](#次章への導入)

---

## GPTConfigの設計

**GPTConfig**は、モデルのアーキテクチャを定義する設定クラスです（gpt.py:26-34）。

```python
@dataclass
class GPTConfig:
    sequence_len: int = 1024       # 最大シーケンス長
    vocab_size: int = 50304        # 語彙サイズ
    n_layer: int = 12              # Transformerブロックの数（深さ）
    n_head: int = 6                # Queryヘッドの数
    n_kv_head: int = 6             # Key/Valueヘッドの数（Multi-Query Attention用）
    n_embd: int = 768              # 埋め込み次元（モデルの幅）
```

### 各パラメータの意味

#### 1. `sequence_len`（シーケンス長）
一度に処理できる最大トークン数。

- **デフォルト**: 1024
- **nanochatの設定**: 2048（より長い文脈を処理）
- **制約**: メモリ使用量は`sequence_len`の2乗に比例（Attentionの計算量）

#### 2. `vocab_size`（語彙サイズ）
トークナイザーの語彙数。

- **デフォルト**: 50304
- **nanochatの設定**: ~50304（トークナイザー訓練時に決定、実際は65,536に近い）
- **役割**: 埋め込み層と出力層の次元を決定

#### 3. `n_layer`（レイヤー数）
Transformerブロックを何層重ねるか。

- **デフォルト**: 12
- **nanochatの実際の値**:
  - d20モデル: 20層（561Mパラメータ）
  - d32モデル: 32層（1.9Bパラメータ）
- **トレードオフ**: 深いほど表現力が高いが、訓練時間とメモリが増加

#### 4. `n_head`（Queryヘッド数）
Attentionで何個のヘッドに分割するか。

- **デフォルト**: 6
- **計算**: `head_dim = n_embd // n_head`（各ヘッドの次元）
- **制約**: `n_embd`は`n_head`で割り切れる必要がある

#### 5. `n_kv_head`（Key/Valueヘッド数）
Multi-Query Attention（MQA）用。

- **デフォルト**: 6（`n_head`と同じ = 標準Attention）
- **MQAの場合**: `n_kv_head < n_head`（例: `n_kv_head=1`で全Queryヘッドが1つのK/Vを共有）
- **効果**: メモリ削減（特にKVキャッシュ）

#### 6. `n_embd`（埋め込み次元）
モデルの「幅」を決定する重要なパラメータ。

- **デフォルト**: 768
- **nanochatの実際の値**: モデルサイズに応じて変化
- **影響**:
  - すべての線形層のサイズ
  - パラメータ数の大部分を占める
  - 計算量とメモリ使用量

### モデルサイズの計算

パラメータ数は主に以下で決まります:

```
総パラメータ数 ≈
  トークン埋め込み: vocab_size × n_embd
  + Transformerブロック × n_layer:
    - Attention: 4 × n_embd × n_embd（Q, K, V, 出力射影）
    - MLP: 2 × n_embd × (4 × n_embd)（拡張層、圧縮層）
  + 出力層: vocab_size × n_embd
```

**d20モデル（n_layer=20, n_embd=768）の場合**:
- 約561M（5.61億）パラメータ

**d32モデル（n_layer=32, n_embd=1536程度）の場合**:
- 約1.9B（19億）パラメータ

---

## RMSNorm: パラメータなし正規化

**RMSNorm（Root Mean Square Normalization）**は、学習可能なパラメータを持たないシンプルな正規化手法です（gpt.py:36-38）。

### 実装

```python
def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))
```

**PyTorchの`F.rms_norm`**を使用しており、非常にシンプルです。

### RMSNormの数学的定義

```
RMSNorm(x) = x / RMS(x)

ここで、RMS(x) = sqrt(mean(x²))
```

**具体例**:
```
x = [1, 2, 3, 4]
RMS(x) = sqrt((1² + 2² + 3² + 4²) / 4) = sqrt(30/4) = sqrt(7.5) ≈ 2.74
RMSNorm(x) = [1/2.74, 2/2.74, 3/2.74, 4/2.74] ≈ [0.36, 0.73, 1.09, 1.46]
```

### LayerNormとの違い

**LayerNorm**:
```
LayerNorm(x) = γ × (x - mean(x)) / std(x) + β
```
- 平均を引いて標準偏差で割る
- 学習可能なパラメータ`γ`（スケール）と`β`（バイアス）を持つ

**RMSNorm**:
```
RMSNorm(x) = x / RMS(x)
```
- 平均を引かない（中心化しない）
- 学習可能なパラメータなし

### なぜRMSNormを使うのか

1. **シンプル**: パラメータがない分、実装が単純
2. **高速**: 平均計算が不要
3. **効果的**: LayerNormと同等の性能（研究で実証済み）
4. **パラメータ削減**: モデル全体のパラメータ数を削減

nanochatでは、すべての正規化にこの`norm`関数を使用します。

**数学的詳細は**: [doc/math/08-layer-normalization.md](../doc/math/08-layer-normalization.md)（作成予定）

---

## Rotary Embeddings（RoPE）

**Rotary Embeddings（回転式位置エンコーディング、RoPE）**は、トークンの位置情報を回転変換で表現する手法です。

### apply_rotary_emb関数（gpt.py:41-49）

```python
def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]  # ベクトルを2分割
    y1 = x1 * cos + x2 * sin         # 回転変換（複素数の回転と等価）
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3)     # 再結合
    out = out.to(x.dtype)            # 元の型に戻す
    return out
```

### 入力と出力

**入力**:
- `x`: Query または Key のテンソル、形状 `(B, n_head, T, head_dim)`
- `cos`, `sin`: 事前計算された回転角度、形状 `(1, T, 1, head_dim//2)`

**出力**:
- 回転変換されたテンソル、形状は入力と同じ

### 回転変換の仕組み

1. **ベクトルを2分割**: `[x1, x2]`
2. **回転公式を適用**:
   ```
   y1 = x1 * cos + x2 * sin
   y2 = -x1 * sin + x2 * cos
   ```
   これは2次元平面での回転行列と等価:
   ```
   [y1]   [cos  sin] [x1]
   [y2] = [-sin cos] [x2]
   ```
3. **再結合**: `[y1, y2]`

### 位置情報の埋め込み

各位置`t`に対して、異なる回転角度を持ちます:

```
位置0: 回転角度 0°
位置1: 回転角度 θ
位置2: 回転角度 2θ
位置3: 回転角度 3θ
...
```

この回転により、**相対位置が内積に反映**されます。

### repeat_kv関数（gpt.py:52-61）

Multi-Query Attention用のヘルパー関数。

```python
def repeat_kv(x, n_rep):
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    if n_rep == 1:
        return x
    bs, n_kv_heads, slen, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )
```

**役割**: Key/ValueヘッドをQueryヘッド数に合わせて複製します。

**例**:
```
n_head = 6, n_kv_head = 2, n_rep = 3

入力 K: (B, 2, T, head_dim)  ← 2つのKVヘッド
出力 K: (B, 6, T, head_dim)  ← 各KVヘッドを3回複製
```

### 回転角度の事前計算（gpt.py:201-215）

```python
def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
    # チャンネルごとの周波数を計算
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))

    # 時間ステップごとの周波数
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)  # (seq_len, head_dim//2)

    # cosとsinを計算
    cos, sin = freqs.cos(), freqs.sin()
    cos, sin = cos.bfloat16(), sin.bfloat16()

    # バッチ次元とヘッド次元を追加
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    return cos, sin
```

**計算の流れ**:
1. 各チャンネルペアに周波数を割り当て（低周波から高周波）
2. 各時間ステップ`t`に対して`t × inv_freq`を計算
3. `cos`と`sin`を取得
4. バッチとヘッド次元を追加してブロードキャスト可能にする

**数学的詳細は**: [doc/math/09-positional-encoding.md](../doc/math/09-positional-encoding.md)（作成予定）

---

## CausalSelfAttention: 自己注意機構

**CausalSelfAttention**は、Transformerの核心をなす自己注意機構です（gpt.py:64-126）。

### クラス定義

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head

        # Query, Key, Value の射影層
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)

        # 出力射影層
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
```

**パラメータ**:
- `c_q`: Query射影（`n_embd` → `n_head × head_dim`）
- `c_k`: Key射影（`n_embd` → `n_kv_head × head_dim`）
- `c_v`: Value射影（`n_embd` → `n_kv_head × head_dim`）
- `c_proj`: 出力射影（`n_embd` → `n_embd`）

**注意**: すべて`bias=False`（バイアスなし）

### Forwardパス（gpt.py:79-126）

```python
def forward(self, x, cos_sin, kv_cache):
    B, T, C = x.size()  # バッチサイズ、シーケンス長、埋め込み次元

    # 1. Query, Key, Value の計算
    q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
    k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
    v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

    # 2. Rotary Embeddingsを適用
    cos, sin = cos_sin
    q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)

    # 3. QK Normalization
    q, k = norm(q), norm(k)

    # 4. ヘッド次元を先頭に移動: (B, T, H, D) → (B, H, T, D)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    # 5. KVキャッシュの処理（推論時）
    if kv_cache is not None:
        k, v = kv_cache.insert_kv(self.layer_idx, k, v)

    # 6. Multi-Query Attention: KVヘッドを複製
    nrep = self.n_head // self.n_kv_head
    k, v = repeat_kv(k, nrep), repeat_kv(v, nrep)

    # 7. Scaled Dot-Product Attention
    # (訓練時 / 推論時で分岐)
    if kv_cache is None or Tq == Tk:
        # 訓練時: causal attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    elif Tq == 1:
        # 推論時（1トークンのみ）: すべてのキャッシュに注目
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    else:
        # 推論時（複数トークン）: カスタムマスク
        # ... (詳細は後述)

    # 8. ヘッドを結合して出力射影
    y = y.transpose(1, 2).contiguous().view(B, T, -1)
    y = self.c_proj(y)
    return y
```

### 各ステップの詳細

#### ステップ1: Q, K, Vの計算

```python
q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
```

**形状の変化**:
```
x: (B, T, n_embd)
  ↓ c_q
(B, T, n_head * head_dim)
  ↓ view
(B, T, n_head, head_dim)
```

#### ステップ2: Rotary Embeddings

```python
q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
```

位置情報を**Query**と**Key**に埋め込みます（**Valueには適用しない**）。

#### ステップ3: QK Normalization

```python
q, k = norm(q), norm(k)
```

QueryとKeyを正規化してAttentionを安定化します。これは最新の研究で推奨されている手法です。

#### ステップ4-6: Multi-Query Attention

```python
# ヘッド次元を先頭に
q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

# KVヘッドを複製
nrep = self.n_head // self.n_kv_head
k, v = repeat_kv(k, nrep), repeat_kv(v, nrep)
```

KVヘッド数がQueryヘッド数より少ない場合、複製して合わせます。

#### ステップ7: Scaled Dot-Product Attention

PyTorchの`F.scaled_dot_product_attention`を使用します。これは効率的に実装されており、FlashAttentionなどの最適化を自動的に適用します。

**訓練時**:
```python
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```
- `is_causal=True`: 未来のトークンをマスク

**推論時（1トークン）**:
```python
y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
```
- すべてのキャッシュに注目可能

**推論時（複数トークン、gpt.py:113-121）**:
```python
# プレフィックス（キャッシュ）とチャンク（新規）の両方に注目
attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device)
prefix_len = Tk - Tq
if prefix_len > 0:
    attn_mask[:, :prefix_len] = True  # プレフィックスには自由に注目
# チャンク内はcausal
attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
```

#### ステップ8: 出力射影

```python
y = y.transpose(1, 2).contiguous().view(B, T, -1)
y = self.c_proj(y)
```

**形状の変化**:
```
y: (B, n_head, T, head_dim)
  ↓ transpose
(B, T, n_head, head_dim)
  ↓ view
(B, T, n_embd)
  ↓ c_proj
(B, T, n_embd)
```

**数学的詳細は**: [doc/math/07-attention-mechanism.md](../doc/math/07-attention-mechanism.md)（作成予定）

---

## MLP: フィードフォワードネットワーク

**MLP（Multi-Layer Perceptron）**は、Attentionの後に適用される非線形変換です（gpt.py:129-139）。

### 実装

```python
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)    # 拡張層
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)  # 圧縮層

    def forward(self, x):
        x = self.c_fc(x)         # (B, T, n_embd) → (B, T, 4*n_embd)
        x = F.relu(x).square()   # ReLU²活性化
        x = self.c_proj(x)       # (B, T, 4*n_embd) → (B, T, n_embd)
        return x
```

### なぜ4倍に拡張？

```
n_embd (例: 768)
    ↓ [拡張層]
4*n_embd (例: 3072)
    ↓ [ReLU²]
4*n_embd
    ↓ [圧縮層]
n_embd
```

**理由**:
1. **表現力の向上**: より多くの非線形変換を可能にする
2. **情報のボトルネック回避**: 一時的に次元を増やして情報を保持
3. **GPT-2以降の標準**: 4倍が経験的に良い性能を示す

### ReLU²活性化

```python
x = F.relu(x).square()
```

**ReLU²の定義**:
```
ReLU²(x) = (ReLU(x))² = (max(0, x))²
```

**例**:
```
x = [-2, -1, 0, 1, 2]
ReLU(x) = [0, 0, 0, 1, 2]
ReLU²(x) = [0, 0, 0, 1, 4]
```

**なぜReLU²？**
- **計算効率**: GELUより高速
- **シンプル**: 実装が単純
- **効果的**: 最近の研究で良い性能が報告されている

**GELUとの比較**:
- GELU: `x * Φ(x)`（Φは標準正規分布の累積分布関数）
- ReLU²: `max(0, x)²`

nanochatはシンプルさと効率を重視してReLU²を採用しています。

**数学的詳細は**: [doc/math/11-activation-functions.md](../doc/math/11-activation-functions.md)（作成予定）

---

## Block: Transformerブロック

**Block**は、AttentionとMLPを組み合わせたTransformerの基本単位です（gpt.py:142-151）。

### 実装

```python
class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache):
        x = x + self.attn(norm(x), cos_sin, kv_cache)  # 残差接続 + Attention
        x = x + self.mlp(norm(x))                       # 残差接続 + MLP
        return x
```

### Pre-Normalization構造

```
入力 x
  │
  ├─→ norm(x) → Attention → ＋─→ 中間出力
  └───────────────────────────┘
                          (残差接続)

中間出力
  │
  ├─→ norm(中間出力) → MLP → ＋─→ 出力
  └───────────────────────────┘
                       (残差接続)
```

**重要なポイント**:

1. **Pre-Normalization**: 正規化を**サブレイヤーの前**に適用
   - メリット: 訓練の安定性向上
   - 元のTransformer（Post-Norm）より訓練しやすい

2. **残差接続**: `x = x + ...`
   - メリット: 勾配消失問題を緩和
   - 深いネットワークの訓練を可能にする

### データの流れ

```python
# 入力
x: (B, T, n_embd)

# Attentionブロック
norm(x): (B, T, n_embd)  ← 正規化
  ↓ self.attn
attn_out: (B, T, n_embd)
  ↓ 残差接続
x = x + attn_out: (B, T, n_embd)

# MLPブロック
norm(x): (B, T, n_embd)  ← 正規化
  ↓ self.mlp
mlp_out: (B, T, n_embd)
  ↓ 残差接続
x = x + mlp_out: (B, T, n_embd)

# 出力
x: (B, T, n_embd)
```

---

## GPT: メインモデルクラス

**GPT**クラスは、すべてのコンポーネントを統合したメインモデルです（gpt.py:154-322）。

### 初期化（gpt.py:155-173）

```python
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),  # トークン埋め込み
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Rotary Embeddingsの事前計算
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # 埋め込みをbfloat16にキャスト
        self.transformer.wte.to(dtype=torch.bfloat16)
```

**構成要素**:
1. **`wte`**: トークン埋め込み層
2. **`h`**: Transformerブロックのリスト
3. **`lm_head`**: 出力層（語彙サイズに射影）
4. **`cos`, `sin`**: Rotary Embeddingsのキャッシュ

**`persistent=False`**: チェックポイントに保存しない（再計算可能なため）

### Forwardパス（gpt.py:259-291）

```python
def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
    B, T = idx.size()

    # 1. Rotary Embeddingsの取得
    T0 = 0 if kv_cache is None else kv_cache.get_pos()
    cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

    # 2. トークン埋め込み + 正規化
    x = self.transformer.wte(idx)  # (B, T) → (B, T, n_embd)
    x = norm(x)

    # 3. Transformerブロックを順次適用
    for block in self.transformer.h:
        x = block(x, cos_sin, kv_cache)

    # 4. 最終正規化
    x = norm(x)

    # 5. ロジット計算
    softcap = 15
    if targets is not None:
        # 訓練モード: 損失を計算
        logits = self.lm_head(x)
        logits = softcap * torch.tanh(logits / softcap)  # Logits softcap
        logits = logits.float()  # fp32で損失計算
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               targets.view(-1),
                               ignore_index=-1,
                               reduction=loss_reduction)
        return loss
    else:
        # 推論モード: ロジットを返す
        logits = self.lm_head(x)
        logits = softcap * torch.tanh(logits / softcap)
        return logits
```

### Logits Softcap

```python
softcap = 15
logits = softcap * torch.tanh(logits / softcap)
```

**目的**: ロジットの値を`[-15, 15]`の範囲に制限して数値安定性を向上。

**効果**:
```
logits = [100, -100, 10, -10, 0]
  ↓ softcap=15で制限
logits ≈ [15, -15, 9.99, -9.99, 0]
```

### 推論（generate）メソッド（gpt.py:293-322）

```python
@torch.inference_mode()
def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
    """自己回帰的にトークンを生成"""
    device = self.get_device()
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    ids = torch.tensor([tokens], dtype=torch.long, device=device)

    for _ in range(max_tokens):
        # Forward
        logits = self.forward(ids)  # (B, T, vocab_size)
        logits = logits[:, -1, :]   # 最後のトークンのロジットのみ

        # Top-k サンプリング
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        # 温度パラメータ + サンプリング
        if temperature > 0:
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
        else:
            # 貪欲法
            next_ids = torch.argmax(logits, dim=-1, keepdim=True)

        # 生成したトークンを入力に追加
        ids = torch.cat((ids, next_ids), dim=1)
        yield next_ids.item()
```

**生成の流れ**:
1. 入力トークン列をforward
2. 最後のトークンのロジットを取得
3. Top-kフィルタリング
4. 温度パラメータで調整
5. Softmaxで確率化
6. サンプリングまたは貪欲選択
7. 生成したトークンを入力に追加
8. 繰り返し

---

## 重みの初期化

適切な重みの初期化は、訓練の成功に不可欠です。

### init_weights（gpt.py:175-186）

```python
def init_weights(self):
    self.apply(self._init_weights)

    # 出力層を0で初期化
    torch.nn.init.zeros_(self.lm_head.weight)

    # 各ブロックの出力射影を0で初期化
    for block in self.transformer.h:
        torch.nn.init.zeros_(block.mlp.c_proj.weight)
        torch.nn.init.zeros_(block.attn.c_proj.weight)

    # Rotary Embeddingsを再計算
    head_dim = self.config.n_embd // self.config.n_head
    cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
    self.cos, self.sin = cos, sin
```

### _init_weights（gpt.py:188-198）

```python
def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        # https://arxiv.org/pdf/2310.17813
        fan_out = module.weight.size(0)
        fan_in = module.weight.size(1)
        std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)
```

**初期化戦略**:

1. **線形層**: カスタム正規分布
   - `std = 1/√fan_in × min(1, √(fan_out/fan_in))`
   - fan_inとfan_outのバランスを考慮

2. **埋め込み層**: 標準正規分布（std=1.0）

3. **出力射影**: ゼロ初期化
   - `c_proj`, `lm_head`をゼロで初期化
   - 訓練初期の安定性向上

**なぜゼロ初期化？**
- 残差接続がある場合、出力射影を0で初期化すると、最初は恒等写像になる
- これにより、訓練の初期段階で安定性が向上

---

## オプティマイザーのセットアップ

nanochatは、異なるパラメータタイプに異なる最適化手法を使用します（gpt.py:228-257）。

### setup_optimizers（gpt.py:228-257）

```python
def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
    # パラメータを3グループに分離
    matrix_params = list(self.transformer.h.parameters())        # Transformerブロック
    embedding_params = list(self.transformer.wte.parameters())   # トークン埋め込み
    lm_head_params = list(self.lm_head.parameters())             # 出力層

    # モデル次元に応じたLRスケーリング
    model_dim = self.config.n_embd
    dmodel_lr_scale = (model_dim / 768) ** -0.5

    # AdamWオプティマイザー（埋め込みとlm_head用）
    adam_groups = [
        dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
        dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
    ]
    adamw_optimizer = DistAdamW(adam_groups, betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)

    # Muonオプティマイザー（線形層用）
    muon_optimizer = DistMuon(matrix_params, lr=matrix_lr, momentum=0.95)

    # 2つのオプティマイザーを返す
    return [adamw_optimizer, muon_optimizer]
```

### パラメータグループ

1. **行列パラメータ（Transformerブロック）**:
   - **最適化**: Muon
   - **学習率**: 0.02
   - **対象**: Attention、MLPの線形層

2. **埋め込みパラメータ**:
   - **最適化**: AdamW
   - **学習率**: 0.2 × dmodel_lr_scale
   - **対象**: トークン埋め込み

3. **出力層パラメータ**:
   - **最適化**: AdamW
   - **学習率**: 0.004 × dmodel_lr_scale
   - **対象**: lm_head

### dmodel_lr_scale

```python
dmodel_lr_scale = (model_dim / 768) ** -0.5
```

**目的**: モデル次元に応じて学習率を調整

**例**:
```
model_dim = 768:  dmodel_lr_scale = 1.0
model_dim = 1536: dmodel_lr_scale ≈ 0.707
model_dim = 3072: dmodel_lr_scale = 0.5
```

大きいモデルほど学習率を下げることで、訓練の安定性を保ちます。

**詳細は**: [第6章: 最適化手法](05-optimization.md)（次章以降）

---

## まとめ：モデル実装の重要ポイント

### 1. GPTConfig
- モデルのアーキテクチャを定義
- 主要パラメータ: n_layer, n_embd, n_head, sequence_len

### 2. RMSNorm
- パラメータなしのシンプルな正規化
- LayerNormより高速で効果的

### 3. Rotary Embeddings
- 相対位置を回転変換で表現
- QueryとKeyに適用

### 4. CausalSelfAttention
- Q, K, Vの計算
- QK Normalization
- Multi-Query Attention
- Causal Masking

### 5. MLP
- 4倍に拡張 → ReLU² → 元に戻す
- 非線形変換による表現力向上

### 6. Block
- Pre-Normalization構造
- 残差接続

### 7. GPT
- すべてのコンポーネントを統合
- Logits Softcapで数値安定性
- 2種類のオプティマイザー

---

## 次章への導入

第4章では、GPTモデルの詳細実装を学びました。

### これまでに学んだこと
- 各コンポーネントの実装レベルの理解
- RMSNorm、RoPE、MQAなどの最新技術
- 重みの初期化戦略
- 複数のオプティマイザーの使い分け

### 次章で学ぶこと

**第5章: データパイプラインと訓練プロセス**
- データセットの管理（dataset.py）
- データローダーの実装（dataloader.py）
- 訓練ループの詳細
- 損失計算と評価
- 分散訓練（DDP）

モデルの実装を理解したので、次はそのモデルをどうやって訓練するかを学びます。データの準備から訓練ループの実行まで、訓練パイプライン全体を詳しく見ていきます。

---

**参照ドキュメント**:
- [nanochat/gpt.py](/Users/edom18/MyDesktop/PythonProjects/nanochat/nanochat/gpt.py:1) - GPTモデルの実装全体

**関連する数学ドキュメント**:
- [Attention機構の数式](../doc/math/07-attention-mechanism.md)（作成予定）
- [正規化手法（RMSNorm）](../doc/math/08-layer-normalization.md)（作成予定）
- [位置エンコーディング（RoPE）](../doc/math/09-positional-encoding.md)（作成予定）
- [活性化関数（ReLU²）](../doc/math/11-activation-functions.md)（作成予定）

---

**前へ**: [第3章: トークナイザーとBPE](03-tokenizer.md)
**次へ**: [第5章: データパイプラインと訓練プロセス](05-training-pipeline.md)
**戻る**: [ドキュメント作成計画](../todo/documentation-plan.md)
