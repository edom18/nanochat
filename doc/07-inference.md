# 第7章: 推論エンジンとサンプリング

## 目次
- [7.1 推論とは](#71-推論とは)
- [7.2 ナイーブな推論実装](#72-ナイーブな推論実装)
  - [7.2.1 自己回帰生成の基本](#721-自己回帰生成の基本)
  - [7.2.2 GPTモデルのgenerate関数](#722-gptモデルのgenerate関数)
  - [7.2.3 ナイーブ実装の問題点](#723-ナイーブ実装の問題点)
- [7.3 サンプリング手法](#73-サンプリング手法)
  - [7.3.1 貪欲サンプリング（Greedy Sampling）](#731-貪欲サンプリングgreedy-sampling)
  - [7.3.2 温度パラメータ（Temperature）](#732-温度パラメータtemperature)
  - [7.3.3 Top-kサンプリング](#733-top-kサンプリング)
  - [7.3.4 サンプリング関数の実装](#734-サンプリング関数の実装)
- [7.4 KVキャッシュ](#74-kvキャッシュ)
  - [7.4.1 なぜKVキャッシュが必要か](#741-なぜkvキャッシュが必要か)
  - [7.4.2 KVキャッシュの仕組み](#742-kvキャッシュの仕組み)
  - [7.4.3 KVCacheクラスの実装](#743-kvcacheclassの実装)
  - [7.4.4 動的なキャッシュ拡張](#744-動的なキャッシュ拡張)
- [7.5 Engineクラス：効率的な推論エンジン](#75-engineクラス効率的な推論エンジン)
  - [7.5.1 Engineの設計思想](#751-engineの設計思想)
  - [7.5.2 プリフィルとデコード](#752-プリフィルとデコード)
  - [7.5.3 バッチ生成とKVキャッシュの複製](#753-バッチ生成とkvキャッシュの複製)
  - [7.5.4 ストリーミング生成](#754-ストリーミング生成)
- [7.6 ツール使用（Calculator）](#76-ツール使用calculator)
  - [7.6.1 ツール使用の仕組み](#761-ツール使用の仕組み)
  - [7.6.2 強制トークン挿入](#762-強制トークン挿入)
  - [7.6.3 Calculator実装](#763-calculator実装)
- [7.7 実装の詳細](#77-実装の詳細)
  - [7.7.1 RowStateによる状態管理](#771-rowstateによる状態管理)
  - [7.7.2 生成ループの詳細](#772-生成ループの詳細)
  - [7.7.3 終了条件の処理](#773-終了条件の処理)
- [7.8 使用例](#78-使用例)
- [7.9 まとめ](#79-まとめ)

---

## 7.1 推論とは

**推論（Inference）**は、訓練済みのモデルを使って新しい入力に対する出力を生成するプロセスです。

### 訓練と推論の違い

| 観点 | 訓練（Training） | 推論（Inference） |
|------|-----------------|------------------|
| **目的** | パラメータの最適化 | 出力の生成 |
| **入出力** | 入力と正解ラベルの両方 | 入力のみ |
| **勾配計算** | 必要（backward pass） | 不要 |
| **モード** | `model.train()` | `model.eval()` |
| **ドロップアウト** | 有効 | 無効 |
| **バッチ正規化** | 訓練統計を更新 | 保存された統計を使用 |
| **速度要件** | スループット重視 | レイテンシ重視 |

### 言語モデルの推論

言語モデルの推論は**自己回帰生成（Autoregressive Generation）**で行われます：

```
1. プロンプト（入力テキスト）をトークン化
   例: "The capital of France is" → [464, 3139, 286, 4881, 318]

2. モデルに入力し、次のトークンを予測
   → 次のトークン: "Paris" (トークンID: 6342)

3. 生成されたトークンをシーケンスに追加
   → [464, 3139, 286, 4881, 318, 6342]

4. 2に戻る（終了条件まで繰り返し）
```

この「1トークンずつ生成」するプロセスが自己回帰生成の特徴です。

---

## 7.2 ナイーブな推論実装

### 7.2.1 自己回帰生成の基本

最もシンプルな推論実装は、毎回全シーケンスをモデルに入力する方法です。

```
┌─────────────────────────────────────────────────┐
│         Naive Autoregressive Generation          │
└─────────────────────────────────────────────────┘

Step 1:
  Input: [The, capital, of, France, is]
         ↓ モデル（全シーケンスを処理）
  Output: [logits at each position]
         ↓ 最後のトークンのlogitsからサンプリング
  Sampled: "Paris"

Step 2:
  Input: [The, capital, of, France, is, Paris]
         ↓ モデル（全シーケンスを再び処理 ← 冗長！）
  Output: [logits at each position]
         ↓ 最後のトークンのlogitsからサンプリング
  Sampled: ","

Step 3:
  Input: [The, capital, of, France, is, Paris, ,]
         ↓ モデル（全シーケンスをまた処理 ← さらに冗長！）
  ...
```

### 7.2.2 GPTモデルのgenerate関数

nanochatのGPTモデルには、このナイーブな実装があります。

#### gpt.py:293-322

```python
@torch.inference_mode()
def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
    """
    ナイーブな自己回帰ストリーミング推論
    シンプルにするため：
    - バッチサイズは1
    - tokensとyieldされるtokenはPythonのリストとint
    """
    assert isinstance(tokens, list)
    device = self.get_device()

    # 乱数生成器の初期化
    rng = None
    if temperature > 0:
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

    # トークンをテンソルに変換（バッチ次元を追加）
    ids = torch.tensor([tokens], dtype=torch.long, device=device)  # (1, T)

    # 自己回帰ループ
    for _ in range(max_tokens):
        # 順伝播（全シーケンスを処理）
        logits = self.forward(ids)  # (B, T, vocab_size)
        logits = logits[:, -1, :]   # 最後のトークンのlogitsを取得 (B, vocab_size)

        # Top-kフィルタリング（オプション）
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        # サンプリング
        if temperature > 0:
            logits = logits / temperature  # 温度スケーリング
            probs = F.softmax(logits, dim=-1)
            next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
        else:
            # temperature=0: 貪欲サンプリング
            next_ids = torch.argmax(logits, dim=-1, keepdim=True)

        # 生成されたトークンをシーケンスに追加
        ids = torch.cat((ids, next_ids), dim=1)

        # トークンをyield（ストリーミング）
        token = next_ids.item()
        yield token
```

**ポイント**：
- `@torch.inference_mode()`: 勾配計算を完全に無効化（`no_grad()`より効率的）
- ストリーミング生成: `yield`で1トークンずつ返す
- 毎ステップで`ids`が長くなり、全シーケンスを再計算

### 7.2.3 ナイーブ実装の問題点

#### 計算量の無駄

各ステップで全シーケンスを処理するため、計算量が**二次関数的に増加**します。

```
トークン数 T に対する計算量:
Step 1: T個のトークンを処理
Step 2: T+1個のトークンを処理
Step 3: T+2個のトークンを処理
...
Step N: T+N個のトークンを処理

合計: T + (T+1) + (T+2) + ... + (T+N)
    = N*T + N*(N+1)/2
    ≈ O(N*T + N²)
```

実際には、**以前のステップで計算した中間結果を再利用**できます。これが**KVキャッシュ**の役割です。

---

## 7.3 サンプリング手法

モデルの出力logitsから次のトークンを選ぶ方法を**サンプリング**と呼びます。

### 7.3.1 貪欲サンプリング（Greedy Sampling）

最も確率の高いトークンを常に選択します。

```python
next_token = torch.argmax(logits, dim=-1)
```

**特徴**：
- ✅ 決定的（同じ入力に対して常に同じ出力）
- ✅ シンプル
- ❌ 多様性がない
- ❌ 繰り返しが多い（ループに陥りやすい）

### 7.3.2 温度パラメータ（Temperature）

**温度（Temperature）**は、確率分布の「鋭さ」を制御するパラメータです。

```python
logits = logits / temperature
probs = F.softmax(logits, dim=-1)
```

#### 温度の効果

```
元のlogits: [2.0, 1.0, 0.5]

Temperature = 1.0（デフォルト）:
  probs = softmax([2.0, 1.0, 0.5])
        = [0.506, 0.307, 0.186]
  → バランスの取れた分布

Temperature = 0.5（低温）:
  probs = softmax([4.0, 2.0, 1.0])
        = [0.705, 0.214, 0.079]
  → より鋭い分布（高確率トークンに集中）

Temperature = 2.0（高温）:
  probs = softmax([1.0, 0.5, 0.25])
        = [0.410, 0.336, 0.253]
  → より平坦な分布（多様性が高い）

Temperature = 0.0:
  → 貪欲サンプリングと同じ（argmax）
```

**温度の使い分け**：

| 温度 | 用途 |
|------|------|
| **0.0** | 決定的な出力が必要な場合（コード生成、計算など） |
| **0.3-0.7** | 事実的な質問応答（適度な確実性） |
| **0.8-1.0** | バランスの取れた会話 |
| **1.2-1.5** | 創造的な文章生成（詩、物語など） |

> Softmax関数と温度の数学的詳細は [`doc/math/03-softmax.md`](math/03-softmax.md) を参照してください。

### 7.3.3 Top-kサンプリング

**Top-k**は、確率上位k個のトークンのみを候補として残し、その中からサンプリングします。

```python
# Top-kフィルタリング
k = min(top_k, logits.size(-1))
vals, idx = torch.topk(logits, k, dim=-1)  # 上位k個を取得

# 上位k個のlogitsのみを使用
vals = vals / temperature
probs = F.softmax(vals, dim=-1)

# 上位k個の中からサンプリング
choice = torch.multinomial(probs, num_samples=1, generator=rng)
next_token = idx.gather(1, choice)  # インデックスを元の語彙に戻す
```

#### Top-kの効果

```
語彙サイズ: 50,000
確率分布: [0.3, 0.25, 0.2, 0.1, 0.05, 0.04, 0.03, ..., 0.00001, ...]

Top-k = 5:
  候補: 上位5個のみ [0.3, 0.25, 0.2, 0.1, 0.05]
  正規化: [0.333, 0.278, 0.222, 0.111, 0.056]
  → 低確率のノイズトークンを排除

Top-k = 100:
  候補: 上位100個
  → より多様性が高い
```

**メリット**：
- 低確率の「ノイズ」トークンを排除
- 出力の品質向上
- 典型的な値: k=40〜200

### 7.3.4 サンプリング関数の実装

nanochatの統一されたサンプリング関数: `engine.py:128-144`

```python
@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """
    logits (B, vocab_size)から次のトークンをサンプリング
    戻り値: (B, 1)の形状のトークンID
    """
    assert temperature >= 0.0, "temperature must be non-negative"

    # Temperature = 0.0: 貪欲サンプリング
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    # Top-kサンプリング
    if top_k is not None:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)

    # 通常のサンプリング（Top-kなし）
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng)
```

**設計のポイント**：
- バッチ対応（複数のシーケンスを同時処理）
- 温度=0で貪欲サンプリング
- Top-kの有無で分岐
- 乱数生成器を引数で受け取る（再現性確保）

> 確率的サンプリングの数学的詳細は [`doc/math/12-probability-sampling.md`](math/12-probability-sampling.md) を参照してください。

---

## 7.4 KVキャッシュ

### 7.4.1 なぜKVキャッシュが必要か

Attentionメカニズムでは、各トークンがこれまでの**すべてのトークン**とAttentionを計算します。

```
┌───────────────────────────────────────────────────┐
│          Without KV Cache (Inefficient)            │
└───────────────────────────────────────────────────┘

Step 1: [The, capital, of, France, is]
  → Query, Key, Valueを計算（5トークン分）
  → Attention計算
  → "Paris"を生成

Step 2: [The, capital, of, France, is, Paris]
  → Query, Key, Valueを再計算（6トークン分）← 最初の5トークンは重複！
  → Attention計算
  → ","を生成

Step 3: [The, capital, of, France, is, Paris, ,]
  → Query, Key, Valueを再々計算（7トークン分）← 最初の6トークンは重複！
  ...
```

**問題**：以前のトークンのKey/Valueを毎回再計算している

**解決策**：計算済みのKey/Valueを**キャッシュ**し、新しいトークンのKey/Valueだけを計算

```
┌───────────────────────────────────────────────────┐
│           With KV Cache (Efficient)                │
└───────────────────────────────────────────────────┘

Step 1: [The, capital, of, France, is]
  → Q, K, Vを計算（5トークン分）
  → K, Vをキャッシュに保存
  → "Paris"を生成

Step 2: [Paris]  ← 新しいトークンのみを入力
  → Q, K, Vを計算（1トークンのみ！）
  → キャッシュされたK, Vと結合
  → Attention計算
  → ","を生成

Step 3: [,]  ← 新しいトークンのみ
  → Q, K, Vを計算（1トークンのみ！）
  → キャッシュと結合
  ...
```

### 7.4.2 KVキャッシュの仕組み

#### Attentionでの使用

```python
# KVキャッシュなし
def attention_without_cache(x):
    Q = x @ W_q  # (B, T, D)
    K = x @ W_k  # (B, T, D)
    V = x @ W_v  # (B, T, D)
    return attention(Q, K, V)

# KVキャッシュあり
def attention_with_cache(x, kv_cache, layer_idx):
    Q = x @ W_q          # (B, T_new, D) - 新しいトークンのみ
    K = x @ W_k          # (B, T_new, D)
    V = x @ W_v          # (B, T_new, D)

    # キャッシュに挿入し、累積されたK, Vを取得
    K_full, V_full = kv_cache.insert_kv(layer_idx, K, V)
    # K_full: (B, T_total, D) - これまでのすべてのKey
    # V_full: (B, T_total, D) - これまでのすべてのValue

    return attention(Q, K_full, V_full)
```

#### キャッシュの構造

```
KVキャッシュの形状: (num_layers, 2, B, H, T_max, D)

次元の意味:
- num_layers: Transformerの層数（各層ごとに独立したキャッシュ）
- 2: Key (0) と Value (1)
- B: バッチサイズ
- H: ヘッド数
- T_max: 最大シーケンス長（動的に拡張可能）
- D: ヘッド次元
```

### 7.4.3 KVCacheクラスの実装

#### 初期化: `engine.py:56-72`

```python
class KVCache:
    """
    GPTモデルと連携してKVキャッシュを管理
    注: .posは最後のTransformer層が挿入後に自動的に進む
    """

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers):
        # K/Vそれぞれの形状: (B, H, T, D)、各Transformer層ごとに1つ
        self.kv_shape = (num_layers, 2, batch_size, num_heads, seq_len, head_dim)
        self.kv_cache = None  # 遅延初期化（dtype/deviceが必要なため）
        self.pos = 0  # キャッシュ内の現在位置

    def reset(self):
        """キャッシュをリセット"""
        self.pos = 0

    def get_pos(self):
        """現在のキャッシュ位置を取得"""
        return self.pos
```

#### KVの挿入: `engine.py:101-124`

```python
def insert_kv(self, layer_idx, k, v):
    """
    新しいKey/Valueをキャッシュに挿入し、累積されたキャッシュを返す
    """
    # 遅延初期化（最初の挿入時にdtype/deviceを検出）
    if self.kv_cache is None:
        self.kv_cache = torch.empty(self.kv_shape, dtype=k.dtype, device=k.device)

    # 挿入する新しいKey/Valueの形状
    B, H, T_add, D = k.size()

    # キャッシュの範囲: [pos, pos + T_add)
    t0, t1 = self.pos, self.pos + T_add

    # 動的にキャッシュを拡張（必要に応じて）
    if t1 > self.kv_cache.size(4):
        t_needed = t1 + 1024  # 必要な分 + 1024のバッファ
        t_needed = (t_needed + 1023) & ~1023  # 1024の倍数に切り上げ
        current_shape = list(self.kv_cache.shape)
        current_shape[4] = t_needed
        self.kv_cache.resize_(current_shape)

    # K, Vをキャッシュに挿入
    self.kv_cache[layer_idx, 0, :, :, t0:t1] = k
    self.kv_cache[layer_idx, 1, :, :, t0:t1] = v

    # 現在位置までの累積キャッシュを返す（ビューとして）
    key_view = self.kv_cache[layer_idx, 0, :, :, :t1]
    value_view = self.kv_cache[layer_idx, 1, :, :, :t1]

    # 最後の層の処理後にposを進める
    if layer_idx == self.kv_cache.size(0) - 1:
        self.pos = t1

    return key_view, value_view
```

**重要な仕組み**：
1. **遅延初期化**: 最初の挿入時にdtype/deviceを検出
2. **動的拡張**: シーケンスが長くなればキャッシュも拡張
3. **ビューを返す**: コピーではなくビューで効率的
4. **自動位置更新**: 最後の層が処理したら`pos`を進める

#### プリフィル: `engine.py:74-99`

```python
def prefill(self, other):
    """
    別のKVキャッシュでプリフィル
    バッチ次元を拡張可能（バッチ1でプリフィル後、複数サンプル生成に使用）
    """
    # 形状の検証
    assert self.kv_cache is None, "Cannot prefill a non-empty KV cache"
    assert other.kv_cache is not None, "Cannot prefill with a None KV cache"

    for ix, (dim1, dim2) in enumerate(zip(self.kv_shape, other.kv_shape)):
        if ix in [0, 1, 3, 5]:
            # num_layers, K/V, num_heads, head_dim は一致必須
            assert dim1 == dim2
        elif ix == 2:
            # batch_size は拡張可能（dim2=1 → dim1=N）
            assert dim1 == dim2 or dim2 == 1
        elif ix == 4:
            # seq_len: selfの方が長い必要あり
            assert dim1 >= dim2

    # キャッシュを初期化
    dtype, device = other.kv_cache.dtype, other.kv_cache.device
    self.kv_cache = torch.empty(self.kv_shape, dtype=dtype, device=device)

    # データをコピー（ブロードキャストされる）
    self.kv_cache[:, :, :, :, :other.pos, :] = other.kv_cache

    # 位置を更新
    self.pos = other.pos
```

**用途**：
- バッチ1でプロンプトをプリフィル
- そのキャッシュを複製して、複数のサンプルを並列生成

### 7.4.4 動的なキャッシュ拡張

```python
# 動的拡張のロジック（engine.py:109-114）
if t1 > self.kv_cache.size(4):
    t_needed = t1 + 1024              # 必要な分 + バッファ1024
    t_needed = (t_needed + 1023) & ~1023  # 1024の倍数に切り上げ
    current_shape = list(self.kv_cache.shape)
    current_shape[4] = t_needed
    self.kv_cache.resize_(current_shape)
```

**なぜ1024の倍数？**
- メモリアライメントの最適化
- 頻繁なリサイズを避ける（バッファを確保）
- GPUメモリの効率的な使用

**計算量の削減効果**：
```
シーケンス長Nのとき:

KVキャッシュなし: O(N²)
  各ステップで全シーケンスを再計算

KVキャッシュあり: O(N)
  各ステップで1トークンのみ計算
```

---

## 7.5 Engineクラス：効率的な推論エンジン

### 7.5.1 Engineの設計思想

`Engine`クラスは、KVキャッシュを活用した高速な推論を提供します。

**設計原則**：
1. **トークンIDのみを扱う**（トークナイゼーションとは分離）
2. **バッチ生成対応**（複数サンプルを並列生成）
3. **ストリーミング生成**（1トークンずつyield）
4. **ツール使用のサポート**（Calculator等）

#### engine.py:157-162

```python
class Engine:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer  # ツール使用に必要
```

### 7.5.2 プリフィルとデコード

推論は**2つのフェーズ**に分かれます：

```
┌──────────────────────────────────────────────────┐
│           Prefill Phase (プリフィル)               │
└──────────────────────────────────────────────────┘

入力プロンプト: [The, capital, of, France, is]
     ↓
モデルに一括入力（バッチサイズ1）
     ↓
全トークンのKey/Valueを計算してキャッシュ
     ↓
最後のトークンのlogitsから最初のトークンをサンプリング
     ↓
サンプル: "Paris"

┌──────────────────────────────────────────────────┐
│           Decode Phase (デコード)                  │
└──────────────────────────────────────────────────┘

Step 1: ["Paris"]  ← 1トークンのみ入力
     ↓
新しいトークンのK/Vを計算し、キャッシュに追加
     ↓
キャッシュと結合してAttention
     ↓
次のトークンをサンプリング: ","

Step 2: [","]  ← 1トークンのみ
     ↓
（繰り返し）
```

#### プリフィルの実装: `engine.py:180-192`

```python
# [1] バッチ1でプロンプトトークンをプリフィル
m = self.model.config
kv_model_kwargs = {
    "num_heads": m.n_kv_head,
    "head_dim": m.n_embd // m.n_head,
    "num_layers": m.n_layer
}

# プリフィル用のKVキャッシュ（バッチサイズ1）
kv_cache_prefill = KVCache(
    batch_size=1,
    seq_len=len(tokens),
    **kv_model_kwargs,
)

# プロンプト全体を一括処理
ids = torch.tensor([tokens], dtype=torch.long, device=device)
logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
logits = logits[:, -1, :]  # 最後のトークンのlogitsのみ

# 最初のトークンをサンプリング
next_ids = sample_next_token(logits, rng, temperature, top_k)
sampled_tokens = next_ids[:, 0].tolist()
```

### 7.5.3 バッチ生成とKVキャッシュの複製

複数のサンプルを並列生成するため、プリフィルで得たキャッシュを**複製**します。

#### engine.py:194-202

```python
# [2] KVキャッシュを各サンプル/行ごとに複製
kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else self.model.config.sequence_len

kv_cache_decode = KVCache(
    batch_size=num_samples,  # バッチサイズを拡張
    seq_len=kv_length_hint,
    **kv_model_kwargs,
)

# プリフィルキャッシュでデコードキャッシュを初期化
kv_cache_decode.prefill(kv_cache_prefill)
del kv_cache_prefill  # メモリ解放
```

**仕組み**：
```
プリフィルキャッシュ (B=1):
  shape: (L, 2, 1, H, T, D)

↓ prefill()

デコードキャッシュ (B=num_samples):
  shape: (L, 2, num_samples, H, T, D)
  → バッチ次元がブロードキャストで拡張される
```

これにより、**1回のプリフィルで複数サンプルを効率的に生成**できます。

### 7.5.4 ストリーミング生成

`Engine.generate()`はPythonジェネレータとして実装されています。

#### engine.py:208-268

```python
# [3] 各サンプルの状態を初期化
row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

# [4] メイン生成ループ
num_generated = 0
first_iteration = True

while True:
    # 終了条件: 最大トークン数に到達
    if max_tokens is not None and num_generated >= max_tokens:
        break
    # 終了条件: 全行が完了
    if all(state.completed for state in row_states):
        break

    # サンプリングされたトークンを取得
    if first_iteration:
        # プリフィルで既にサンプリング済み
        sampled_tokens = [sampled_tokens[0]] * num_samples  # 全行にブロードキャスト
        first_iteration = False
    else:
        # モデルを順伝播して次のトークンを取得
        logits = self.model.forward(ids, kv_cache=kv_cache_decode)  # (B, T, vocab_size)
        logits = logits[:, -1, :]  # 最後のステップのlogits (B, vocab_size)
        next_ids = sample_next_token(logits, rng, temperature, top_k)  # (B, 1)
        sampled_tokens = next_ids[:, 0].tolist()

    # 各行を処理: 次のトークンを選択、状態を更新
    token_column = []  # 各行の次のトークンID
    token_masks = []   # サンプリング(1)か強制(0)か

    for i, state in enumerate(row_states):
        # 強制トークンがあればそれを使用、なければサンプリング結果
        is_forced = len(state.forced_tokens) > 0
        token_masks.append(0 if is_forced else 1)
        next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
        token_column.append(next_token)

        # 状態を更新
        state.current_tokens.append(next_token)

        # 終了トークンのチェック
        if next_token == assistant_end or next_token == bos:
            state.completed = True

        # ツールロジック（後述）
        # ...

    # トークンカラムをyield（ストリーミング）
    yield token_column, token_masks
    num_generated += 1

    # 次のイテレーション用のIDを準備
    ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)
```

**ストリーミングの利点**：
- ユーザーに即座にフィードバック（トークンごとに表示）
- メモリ効率（全シーケンスを保持不要）
- 早期停止が可能

---

## 7.6 ツール使用（Calculator）

### 7.6.1 ツール使用の仕組み

nanochatは、生成中に**外部ツール（Calculator）**を呼び出す機能を持ちます。

```
ユーザー: "What is 123 * 456?"

モデル生成:
  "Let me calculate that.
   <|python_start|>123*456<|python_end|>"

↓ Calculator呼び出し

強制挿入:
  "<|output_start|>56088<|output_end|>"

最終出力:
  "Let me calculate that.
   <|python_start|>123*456<|python_end|>
   <|output_start|>56088<|output_end|>
   The answer is 56,088."
```

#### 特殊トークン

| トークン | 意味 |
|---------|------|
| `<|python_start|>` | Python式の開始 |
| `<|python_end|>` | Python式の終了 |
| `<|output_start|>` | ツール出力の開始 |
| `<|output_end|>` | ツール出力の終了 |

### 7.6.2 強制トークン挿入

モデルがサンプリングするのではなく、システムが**強制的にトークンを挿入**します。

#### RowStateによる管理: `engine.py:148-155`

```python
class RowState:
    """生成中の各行の状態を追跡"""
    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or []  # 現在のトークンシーケンス
        self.forced_tokens = deque()  # 強制挿入するトークンのキュー
        self.in_python_block = False  # Python式ブロック内か
        self.python_expr_tokens = []  # 現在のPython式のトークン
        self.completed = False  # この行の生成が完了したか
```

#### 強制挿入の処理: `engine.py:236-261`

```python
for i, state in enumerate(row_states):
    # 強制トークンがあればそれを使用、なければサンプリング
    is_forced = len(state.forced_tokens) > 0
    token_masks.append(0 if is_forced else 1)  # マスク: 0=強制, 1=サンプリング
    next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
    token_column.append(next_token)

    # 状態を更新
    state.current_tokens.append(next_token)

    # 終了トークンのチェック
    if next_token == assistant_end or next_token == bos:
        state.completed = True

    # ツールロジック
    if next_token == python_start:
        # Python式ブロック開始
        state.in_python_block = True
        state.python_expr_tokens = []

    elif next_token == python_end and state.in_python_block:
        # Python式ブロック終了 → Calculatorを呼び出し
        state.in_python_block = False
        if state.python_expr_tokens:
            expr = self.tokenizer.decode(state.python_expr_tokens)
            result = use_calculator(expr)
            if result is not None:
                # 結果を強制トークンキューに追加
                result_tokens = self.tokenizer.encode(str(result))
                state.forced_tokens.append(output_start)
                state.forced_tokens.extend(result_tokens)
                state.forced_tokens.append(output_end)
        state.python_expr_tokens = []

    elif state.in_python_block:
        # Python式ブロック内 → トークンを収集
        state.python_expr_tokens.append(next_token)
```

### 7.6.3 Calculator実装

#### engine.py:35-53

```python
def eval_with_timeout(formula, max_time=3):
    """タイムアウト付きで数式を評価"""
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return eval(formula)
    except Exception as e:
        signal.alarm(0)
        return None

def use_calculator(expr):
    """数式を安全に評価"""
    expr = expr.replace(",", "")  # カンマを削除

    # 許可された文字のみ（数字と基本演算子）
    if any([x not in "0123456789*+-/.() " for x in expr]):
        return None

    # べき乗演算子は禁止（計算コストが高い可能性）
    if "**" in expr:
        return None

    return eval_with_timeout(expr)
```

**安全性の考慮**：
- タイムアウト（3秒）
- 許可された文字のみ
- べき乗演算子の禁止
- 例外処理

---

## 7.7 実装の詳細

### 7.7.1 RowStateによる状態管理

各サンプル（行）は独立した状態を持ちます：

```python
row_states = [RowState(tokens.copy()) for _ in range(num_samples)]
```

**なぜ行ごとの状態が必要？**
- 各サンプルが異なる長さで終了する可能性
- ツール使用の状態が独立
- 強制トークンのキューが独立

### 7.7.2 生成ループの詳細

```python
while True:
    # [1] 終了条件のチェック
    if max_tokens is not None and num_generated >= max_tokens:
        break
    if all(state.completed for state in row_states):
        break

    # [2] サンプリング（初回以外）
    if not first_iteration:
        logits = self.model.forward(ids, kv_cache=kv_cache_decode)
        logits = logits[:, -1, :]
        next_ids = sample_next_token(logits, rng, temperature, top_k)
        sampled_tokens = next_ids[:, 0].tolist()

    # [3] 各行の処理
    token_column = []
    token_masks = []
    for i, state in enumerate(row_states):
        # トークン選択（サンプリング or 強制）
        is_forced = len(state.forced_tokens) > 0
        next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]

        # 状態更新
        state.current_tokens.append(next_token)

        # ツールロジック & 終了チェック
        # ...

        token_column.append(next_token)
        token_masks.append(0 if is_forced else 1)

    # [4] Yield（ストリーミング）
    yield token_column, token_masks

    # [5] 次のイテレーションの準備
    ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)
    num_generated += 1
```

### 7.7.3 終了条件の処理

#### 2つの終了条件

1. **最大トークン数に到達**
   ```python
   if max_tokens is not None and num_generated >= max_tokens:
       break
   ```

2. **全行が完了**
   ```python
   if all(state.completed for state in row_states):
       break
   ```

#### 行の完了判定

```python
# 終了トークンで完了マーク
if next_token == assistant_end or next_token == bos:
    state.completed = True
```

#### generate_batch関数

ストリーミングではなく、**完全なシーケンスを返す**バージョン：

#### engine.py:269-291

```python
def generate_batch(self, tokens, num_samples=1, **kwargs):
    """
    非ストリーミングのバッチ生成
    最終的なトークンシーケンスのリストを返す
    終了トークン（assistant_end, bos）は含まれない
    """
    assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
    bos = self.tokenizer.get_bos_token_id()

    results = [tokens.copy() for _ in range(num_samples)]
    masks = [[0] * len(tokens) for _ in range(num_samples)]
    completed = [False] * num_samples

    for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
        for i, (token, mask) in enumerate(zip(token_column, token_masks)):
            if not completed[i]:
                if token == assistant_end or token == bos:
                    completed[i] = True
                else:
                    results[i].append(token)
                    masks[i].append(mask)
        # 全行完了で早期終了
        if all(completed):
            break

    return results, masks
```

**戻り値**：
- `results`: 各サンプルのトークンシーケンス（終了トークンなし）
- `masks`: 各トークンがサンプリング(1)か強制(0)か

---

## 7.8 使用例

### 基本的な使用

#### scripts/base_train.py:222-228

```python
from nanochat.engine import Engine

# Engineの初期化
engine = Engine(model, tokenizer)

# プロンプトのトークン化
prompt = "The capital of France is"
tokens = tokenizer(prompt, prepend="<|bos|>")

# 生成（ストリーミング）
for token_column, token_masks in engine.generate(
    tokens,
    num_samples=1,
    max_tokens=16,
    temperature=0
):
    token = token_column[0]  # 最初の行のトークン
    chunk = tokenizer.decode([token])
    print(chunk, end="", flush=True)
```

### バッチ生成

```python
# 複数サンプルを並列生成
results, masks = engine.generate_batch(
    tokens,
    num_samples=5,
    max_tokens=32,
    temperature=0.8,
    top_k=40,
    seed=42
)

# 各サンプルを表示
for i, result in enumerate(results):
    text = tokenizer.decode(result)
    print(f"Sample {i+1}: {text}")
```

### 性能比較

#### engine.py:294-343（テストコード）

```python
# ナイーブな実装（model.generate）
t0 = time.time()
for token in model.generate(prompt_tokens, max_tokens=64, temperature=0):
    print(tokenizer.decode([token]), end="", flush=True)
t1 = time.time()
print(f"Reference time: {t1 - t0:.2f}s")

# Engineによる実装（KVキャッシュ使用）
engine = Engine(model, tokenizer)
t0 = time.time()
for token_column, token_masks in engine.generate(prompt_tokens, num_samples=1, max_tokens=64, temperature=0):
    print(tokenizer.decode([token_column[0]]), end="", flush=True)
t1 = time.time()
print(f"Engine time: {t1 - t0:.2f}s")
```

**期待される性能向上**：
- ナイーブ実装: O(N²)の計算量
- Engine: O(N)の計算量
- **数倍〜数十倍の高速化**（シーケンス長に依存）

---

## 7.9 まとめ

この章では、nanochatの推論エンジンとサンプリング手法について学びました。

### 主要な概念

1. **推論の基本**
   - 自己回帰生成: 1トークンずつ生成
   - 訓練モードとの違い（勾配計算なし、決定的な動作）

2. **サンプリング手法**
   - **貪欲サンプリング**: 最高確率のトークンを選択
   - **温度**: 確率分布の鋭さを制御（0=決定的、高=多様）
   - **Top-k**: 上位k個の候補から選択

3. **KVキャッシュ**
   - 計算済みのKey/Valueを再利用
   - O(N²) → O(N)の計算量削減
   - 動的にキャッシュを拡張

4. **Engineクラス**
   - **プリフィル**: プロンプトを一括処理
   - **デコード**: 1トークンずつ生成
   - **バッチ生成**: KVキャッシュを複製して複数サンプルを並列生成
   - **ストリーミング**: 1トークンずつyield

5. **ツール使用**
   - Calculator: 数式を評価
   - 強制トークン挿入: ツール出力をモデル生成に統合
   - RowStateで各サンプルの状態を管理

6. **実装の工夫**
   - 遅延初期化（dtype/device検出）
   - 動的キャッシュ拡張（1024の倍数）
   - 安全なツール評価（タイムアウト、文字制限）

### コード参照

| 概念 | ファイル | 行番号 |
|------|---------|--------|
| ナイーブな推論 | `nanochat/gpt.py` | 293-322 |
| サンプリング関数 | `nanochat/engine.py` | 128-144 |
| KVCacheクラス | `nanochat/engine.py` | 56-124 |
| Engineクラス | `nanochat/engine.py` | 157-291 |
| Calculator | `nanochat/engine.py` | 35-53 |
| ツールロジック | `nanochat/engine.py` | 246-261 |

### パフォーマンス

```
┌─────────────────────────────────────────────────┐
│        Naive vs Engine Performance               │
└─────────────────────────────────────────────────┘

シーケンス長 N に対する計算量:

Naive (model.generate):
  各ステップで全シーケンスを再計算
  計算量: O(N²)

Engine (with KV Cache):
  各ステップで1トークンのみ計算
  計算量: O(N)

速度向上: 数倍〜数十倍（N=100で約10倍、N=1000で約100倍）
```

### 次のステップ

第8章では、モデルの**評価とベンチマーク**について学びます：
- COREメトリクス
- タスク別評価（ARC, GSM8K, MMLU, HumanEval）
- Bits per Byte評価
- 評価スクリプトの使い方

サンプリングと確率の数学的詳細については、以下のドキュメントを参照してください：
- [`doc/math/03-softmax.md`](math/03-softmax.md): Softmax関数
- [`doc/math/12-probability-sampling.md`](math/12-probability-sampling.md): 確率的サンプリング

---

**前の章**: [第6章: 最適化手法（Muon, AdamW）](06-optimization.md)
**次の章**: [第8章: 評価とベンチマーク](08-evaluation.md)
**目次に戻る**: [ドキュメントTOP](01-project-overview.md)
