# 第2章: Transformerアーキテクチャの基礎

## 目次
1. [Transformerとは](#transformerとは)
2. [Encoder-DecoderからDecoder-onlyへ](#encoder-decoderからdecoder-onlyへ)
3. [GPTモデルの基本構造](#gptモデルの基本構造)
4. [Self-Attention機構の概要](#self-attention機構の概要)
5. [位置エンコーディング](#位置エンコーディング)
6. [nanochatのTransformerの特徴](#nanochatのtransformerの特徴)
7. [次章への導入](#次章への導入)

---

## Transformerとは

**Transformer**は、2017年にGoogleが発表した「Attention is All You Need」という論文で提案された、ニューラルネットワークのアーキテクチャです。

### なぜTransformerが重要なのか

それまでの自然言語処理は、主に**RNN（Recurrent Neural Network、再帰型ニューラルネットワーク）**や**LSTM（Long Short-Term Memory）**という仕組みが使われていました。これらは、文章を1単語ずつ順番に処理していく方式です。

**RNN/LSTMの問題点**:
- **逐次処理**: 前の単語を処理しないと次の単語を処理できない（並列化が困難）
- **長期依存の難しさ**: 長い文章になると、最初の方の情報を忘れてしまう

**Transformerの革新**:
- **並列処理**: すべての単語を同時に処理できる
- **Attention機構**: 文章の中で重要な部分に「注目」できる
- **長距離依存の解決**: 文章の最初と最後の関係も捉えられる

この革新により、Transformerは自然言語処理における**デファクトスタンダード**となり、GPT、BERT、ChatGPTなど、現代のすべての大規模言語モデルの基盤技術となっています。

---

## Encoder-DecoderからDecoder-onlyへ

Transformerには、いくつかのバリエーションがあります。

### 1. Encoder-Decoder Transformer（オリジナル）

元々のTransformerは、**Encoder-Decoder**構造でした。

```
入力文 → [Encoder] → 内部表現 → [Decoder] → 出力文
```

**用途**: 翻訳（英語 → 日本語など）

- **Encoder**: 入力文を理解して内部表現に変換
- **Decoder**: 内部表現を元に出力文を生成

**例**: Google翻訳などの機械翻訳

### 2. Encoder-only Transformer

Encoderだけを使うモデル。

```
入力文 → [Encoder] → 分類/抽出
```

**用途**: 文章分類、感情分析、固有表現抽出

**代表例**: BERT（Bidirectional Encoder Representations from Transformers）

- 双方向（文章全体を見渡せる）
- 入力文の「理解」に特化

### 3. Decoder-only Transformer（GPT）

**Decoderだけを使うモデル**。これがGPTシリーズ（およびnanochat）が採用している構造です。

```
プロンプト → [Decoder] → 次のトークン予測 → 生成文
```

**用途**: 文章生成、対話、コード生成

**代表例**: GPT-2, GPT-3, GPT-4, ChatGPT

- **Causal（因果的）**: 過去のトークンだけを見て、次を予測
- **自己回帰（Autoregressive）**: 生成したトークンを入力に戻して、また次を生成

### なぜDecoder-onlyなのか？

nanochatがDecoder-onlyを採用している理由:

1. **シンプルさ**: Encoder-Decoderより構造が単純
2. **汎用性**: 1つのモデルでテキスト生成、対話、質問応答など多様なタスクに対応
3. **スケーラビリティ**: パラメータを増やすほど性能が向上しやすい
4. **事前学習の効率**: 「次のトークン予測」という単純なタスクで学習可能

---

## GPTモデルの基本構造

nanochatのGPTモデルは、以下の階層構造で構成されています。

### 全体構造

```
入力トークンID
    ↓
[Token Embedding] ← 各トークンIDを高次元ベクトルに変換
    ↓
[Normalization] ← RMSNormで正規化
    ↓
[Transformer Block 1]
    ├─ Self-Attention
    └─ MLP（フィードフォワード）
    ↓
[Transformer Block 2]
    ├─ Self-Attention
    └─ MLP
    ↓
    ⋮  （depth = 12〜32層）
    ↓
[Transformer Block N]
    ├─ Self-Attention
    └─ MLP
    ↓
[Normalization] ← 最終正規化
    ↓
[lm_head] ← 語彙サイズの次元に射影
    ↓
Logits（各トークンの生成確率）
    ↓
[Softmax] ← 確率分布に変換
    ↓
次のトークン予測
```

### GPTクラスの構成（gpt.py:154）

```python
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),  # トークン埋め込み
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),  # Transformerブロック
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # 出力層
```

**主要コンポーネント**:

1. **`wte`（Token Embedding）**: トークンIDを`n_embd`次元のベクトルに変換
2. **`h`（Transformer Blocks）**: `n_layer`個のTransformerブロック
3. **`lm_head`（Language Model Head）**: 最終的に語彙サイズの次元に射影

### 設定パラメータ（GPTConfig, gpt.py:26）

```python
@dataclass
class GPTConfig:
    sequence_len: int = 1024       # 最大シーケンス長
    vocab_size: int = 50304        # 語彙サイズ
    n_layer: int = 12              # Transformerブロックの数（深さ）
    n_head: int = 6                # Queryヘッド数
    n_kv_head: int = 6             # Key/Valueヘッド数（Multi-Query Attention用）
    n_embd: int = 768              # 埋め込み次元（モデルの幅）
```

**nanochatの実際のモデル**:
- **d20モデル**（speedrun.sh）: `n_layer=20`, 561Mパラメータ
- **d32モデル**（より大規模）: `n_layer=32`, 1.9Bパラメータ

---

## Self-Attention機構の概要

**Self-Attention（自己注意機構）**は、Transformerの中核をなす仕組みです。

### Attentionとは何か

人間が文章を読むとき、重要な単語に「注目」します。例えば:

> 「彼女は**図書館**で本を読んだ。そこは静かだった。」

この文で「そこ」が何を指すかを理解するには、前の「図書館」に注目する必要があります。

**Attention機構**は、この「注目」をニューラルネットワークで実現する仕組みです。

### Self-Attentionの3つの要素（Q, K, V）

Self-Attentionでは、各トークンを3つの異なる役割に変換します:

1. **Query（クエリ、質問）**: 「私は誰に注目すべきか？」
2. **Key（キー、鍵）**: 「私はこういう情報を持っています」
3. **Value（バリュー、値）**: 「私の実際の内容はこれです」

**概念的な流れ**:
```
各トークン
    ↓
┌─────────┬─────────┬─────────┐
│ Query   │ Key     │ Value   │  ← 線形変換で生成
└─────────┴─────────┴─────────┘
    ↓         ↓         ↓
Query と Key の内積 → Attention重み（どのトークンに注目するか）
    ↓
Attention重み × Value → 文脈を考慮した新しい表現
```

### Causal Self-Attention（因果的自己注意）

GPTでは、**Causal Attention**を使います。

**重要なルール**: 未来のトークンは見てはいけない

```
入力: "彼女は図書館で"

トークン1「彼女」: トークン1だけを見る
トークン2「は」  : トークン1, 2を見る
トークン3「図書館」: トークン1, 2, 3を見る
トークン4「で」  : トークン1, 2, 3, 4を見る
```

これを**Causal Masking**（因果的マスキング）で実現します。未来のトークンへのAttentionをマスク（隠す）ことで、訓練時に「次のトークン予測」が正しく機能します。

### nanochatの実装（gpt.py:64）

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)  # Query射影
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)  # Key射影
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)  # Value射影
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)  # 出力射影
```

**Attentionの流れ（gpt.py:79-126）**:
1. 入力`x`をQuery、Key、Valueに線形変換
2. Rotary Embeddingsを適用（位置情報の付与）
3. QK Normalizationで正規化
4. Scaled Dot-Product Attentionを計算
5. 出力を`c_proj`で元の次元に戻す

**数学的詳細は**: [doc/math/07-attention-mechanism.md](../doc/math/07-attention-mechanism.md)（作成予定）

---

## 位置エンコーディング

Transformerは並列処理を行うため、単語の**順序情報**を持ちません。

### なぜ位置情報が必要か

例えば、以下の2つの文:
1. 「犬が猫を追いかけた」
2. 「猫が犬を追いかけた」

単語は同じでも、順序が違えば意味が変わります。Transformerはこの順序を理解するために、**位置エンコーディング**が必要です。

### 2つのアプローチ

#### 1. 絶対位置エンコーディング（Absolute Positional Encoding）
各トークンの**絶対的な位置**（1番目、2番目、...）を埋め込む。

**例**: オリジナルのTransformer、BERT

#### 2. 相対位置エンコーディング（Relative Positional Encoding）
トークン間の**相対的な距離**を埋め込む。

**例**: Rotary Embeddings（RoPE）← nanochatが採用

### Rotary Embeddings（RoPE）

nanochatは、**Rotary Embeddings（RoPE、回転式位置エンコーディング）**を使用します。

**特徴**:
- QueryとKeyに回転変換を適用
- 相対位置を内積で表現
- 長い文章でも性能劣化が少ない

**概念**:
```
トークンの位置に応じて、Query/Keyベクトルを「回転」させる

位置0: 回転なし
位置1: 少し回転
位置2: もう少し回転
  ⋮
```

この「回転」により、トークン間の相対的な距離が内積に反映されます。

**実装（gpt.py:41-49）**:
```python
def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]  # ベクトルを2つに分割
    y1 = x1 * cos + x2 * sin         # 回転（複素数の回転と等価）
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3)     # 再結合
    return out
```

**数学的詳細は**: [doc/math/09-positional-encoding.md](../doc/math/09-positional-encoding.md)（作成予定）

---

## nanochatのTransformerの特徴

nanochatのTransformerは、最新の研究成果を取り入れた現代的な実装です。

### 主要な技術的特徴（gpt.py:1-12）

1. **Rotary Embeddings（RoPE）**
   - 従来の絶対位置エンコーディングを使わない
   - Query/Keyに回転変換を適用して相対位置を表現
   - **実装**: `gpt.py:41-49`, `gpt.py:201-215`

2. **QK Normalization**
   - Query と Key を正規化して Attention を安定化
   - **実装**: `gpt.py:90`

3. **Untied Weights（重みの分離）**
   - トークン埋め込み（`wte`）と出力層（`lm_head`）で重みを共有しない
   - より柔軟な学習が可能
   - **実装**: `gpt.py:159`, `gpt.py:162`

4. **ReLU² Activation**
   - MLPの活性化関数に ReLU の2乗を使用
   - GELU より計算が高速
   - **実装**: `gpt.py:137`

5. **Pre-Normalization + Post-token embedding Norm**
   - トークン埋め込み直後に正規化
   - 各ブロックでも入力を正規化してから処理
   - **実装**: `gpt.py:149-150`, `gpt.py:272`, `gpt.py:275`

6. **RMSNorm（パラメータなし）**
   - 学習可能なパラメータを持たないシンプルな正規化
   - **実装**: `gpt.py:36-38`

7. **バイアスなし線形層**
   - すべての線形層で `bias=False`
   - パラメータ数削減とシンプル化
   - **実装**: `gpt.py:74-77`, `gpt.py:132-133`, `gpt.py:162`

8. **Multi-Query Attention（MQA）**
   - Key/Value を複数のQueryヘッドで共有
   - メモリ効率の向上（特に推論時のKVキャッシュ）
   - **実装**: `gpt.py:69-76`, `gpt.py:99-101`

9. **Logits Softcap**
   - 出力ロジットを tanh で制限して数値安定性向上
   - **実装**: `gpt.py:278`, `gpt.py:283`, `gpt.py:290`

### Transformerブロックの構造

各Transformerブロックは、**Self-Attention**と**MLP**の2つのサブレイヤーで構成されます。

```python
class Block(nn.Module):  # gpt.py:142
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache):
        x = x + self.attn(norm(x), cos_sin, kv_cache)  # 残差接続 + Attention
        x = x + self.mlp(norm(x))                       # 残差接続 + MLP
        return x
```

**重要な構造**:

1. **残差接続（Residual Connection）**
   - `x = x + ...` の形式
   - 勾配消失問題を緩和
   - 深いネットワークの訓練を可能にする

2. **Pre-Normalization**
   - `norm(x)` を Attention/MLP の**前**に適用
   - 訓練の安定性向上

**データの流れ**:
```
入力 x
  │
  ├─→ norm(x) → Attention → ＋─→ 出力1
  └───────────────────────────┘
                          （残差接続）

出力1
  │
  ├─→ norm(出力1) → MLP → ＋─→ 出力2
  └─────────────────────────┘
                       （残差接続）
```

### MLP（Multi-Layer Perceptron）

**MLPブロック**は、Attentionで得た情報をさらに変換します。

```python
class MLP(nn.Module):  # gpt.py:129
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)    # 拡張
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)  # 圧縮

    def forward(self, x):
        x = self.c_fc(x)         # n_embd → 4*n_embd（拡張）
        x = F.relu(x).square()   # ReLU²活性化
        x = self.c_proj(x)       # 4*n_embd → n_embd（元に戻す）
        return x
```

**次元の変化**:
```
n_embd (例: 768)
    ↓ [拡張層]
4*n_embd (例: 3072)
    ↓ [ReLU²]
4*n_embd
    ↓ [射影層]
n_embd (元に戻る)
```

**なぜ4倍に拡張？**
- より多くの非線形変換を可能にする
- 表現力の向上
- GPT-2以降の標準的な設計

---

## Forward処理の全体像

GPTモデルの推論（forward）は、以下の流れで行われます。

### 訓練時（gpt.py:259-286）

```python
def forward(self, idx, targets=None, kv_cache=None):
    B, T = idx.size()  # バッチサイズ、シーケンス長

    # 1. トークン埋め込み
    x = self.transformer.wte(idx)  # (B, T) → (B, T, n_embd)
    x = norm(x)  # 埋め込み直後に正規化

    # 2. Transformerブロックを順次適用
    for block in self.transformer.h:
        x = block(x, cos_sin, kv_cache)

    # 3. 最終正規化
    x = norm(x)

    # 4. ロジット計算
    logits = self.lm_head(x)  # (B, T, n_embd) → (B, T, vocab_size)
    logits = softcap * torch.tanh(logits / softcap)  # Softcap

    # 5. 損失計算（訓練時のみ）
    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               targets.view(-1),
                               ignore_index=-1)
        return loss
    else:
        return logits
```

### 推論時の生成（gpt.py:293-322）

```python
@torch.inference_mode()
def generate(self, tokens, max_tokens, temperature=1.0, top_k=None):
    """自己回帰的にトークンを生成"""
    for _ in range(max_tokens):
        logits = self.forward(ids)      # (B, T, vocab_size)
        logits = logits[:, -1, :]       # 最後のトークンのロジットのみ

        # Top-k サンプリング
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        # 温度パラメータで調整 → 確率分布化 → サンプリング
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        next_ids = torch.multinomial(probs, num_samples=1)

        # 生成したトークンを入力に追加
        ids = torch.cat((ids, next_ids), dim=1)
        yield next_ids.item()
```

**自己回帰生成の流れ**:
```
入力: [トークン1, トークン2, トークン3]
  ↓ forward
ロジット: [0.1, 0.3, 0.05, ...]  ← 各トークンの生成確率
  ↓ サンプリング
次のトークン: トークン4
  ↓ 入力に追加
新しい入力: [トークン1, トークン2, トークン3, トークン4]
  ↓ forward
次のトークン: トークン5
  ↓ ... 繰り返し
```

---

## まとめ：Transformerの重要概念

### 1. 並列処理とAttention
- RNN/LSTMと異なり、すべてのトークンを同時処理
- Attention機構で重要な部分に注目

### 2. Decoder-only構造
- Causal Attentionで未来を見ない
- 自己回帰的に1トークンずつ生成

### 3. 階層的な構造
- トークン埋め込み → Transformerブロック × N → 出力層
- 各ブロック: Attention + MLP + 残差接続

### 4. 位置エンコーディング
- Rotary Embeddingsで相対位置を表現
- 絶対位置エンコーディングより柔軟

### 5. 現代的な最適化
- RMSNorm, QK Norm, ReLU², Multi-Query Attention
- パラメータ削減と訓練安定化

---

## 次章への導入

第2章では、Transformerの基本構造とGPTモデルの仕組みを学びました。

### これまでに学んだこと
- Transformerは並列処理とAttention機構が特徴
- GPTはDecoder-only構造で自己回帰的に生成
- Self-Attentionは文脈を考慮した表現を作る
- Rotary Embeddingsで位置情報を表現
- nanochatは最新技術を取り入れた現代的実装

### 次章で学ぶこと

**第3章: トークナイザーとBPE**
- テキストをトークンに分割する仕組み
- BPE（Byte Pair Encoding）アルゴリズム
- RustBPEによる高速化
- トークナイザーの訓練と評価

TransformerはトークンIDの列を処理しますが、その前段階として**テキストをトークンに変換する**必要があります。次章では、この重要な前処理ステップを詳しく見ていきます。

---

**参照ドキュメント**:
- [nanochat/gpt.py](/Users/edom18/MyDesktop/PythonProjects/nanochat/nanochat/gpt.py:1) - GPTモデルの実装全体
  - `GPTConfig`: gpt.py:26
  - `norm`（RMSNorm）: gpt.py:36-38
  - `apply_rotary_emb`: gpt.py:41-49
  - `CausalSelfAttention`: gpt.py:64-126
  - `MLP`: gpt.py:129-139
  - `Block`: gpt.py:142-151
  - `GPT`: gpt.py:154-322

**関連する数学ドキュメント**:
- [Softmax関数](../doc/math/03-softmax.md)（作成予定）
- [Attention機構の数式](../doc/math/07-attention-mechanism.md)（作成予定）
- [正規化手法（RMSNorm）](../doc/math/08-layer-normalization.md)（作成予定）
- [位置エンコーディング（RoPE）](../doc/math/09-positional-encoding.md)（作成予定）
- [活性化関数（ReLU²）](../doc/math/11-activation-functions.md)（作成予定）

---

**前へ**: [第1章: プロジェクト概要](01-project-overview.md)
**次へ**: [第3章: トークナイザーとBPE](03-tokenizer.md)
**戻る**: [ドキュメント作成計画](../todo/documentation-plan.md)
