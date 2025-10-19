# 第3章: トークナイザーとBPE

## 目次
1. [トークナイザーとは](#トークナイザーとは)
2. [BPE（Byte Pair Encoding）の仕組み](#bpebyte-pair-encodingの仕組み)
3. [nanochatのトークナイザー実装](#nanochatのトークナイザー実装)
4. [トークナイザーの訓練](#トークナイザーの訓練)
5. [評価と圧縮率](#評価と圧縮率)
6. [会話のトークン化](#会話のトークン化)
7. [次章への導入](#次章への導入)

---

## トークナイザーとは

**トークナイザー（Tokenizer）**は、テキストを**トークン（token）**と呼ばれる小さな単位に分割する仕組みです。

### なぜトークン化が必要なのか

Transformerモデルは、文字列を直接処理することができません。代わりに、**数値のID**として処理します。

```
テキスト: "Hello world!"
    ↓ トークン化
トークン: ["Hello", " world", "!"]
    ↓ ID変換
トークンID: [15496, 995, 0]
    ↓ Transformerで処理
```

### トークン化の3つの方法

#### 1. 文字レベル（Character-level）
各文字を1トークンとする。

```
"Hello" → ['H', 'e', 'l', 'l', 'o'] → [72, 101, 108, 108, 111]
```

**メリット**: 語彙サイズが小さい（英語なら約100文字）
**デメリット**: シーケンスが長くなる、意味的なまとまりがない

#### 2. 単語レベル（Word-level）
スペースで区切って単語を1トークンとする。

```
"Hello world!" → ['Hello', 'world', '!'] → [1234, 5678, 9012]
```

**メリット**: 意味的なまとまりがある
**デメリット**: 語彙サイズが巨大、未知語に対応できない

#### 3. サブワードレベル（Subword-level）
文字と単語の中間。頻出する部分文字列をトークンとする。

```
"unhappiness" → ['un', 'happiness'] → [123, 456]
```

**メリット**: 語彙サイズと意味的まとまりのバランスが良い
**デメリット**: トークン化アルゴリズムの学習が必要

**現代のLLMはサブワードレベルを採用**しています。nanochatも**BPE（Byte Pair Encoding）**というサブワードアルゴリズムを使用します。

---

## BPE（Byte Pair Encoding）の仕組み

**BPE（Byte Pair Encoding）**は、頻出するバイト列のペアを1つのトークンにまとめていくアルゴリズムです。

### BPEの基本原理

#### ステップ1: バイトレベルから始める

テキストを**UTF-8バイト列**として表現します。

```
"Hello" → UTF-8バイト列 → [72, 101, 108, 108, 111]
```

すべてのバイト値（0〜255）が初期語彙となります。

#### ステップ2: 頻出ペアをマージ

テキスト全体で最も頻繁に現れる**バイトペア**を見つけ、それを1つの新しいトークンとして登録します。

**例**:

```
初期状態（バイト列）:
[72, 101, 108, 108, 111]  ← "Hello"
[72, 101, 108, 108, 111]  ← "Hello"（複数回出現）
[72, 101, 108, 112]        ← "Help"

ペア頻度カウント:
(72, 101): 3回  ← 最頻出
(101, 108): 3回
(108, 108): 2回
...

マージ1: (72, 101) → トークンID 256
[256, 108, 108, 111]  ← "Hello"
[256, 108, 108, 111]  ← "Hello"
[256, 108, 112]       ← "Help"

マージ2: (108, 108) → トークンID 257
[256, 257, 111]  ← "Hello"
[256, 257, 111]  ← "Hello"
[256, 108, 112]  ← "Help"

... これを繰り返す
```

#### ステップ3: 語彙サイズに達するまで繰り返す

目標の語彙サイズ（nanochatでは65,536）に達するまで、マージを繰り返します。

```
語彙サイズ = 256（初期バイト） + マージ回数

nanochatの場合:
65,536 = 256 + 65,280回のマージ
```

### BPEの利点

1. **柔軟性**: 頻出する単語は1トークン、稀な単語は複数トークンに分割
2. **未知語対応**: どんな単語も最悪バイトレベルに分解できるので、未知語問題が起きない
3. **圧縮率**: 頻出パターンを効率的にエンコード

### 具体例

```
訓練データ: "low", "lower", "lowest", "low", "low"

初期語彙（簡略化）: l, o, w, e, r, s, t

頻度カウント:
"low" → ['l', 'o', 'w']（3回出現）
"lower" → ['l', 'o', 'w', 'e', 'r']
"lowest" → ['l', 'o', 'w', 'e', 's', 't']

最頻出ペア: ('l', 'o')

マージ1: ('l', 'o') → 'lo'
"low" → ['lo', 'w']
"lower" → ['lo', 'w', 'e', 'r']
"lowest" → ['lo', 'w', 'e', 's', 't']

最頻出ペア: ('lo', 'w')

マージ2: ('lo', 'w') → 'low'
"low" → ['low']
"lower" → ['low', 'e', 'r']
"lowest" → ['low', 'e', 's', 't']

... 繰り返し
```

---

## nanochatのトークナイザー実装

nanochatは2つのトークナイザー実装を提供しています。

### 1. HuggingFaceTokenizer（tokenizer.py:39）

HuggingFaceの`tokenizers`ライブラリを使用。

```python
class HuggingFaceTokenizer:
    """Light wrapper around HuggingFace Tokenizer for some utilities"""

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        tokenizer = HFTokenizer(BPE(
            byte_fallback=True,  # バイトレベルにフォールバック
            unk_token=None,      # 未知トークンなし
            fuse_unk=False,
        ))
        # ... 設定 ...
        tokenizer.train_from_iterator(text_iterator, trainer)
        return cls(tokenizer)
```

**メリット**: 訓練・推論の両方が可能
**デメリット**: APIが複雑

### 2. RustBPETokenizer（tokenizer.py:155）

**Rust**で実装された高速トークナイザー + **tiktoken**で推論。

```python
class RustBPETokenizer:
    """Light wrapper around tiktoken (for efficient inference) but train with rustbpe"""

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # 1) Rustで訓練
        tokenizer = rustbpe.Tokenizer()
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)

        # 2) tiktokenエンコーディングを構築（推論用）
        mergeable_ranks = tokenizer.get_mergeable_ranks()
        enc = tiktoken.Encoding(
            name="rustbpe",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )
        return cls(enc, "<|bos|>")
```

**メリット**: 高速（Rust実装）、推論が効率的（tiktoken）
**デメリット**: Rustのビルドが必要

**nanochatのデフォルト**: RustBPETokenizer

### 特殊トークン（tokenizer.py:13-25）

会話形式のデータでは、ユーザーとアシスタントのメッセージを区別するため、**特殊トークン**を使用します。

```python
SPECIAL_TOKENS = [
    "<|bos|>",              # Beginning of Sequence（ドキュメントの開始）
    "<|user_start|>",       # ユーザーメッセージの開始
    "<|user_end|>",         # ユーザーメッセージの終了
    "<|assistant_start|>",  # アシスタントメッセージの開始
    "<|assistant_end|>",    # アシスタントメッセージの終了
    "<|python_start|>",     # Pythonツール呼び出しの開始
    "<|python_end|>",       # Pythonツール呼び出しの終了
    "<|output_start|>",     # Pythonツール出力の開始
    "<|output_end|>",       # Pythonツール出力の終了
]
```

**使用例**:
```
<|bos|>
<|user_start|>こんにちは！<|user_end|>
<|assistant_start|>こんにちは！何かお手伝いできることはありますか？<|assistant_end|>
```

### GPT-4スタイルの分割パターン（tokenizer.py:30）

BPEを適用する前に、テキストを**正規表現で前分割**します。

```python
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
```

このパターンは以下のルールで分割します:

1. **英語の短縮形**: `'s`, `'t`, `'m`, `'ll`, `'ve`, `'re`
2. **単語**: 連続する文字（`\p{L}`は Unicode の文字クラス）
3. **数字**: 1〜2桁の数字（GPT-4は1〜3桁だが、nanochatは小さい語彙サイズのため1〜2桁）
4. **記号**: スペース以外の記号
5. **改行**: 改行文字
6. **スペース**: 連続するスペース

**なぜ前分割？**
- 意味的なまとまりを保つ（例: "don't" → "don" + "'t"）
- 異なる言語や数字の処理を制御

---

## トークナイザーの訓練

トークナイザーの訓練は`scripts/tok_train.py`で行われます。

### 訓練の流れ（tok_train.py:1-107）

```python
# 1. コマンドライン引数の解析
parser = argparse.ArgumentParser(description='Train a BPE tokenizer')
parser.add_argument('--max_chars', type=int, default=10_000_000_000, help='Maximum characters to train on')
parser.add_argument('--vocab_size', type=int, default=65536, help='Vocabulary size (default: 2^16)')
```

**デフォルト設定**:
- `max_chars`: 20億文字（speedrun.shでは`--max_chars=2000000000`を指定）
- `vocab_size`: 65,536（2^16）

### データイテレータ（tok_train.py:28-44）

```python
def text_iterator():
    """
    1) ドキュメントをバッチでロード
    2) 各ドキュメントを doc_cap 文字に制限
    3) max_chars に達したら終了
    """
    nchars = 0
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            doc_text = doc[:args.doc_cap]  # 各ドキュメントを10,000文字に制限
            nchars += len(doc_text)
            yield doc_text
            if nchars > args.max_chars:
                return
```

**データソース**: HuggingFaceの`fineweb-edu`データセット（高品質な教育コンテンツ）

### 訓練実行（tok_train.py:48-52）

```python
t0 = time.time()
tokenizer = RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size)
t1 = time.time()
print(f"Training time: {t1 - t0:.2f}s")
```

**訓練時間**: 約5分（speedrun.shでの実測）

### 保存とサニティチェック（tok_train.py:56-69）

```python
# 保存
tokenizer.save(tokenizer_dir)

# 簡単なテスト
test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
assert decoded == test_text  # 完全に復元できることを確認
```

### トークンバイト数のキャッシュ（tok_train.py:72-91）

各トークンのバイト数を保存して、**Bits per Byte**評価に使用します。

```python
vocab_size = tokenizer.get_vocab_size()
token_bytes = []
for token_id in range(vocab_size):
    token_str = tokenizer.decode([token_id])
    if token_str in special_set:
        token_bytes.append(0)  # 特殊トークンはカウントしない
    else:
        id_bytes = len(token_str.encode("utf-8"))
        token_bytes.append(id_bytes)

# PyTorchテンソルとして保存
token_bytes = torch.tensor(token_bytes, dtype=torch.int32)
torch.save(token_bytes, token_bytes_path)
```

**Bits per Byte**: 語彙サイズに依存しない評価メトリクス（後述）

---

## 評価と圧縮率

トークナイザーの性能は`scripts/tok_eval.py`で評価されます。

### 圧縮率（Compression Ratio）

トークナイザーの良し悪しは、**圧縮率**で測定できます。

```
圧縮率 = バイト数 / トークン数
```

**高い圧縮率 = 効率的なトークナイザー**

例:
```
テキスト: "Hello world!"（12バイト）
トークン: ["Hello", " world", "!"]（3トークン）
圧縮率 = 12 / 3 = 4.0 バイト/トークン
```

### 評価対象（tok_eval.py:8-144）

tok_eval.pyは、以下の多様なテキストで圧縮率を評価します:

1. **ニュース記事**（英語）
2. **韓国語テキスト**（非英語）
3. **コード**（Python）
4. **数式**（LaTeX）
5. **科学論文**（専門用語）
6. **fineweb-edu訓練データ**（見たことのあるデータ）
7. **fineweb-edu検証データ**（見たことのないデータ）

### 比較対象（tok_eval.py:167-174）

nanochatのトークナイザーを、GPT-2とGPT-4のトークナイザーと比較します。

```python
for tokenizer_name in ["gpt2", "gpt4", "ours"]:
    if tokenizer_name == "gpt2":
        tokenizer = RustBPETokenizer.from_pretrained("gpt2")
    elif tokenizer_name == "gpt4":
        tokenizer = RustBPETokenizer.from_pretrained("cl100k_base")  # GPT-4
    else:
        tokenizer = get_tokenizer()  # nanochat

    vocab_sizes[tokenizer_name] = tokenizer.get_vocab_size()
```

**語彙サイズ**:
- GPT-2: 50,257
- GPT-4: 100,256
- nanochat: 65,536

### 評価結果の例

```
Vocab sizes:
GPT-2: 50257
GPT-4: 100256
Ours: 65536

Comparison with GPT-2:
=====================================================================================
Text Type  Bytes    GPT-2           Ours            Relative     Better
           　       Tokens  Ratio   Tokens  Ratio   Diff %
-------------------------------------------------------------------------------------
news       2011     549     3.66    484     4.15    +11.8%       Ours
korean     724      407     1.78    289     2.51    +29.0%       Ours
code       1392     444     3.14    386     3.61    +13.1%       Ours
math       2916     1220    2.39    985     2.96    +19.3%       Ours
science    847      249     3.40    219     3.87    +12.0%       Ours
fwe-train  ...      ...     4.85    ...     5.02    +3.5%        Ours
fwe-val    ...      ...     4.82    ...     4.98    +3.3%        Ours
```

**解釈**:
- nanochatのトークナイザーは、ほぼすべてのテキストタイプでGPT-2より効率的
- 特に韓国語（+29%）や数学（+19%）で大幅改善
- fineweb-eduデータ（訓練データ）でも改善（+3.5%）

**GPT-4との比較**では、語彙サイズが大きいGPT-4の方が圧縮率は高いですが、nanochatも健闘します。

---

## 会話のトークン化

SFT（教師あり微調整）では、会話形式のデータをトークン化する必要があります。

### render_conversation（tokenizer.py:258-342）

`render_conversation`は、会話データを特殊トークンを含むトークンIDの列に変換します。

#### 入力: 会話データ

```python
conversation = {
    "messages": [
        {"role": "user", "content": "こんにちは！"},
        {"role": "assistant", "content": "こんにちは！何かお手伝いできますか？"},
    ]
}
```

#### 出力: トークンIDとマスク

```python
ids, mask = tokenizer.render_conversation(conversation)

# ids: トークンIDの列
# [bos_id, user_start_id, ..., user_end_id, assistant_start_id, ..., assistant_end_id]

# mask: 訓練対象かどうか（1=訓練対象、0=訓練対象外）
# [0, 0, ..., 0, 0, 1, 1, ..., 1]
```

**重要**: **アシスタントの発言のみ**がmask=1（訓練対象）です。ユーザーの発言は予測する必要がないため、mask=0です。

### 実装の詳細（tokenizer.py:258-342）

```python
def render_conversation(self, conversation, max_tokens=2048):
    ids, mask = [], []

    def add_tokens(token_ids, mask_val):
        ids.extend(token_ids)
        mask.extend([mask_val] * len(token_ids))

    # 特殊トークンの取得
    bos = self.get_bos_token_id()
    user_start = self.encode_special("<|user_start|>")
    user_end = self.encode_special("<|user_end|>")
    assistant_start = self.encode_special("<|assistant_start|>")
    assistant_end = self.encode_special("<|assistant_end|>")

    # 会話のトークン化
    add_tokens(bos, 0)  # ドキュメントの開始

    for message in messages:
        if message["role"] == "user":
            value_ids = self.encode(message["content"])
            add_tokens(user_start, 0)
            add_tokens(value_ids, 0)  # ユーザー発言はmask=0
            add_tokens(user_end, 0)

        elif message["role"] == "assistant":
            add_tokens(assistant_start, 0)
            value_ids = self.encode(message["content"])
            add_tokens(value_ids, 1)  # アシスタント発言はmask=1
            add_tokens(assistant_end, 1)

    # max_tokensで切り詰め
    ids = ids[:max_tokens]
    mask = mask[:max_tokens]
    return ids, mask
```

### ツール呼び出しの処理

アシスタントがPythonツールを使う場合の特殊処理:

```python
content = [
    {"type": "text", "text": "計算します"},
    {"type": "python", "text": "2 + 2"},
    {"type": "python_output", "text": "4"},
    {"type": "text", "text": "答えは4です"}
]
```

トークン化:
```
<|assistant_start|>
計算します
<|python_start|>2 + 2<|python_end|>  ← mask=1（訓練対象）
<|output_start|>4<|output_end|>       ← mask=0（Pythonの出力は訓練対象外）
答えは4です
<|assistant_end|>
```

### visualize_tokenization（tokenizer.py:344-354）

デバッグ用に、トークン化結果を色付きで表示する機能:

```python
def visualize_tokenization(self, ids, mask):
    tokens = []
    for token_id, mask_val in zip(ids, mask):
        token_str = self.decode([token_id])
        color = GREEN if mask_val == 1 else RED  # mask=1は緑、mask=0は赤
        tokens.append(f"{color}{token_str}{RESET}")
    return '|'.join(tokens)
```

**出力例**:
```
RED<|bos|>|RED<|user_start|>|REDこんにちは！|RED<|user_end|>|RED<|assistant_start|>|GREENこんにちは！|GREEN何か|GREENお手伝い|GREENできますか？|GREEN<|assistant_end|>
```

---

## まとめ：トークナイザーの重要ポイント

### 1. トークン化の必要性
- Transformerは数値IDを処理する
- 文字・単語・サブワードの3つのレベル
- 現代のLLMはサブワード（BPE）を採用

### 2. BPEアルゴリズム
- バイトレベルから始める
- 頻出ペアを反復的にマージ
- 語彙サイズまで繰り返す

### 3. nanochatの実装
- RustBPETokenizer（高速訓練）+ tiktoken（効率的推論）
- GPT-4スタイルの前分割パターン
- 特殊トークンで会話を構造化

### 4. 訓練と評価
- 20億文字のfineweb-eduデータで訓練
- 語彙サイズ: 65,536
- 圧縮率で評価（4.8文字/トークン）

### 5. 会話のトークン化
- 特殊トークンでユーザー/アシスタントを区別
- マスクでアシスタント発言のみ訓練対象
- ツール呼び出しも対応

---

## 次章への導入

第3章では、テキストをトークンに変換するトークナイザーの仕組みを学びました。

### これまでに学んだこと
- トークン化は文字列を数値IDに変換する前処理
- BPEは頻出パターンを効率的にエンコードするサブワードアルゴリズム
- nanochatはRust実装で高速訓練を実現
- 会話データは特殊トークンとマスクで構造化

### 次章で学ぶこと

**第4章: モデルの詳細実装**
- GPTモデルの各コンポーネントの詳細
- CausalSelfAttentionの実装
- Rotary Embeddings（RoPE）の数学的背景
- MLPとTransformerブロックの役割
- 重みの初期化と最適化の準備

トークナイザーでテキストをトークンIDに変換したら、次はそれをTransformerモデルで処理します。第4章では、第2章で概観したTransformerの各部品を、実装レベルで詳しく見ていきます。

---

**参照ドキュメント**:
- [nanochat/tokenizer.py](/Users/edom18/MyDesktop/PythonProjects/nanochat/nanochat/tokenizer.py:1) - トークナイザーの実装全体
  - `SPECIAL_TOKENS`: tokenizer.py:13-25
  - `SPLIT_PATTERN`: tokenizer.py:30
  - `HuggingFaceTokenizer`: tokenizer.py:39-148
  - `RustBPETokenizer`: tokenizer.py:155-375
  - `render_conversation`: tokenizer.py:258-342
- [scripts/tok_train.py](/Users/edom18/MyDesktop/PythonProjects/nanochat/scripts/tok_train.py:1) - トークナイザー訓練
- [scripts/tok_eval.py](/Users/edom18/MyDesktop/PythonProjects/nanochat/scripts/tok_eval.py:1) - トークナイザー評価

**関連する数学ドキュメント**:
- （トークナイザーは主にアルゴリズムの話なので、数学ドキュメントへの参照は少ない）

---

**前へ**: [第2章: Transformerアーキテクチャの基礎](02-transformer-basics.md)
**次へ**: [第4章: モデルの詳細実装](04-model-implementation.md)
**戻る**: [ドキュメント作成計画](../todo/documentation-plan.md)
