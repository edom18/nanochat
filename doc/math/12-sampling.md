# 第12章: 確率的サンプリング

## 目次
- [サンプリングとは何か](#サンプリングとは何か)
- [Greedy Decoding](#greedy-decoding)
- [Temperature Sampling](#temperature-sampling)
- [Top-k Sampling](#top-k-sampling)
- [Top-p (Nucleus) Sampling](#top-p-nucleus-sampling)
- [Beam Search](#beam-search)
- [比較とまとめ](#比較とまとめ)
- [nanochatでの実装](#nanochatでの実装)
- [練習問題](#練習問題)

---

## サンプリングとは何か

**サンプリング（Sampling）** は、言語モデルが次のトークンを**確率分布から選択**するプロセスです。

### 基本的な流れ

```
1. モデルが次トークンの確率分布を出力
   logits → Softmax → 確率分布

2. 確率分布から1つのトークンをサンプリング

3. サンプリングされたトークンを入力に追加

4. 繰り返し（自己回帰的に生成）
```

可視化：
```
語彙:     [the,  cat,  sat,  dog,  ran, ...]
Logits:   [2.1,  4.5,  1.8,  3.2,  0.5, ...]
          ↓ Softmax
確率:     [0.10, 0.63, 0.07, 0.20, 0.00, ...]
          ↓ サンプリング
選択:     "cat" (63%の確率で選択される可能性が高い)
```

### なぜ確率的サンプリングが必要か？

#### 決定論的（常に最大確率）の問題

```
入力: "The cat"
確率: [the: 0.5, cat: 0.3, sat: 0.2]
選択: "the" (最大)

次:
入力: "The cat the"
確率: [cat: 0.4, sat: 0.35, the: 0.25]
選択: "cat"

次:
入力: "The cat the cat"
確率: [the: 0.45, sat: 0.3, cat: 0.25]
選択: "the"

結果: "The cat the cat the cat the cat..."
→ 繰り返しループ、多様性の欠如
```

#### 確率的サンプリングの利点

```
同じ入力でも毎回異なる出力:
  実行1: "The cat sat on the mat."
  実行2: "The cat jumped over the fence."
  実行3: "The cat slept peacefully."

→ 多様性、創造性、自然さ
```

### サンプリングの目標

**バランス**を取ること：

1. **品質**: 文法的に正しく、意味が通る
2. **多様性**: 繰り返しを避け、創造的
3. **制御可能性**: ユースケースに応じて調整可能

---

## Greedy Decoding

**Greedy Decoding** は、常に**最も確率の高いトークン**を選びます。

### 数式

```
次のトークン = argmax_i P(token_i | context)

各ステップで確率最大のトークンを選択
```

### 例

```
語彙:     [the,  cat,  sat,  dog,  mat]
確率:     [0.05, 0.60, 0.15, 0.10, 0.10]
選択:     "cat" (0.60が最大)
```

### PyTorchでの実装

```python
import torch

logits = torch.tensor([2.1, 4.5, 1.8, 3.2, 0.5])
probs = F.softmax(logits, dim=-1)
# tensor([0.10, 0.63, 0.07, 0.20, 0.00])

next_token = torch.argmax(probs)
# tensor(1)  ← "cat"のインデックス
```

### 利点と欠点

**利点**:
- **決定論的**: 同じ入力に対して常に同じ出力
- **高速**: argmaxのみ（サンプリング不要）
- **安定**: 高確率トークンを選ぶため品質が安定

**欠点**:
- **繰り返し**: 同じパターンのループに陥りやすい
- **多様性の欠如**: 創造性がない
- **局所最適**: 長期的に最適でない可能性

#### Exposure Biasの問題

```
訓練時:
  教師データの次トークンを常に使用

推論時:
  自分で生成したトークンを使用

ミスが蓄積:
  1つ誤ると、その後の文脈が訓練時と異なる
  → さらに誤りやすくなる
```

---

## Temperature Sampling

**Temperature** は、確率分布の「鋭さ」を制御します。

### 数式

```
Logitsをスケーリング:
  logits_scaled = logits / T

確率分布:
  P_i = exp(logits_i / T) / Σ_j exp(logits_j / T)

ここで:
  T: 温度（Temperature）
  T = 1.0: 通常のSoftmax
  T → 0: Greedy（最大確率のみ）
  T → ∞: 一様分布
```

### 温度の効果

```
元のLogits: [2.0, 4.0, 1.0]

T = 1.0（通常）:
  Softmax([2.0, 4.0, 1.0]) = [0.12, 0.84, 0.04]

T = 0.5（低温、鋭い分布）:
  Softmax([4.0, 8.0, 2.0]) = [0.02, 0.98, 0.00]
  → 最大確率がさらに支配的

T = 2.0（高温、平らな分布）:
  Softmax([1.0, 2.0, 0.5]) = [0.21, 0.58, 0.21]
  → 確率が均等化
```

可視化：
```
T = 0.5（低温）:
 1.0 ┤   ■
     │   ■
 0.5 ┤   ■
     │□ ■ □
 0.0 ┤──────→
      0 1 2

T = 1.0（通常）:
 1.0 ┤
     │  ■
 0.5 ┤  ■
     │□ ■ □
 0.0 ┤──────→
      0 1 2

T = 2.0（高温）:
 1.0 ┤
     │ ■ ■
 0.5 ┤ ■ ■ ■
     │ ■ ■ ■
 0.0 ┤──────→
      0 1 2
```

### PyTorchでの実装

```python
def temperature_sampling(logits, temperature=1.0):
    """
    logits: (vocab_size,) 未正規化のスコア
    temperature: スケーリング係数
    """
    if temperature == 0.0:
        # Greedy
        return torch.argmax(logits)

    # Temperature適用
    scaled_logits = logits / temperature

    # Softmax
    probs = F.softmax(scaled_logits, dim=-1)

    # サンプリング
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token
```

### 温度の選択

```
T = 0.0〜0.5:
  用途: 事実的なタスク（QA、要約）
  特徴: 安定、予測可能、繰り返しのリスク

T = 0.7〜1.0:
  用途: バランス型（会話、記事生成）
  特徴: 適度な創造性と品質のバランス

T = 1.5〜2.0:
  用途: 創造的なタスク（物語、詩）
  特徴: 多様、驚き、品質の低下リスク
```

---

## Top-k Sampling

**Top-k Sampling** は、**上位k個のトークンのみ**を候補として、その中からサンプリングします。

### 数式

```
1. 確率分布を降順にソート
2. 上位k個を選択
3. それ以外の確率をゼロに設定
4. 再正規化してサンプリング
```

### 例

```
語彙:     [the,  cat,  sat,  dog,  mat,  ran,  ...]
確率:     [0.35, 0.25, 0.15, 0.10, 0.08, 0.05, ...]

k = 3 の場合:
  候補: [the: 0.35, cat: 0.25, sat: 0.15]
  その他をマスク: [..., dog: 0, mat: 0, ran: 0, ...]

  再正規化:
    合計 = 0.35 + 0.25 + 0.15 = 0.75
    [the: 0.467, cat: 0.333, sat: 0.200]

  この分布からサンプリング
```

### PyTorchでの実装

```python
def top_k_sampling(logits, k=50):
    """
    logits: (vocab_size,)
    k: 上位k個を保持
    """
    # Top-kの値とインデックスを取得
    top_k_values, top_k_indices = torch.topk(logits, k, dim=-1)

    # Top-k以外を-∞に設定
    logits_masked = torch.full_like(logits, float('-inf'))
    logits_masked.scatter_(0, top_k_indices, top_k_values)

    # Softmax
    probs = F.softmax(logits_masked, dim=-1)

    # サンプリング
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token
```

### 利点と欠点

**利点**:
- **低確率トークンを排除**: ノイズを除去
- **多様性を維持**: k個の中から選択

**欠点**:
- **固定k**: 分布の形状に関わらずk個を選択
  - 平らな分布でもk個
  - 鋭い分布でもk個
  - 最適なkが文脈によって異なる

#### 問題の例

```
鋭い分布:
  [0.90, 0.05, 0.03, 0.01, 0.01, ...]
  k=5 → [0.90, 0.05, 0.03, 0.01, 0.01] を保持
  問題: 0.01のトークンは不要（ノイズ）

平らな分布:
  [0.12, 0.11, 0.10, 0.09, 0.08, 0.08, 0.07, ...]
  k=3 → [0.12, 0.11, 0.10] のみ保持
  問題: 0.09, 0.08も妥当な選択肢なのに排除
```

---

## Top-p (Nucleus) Sampling

**Top-p Sampling**（Nucleus Sampling）は、**累積確率がp以上**になる最小のトークン集合からサンプリングします。

### 数式

```
1. 確率分布を降順にソート
2. 累積確率がp以上になるまでトークンを追加
3. それ以外をマスク
4. 再正規化してサンプリング
```

### 例

```
語彙:     [the,  cat,  sat,  dog,  mat,  ran,  ...]
確率:     [0.40, 0.30, 0.15, 0.08, 0.05, 0.02, ...]

p = 0.9 の場合:
  累積:
    the: 0.40
    cat: 0.40 + 0.30 = 0.70
    sat: 0.70 + 0.15 = 0.85
    dog: 0.85 + 0.08 = 0.93 ← 0.9を超えた

  候補: [the, cat, sat, dog]（累積確率0.93）
  その他をマスク

  再正規化:
    合計 = 0.93
    [the: 0.43, cat: 0.32, sat: 0.16, dog: 0.09]
```

### 動的な候補数

```
鋭い分布:
  [0.90, 0.05, 0.03, ...]
  p=0.9 → 候補1個 [0.90]
  → 確信が高い場合は少ない候補

平らな分布:
  [0.12, 0.11, 0.10, 0.09, 0.08, 0.08, 0.07, ...]
  p=0.9 → 候補9個（累積0.92）
  → 不確実な場合は多い候補

分布の形状に適応！
```

### PyTorchでの実装

```python
def top_p_sampling(logits, p=0.9):
    """
    logits: (vocab_size,)
    p: 累積確率の閾値
    """
    # Softmax
    probs = F.softmax(logits, dim=-1)

    # 降順にソート
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # 累積確率
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # pを超える位置を特定
    sorted_indices_to_remove = cumulative_probs > p

    # 最初のトークンは常に保持（累積確率0から開始）
    sorted_indices_to_remove[0] = False

    # 元のインデックスにマッピング
    indices_to_remove = sorted_indices[sorted_indices_to_remove]

    # マスク適用
    logits_masked = logits.clone()
    logits_masked[indices_to_remove] = float('-inf')

    # 再正規化してサンプリング
    probs = F.softmax(logits_masked, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token
```

### 利点

1. **適応的**: 分布の形状に応じて候補数を調整
2. **低確率の排除**: ノイズを除去しつつ柔軟性を維持
3. **実験的な成功**: GPT-2/3などで使用され、良好な結果

---

## Beam Search

**Beam Search** は、**複数の候補を同時に探索**して、最も確率の高いシーケンス全体を見つけます。

### アイデア

```
Greedy:
  各ステップで最良の1つのみ保持

Beam Search:
  各ステップで最良のk個（ビーム幅）を保持
  最終的に全体の確率が最大のシーケンスを選択
```

### 例

```
ビーム幅 = 2

ステップ1:
  候補: ["The", "A"]
  確率: [0.6, 0.4]

ステップ2:
  "The" の次: ["cat": 0.5, "dog": 0.3, "mat": 0.2]
    → "The cat" (0.6*0.5=0.30)
    → "The dog" (0.6*0.3=0.18)

  "A" の次: ["cat": 0.4, "dog": 0.4, "mat": 0.2]
    → "A cat" (0.4*0.4=0.16)
    → "A dog" (0.4*0.4=0.16)

  全体の上位2つ:
    1. "The cat" (0.30)
    2. "The dog" (0.18)

  これを続ける...
```

### 数式

```
各ステップで、確率の積（または対数確率の和）が最大の
k個のシーケンスを保持

スコア = log P(w_1) + log P(w_2|w_1) + ... + log P(w_n|w_1...w_{n-1})
```

### 利点と欠点

**利点**:
- **大域的に最適**: シーケンス全体を考慮
- **決定論的**: 同じ入力に対して同じ出力（ビーム幅固定の場合）
- **翻訳などで効果的**: 正確さが重要なタスク

**欠点**:
- **繰り返し**: 同じパターンの繰り返しが多い
- **多様性の欠如**: 全候補が似通う傾向
- **計算コスト**: ビーム幅分の並列計算が必要

### 使用例

```
機械翻訳: ビーム幅 4〜10
要約: ビーム幅 3〜5
会話: 通常使用しない（多様性が必要）
```

---

## 比較とまとめ

### サンプリング手法の系譜

```
決定論的:
  └─ Greedy Decoding

確率的:
  ├─ Temperature Sampling
  ├─ Top-k Sampling
  └─ Top-p (Nucleus) Sampling

探索ベース:
  └─ Beam Search
```

### 特性比較

| 手法 | 決定論的 | 多様性 | 品質安定性 | 計算コスト | 用途 |
|------|---------|--------|-----------|-----------|------|
| Greedy | ✓ | 低 | 高 | 低 | 事実的タスク |
| Temperature | ✗ | 調整可能 | 中 | 低 | 汎用 |
| Top-k | ✗ | 中 | 中 | 低〜中 | バランス型 |
| Top-p | ✗ | 高 | 中 | 低〜中 | 創造的タスク |
| Beam Search | ✓ | 低 | 高 | 高 | 翻訳、要約 |

### 組み合わせ

実際には、**複数の手法を組み合わせる**ことが一般的です。

```
例: Temperature + Top-k
  1. Temperature でスケーリング
  2. Top-k で候補を絞る
  3. サンプリング

例: Temperature + Top-p
  1. Temperature でスケーリング
  2. Top-p で動的に候補を選択
  3. サンプリング

nanochatはこのアプローチを使用
```

### 選択のガイドライン

```
事実的なタスク（QA、要約）:
  Greedy または Temperature=0.5〜0.7

バランス型（会話、記事）:
  Temperature=0.7〜1.0 + Top-k/Top-p

創造的なタスク（物語、詩）:
  Temperature=1.0〜1.5 + Top-p

翻訳、要約（正確さ重視）:
  Beam Search (width=4〜10)
```

---

## nanochatでの実装

### sample_next_token関数（engine.py:128-144）

```python
@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """
    logits: (batch, vocab_size) モデルの出力
    rng: 乱数生成器（再現性のため）
    temperature: 温度パラメータ
    top_k: Top-kサンプリングのk値
    """
    assert temperature >= 0.0

    # Greedy（温度0）
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    # Top-k Sampling
    if top_k is not None:
        k = min(top_k, logits.size(-1))  # 語彙サイズを超えないように

        # Top-kの値とインデックスを取得
        vals, idx = torch.topk(logits, k, dim=-1)

        # Temperature適用
        vals = vals / temperature

        # Softmax
        probs = F.softmax(vals, dim=-1)

        # サンプリング
        choice = torch.multinomial(probs, num_samples=1, generator=rng)

        # 元のインデックスに戻す
        return idx.gather(1, choice)

    # Temperature Sampling（Top-k なし）
    else:
        # Temperature適用
        logits = logits / temperature

        # Softmax
        probs = F.softmax(logits, dim=-1)

        # サンプリング
        return torch.multinomial(probs, num_samples=1, generator=rng)
```

### 使用例（run.py）

```python
from nanochat.engine import Engine

# Engineインスタンス作成
engine = Engine(model, tokenizer)

# テキスト生成（デフォルト設定）
response = engine.run(
    prompt="The cat",
    max_new_tokens=50,
    temperature=1.0,  # 通常のSoftmax
    top_k=50,         # Top-50サンプリング
)

# 事実的なタスク（低温）
response = engine.run(
    prompt="What is the capital of France?",
    temperature=0.5,  # 鋭い分布
    top_k=10,         # 保守的な候補
)

# 創造的なタスク（高温）
response = engine.run(
    prompt="Once upon a time",
    temperature=1.2,  # 平らな分布
    top_k=100,        # 多様な候補
)

# Greedy（決定論的）
response = engine.run(
    prompt="The answer is",
    temperature=0.0,  # 常に最大確率
    top_k=None,
)
```

### 実装のポイント

1. **Temperature + Top-k**: 両方の手法を組み合わせ
2. **乱数生成器**: `rng` パラメータで再現性を確保
3. **効率的な実装**: PyTorchの組み込み関数を使用
4. **柔軟性**: `temperature=0` でGreedyに切り替え可能

---

## 練習問題

### 問題1: Temperatureの効果

Logits `[2.0, 4.0, 1.0]` に対して、Temperature `T=0.5` と `T=2.0` でSoftmaxを計算してください（簡略化のため、exp関数の値を使用）。

```
exp(2) ≈ 7.39, exp(4) ≈ 54.60, exp(1) ≈ 2.72
exp(4) ≈ 54.60, exp(8) ≈ 2981, exp(2) ≈ 7.39
exp(1) ≈ 2.72, exp(2) ≈ 7.39, exp(0.5) ≈ 1.65
```

<details>
<summary>解答</summary>

```
T = 0.5（低温、鋭い分布）:
  Scaled logits: [2.0/0.5, 4.0/0.5, 1.0/0.5] = [4, 8, 2]

  Softmax:
    exp([4, 8, 2]) = [54.60, 2981, 7.39]
    合計 = 54.60 + 2981 + 7.39 = 3042.99

    確率 = [54.60/3042.99, 2981/3042.99, 7.39/3042.99]
         ≈ [0.018, 0.980, 0.002]

  最大値が支配的（98%）

T = 2.0（高温、平らな分布）:
  Scaled logits: [2.0/2.0, 4.0/2.0, 1.0/2.0] = [1, 2, 0.5]

  Softmax:
    exp([1, 2, 0.5]) = [2.72, 7.39, 1.65]
    合計 = 2.72 + 7.39 + 1.65 = 11.76

    確率 = [2.72/11.76, 7.39/11.76, 1.65/11.76]
         ≈ [0.231, 0.628, 0.140]

  より均等（最大値が63%）
```
</details>

### 問題2: Top-kサンプリング

確率分布 `[0.40, 0.30, 0.15, 0.08, 0.05, 0.02]` に対して、`k=3` でTop-kサンプリングを適用した場合の再正規化された確率を計算してください。

<details>
<summary>解答</summary>

```
元の確率: [0.40, 0.30, 0.15, 0.08, 0.05, 0.02]

Top-3を選択:
  [0.40, 0.30, 0.15, 0, 0, 0]

再正規化:
  合計 = 0.40 + 0.30 + 0.15 = 0.85

  新しい確率 = [0.40/0.85, 0.30/0.85, 0.15/0.85, 0, 0, 0]
             = [0.471, 0.353, 0.176, 0, 0, 0]

検証: 0.471 + 0.353 + 0.176 = 1.000 ✓
```
</details>

### 問題3: Top-pサンプリング

確率分布 `[0.50, 0.25, 0.15, 0.05, 0.03, 0.02]` に対して、`p=0.8` でTop-pサンプリングを適用した場合、どのトークンが候補に含まれますか？

<details>
<summary>解答</summary>

```
確率: [0.50, 0.25, 0.15, 0.05, 0.03, 0.02]

累積確率:
  トークン0: 0.50
  トークン1: 0.50 + 0.25 = 0.75
  トークン2: 0.75 + 0.15 = 0.90 ← 0.8を超えた

候補: トークン0, 1, 2（累積確率0.90）

再正規化:
  合計 = 0.90
  新しい確率 = [0.50/0.90, 0.25/0.90, 0.15/0.90, 0, 0, 0]
             = [0.556, 0.278, 0.167, 0, 0, 0]
```
</details>

### 問題4: サンプリング手法の選択

以下の状況で、どのサンプリング手法を選ぶべきか理由とともに答えてください。

1. 医療Q&Aシステム（正確性が最重要）
2. チャットボット（自然な会話）
3. 小説の自動生成（創造性重視）

<details>
<summary>解答</summary>

```
1. 医療Q&A（正確性最重要）:
   選択: Greedy または Temperature=0.0〜0.3
   理由:
     - 誤った情報は危険
     - 最も確信度の高い回答を選択
     - 多様性より正確性を優先

2. チャットボット（自然な会話）:
   選択: Temperature=0.7〜1.0 + Top-k (k=50〜100)
   理由:
     - 適度な多様性で自然さを維持
     - 繰り返しを避ける
     - 極端に低確率のトークンは排除

3. 小説の自動生成（創造性重視）:
   選択: Temperature=1.0〜1.5 + Top-p (p=0.9)
   理由:
     - 高い多様性で創造的な表現
     - Top-pで適応的に候補を調整
     - 品質の多少の低下を許容
```
</details>

---

## まとめ

### サンプリングの核心

```
目標: 品質と多様性のバランス

手法:
  - Greedy: 品質 ✓✓✓, 多様性 ✗
  - Temperature: 調整可能
  - Top-k: 固定的なフィルタリング
  - Top-p: 適応的なフィルタリング
```

### 実用的な推奨

```
デフォルト設定:
  Temperature = 0.7〜1.0
  Top-k = 50 または Top-p = 0.9

事実的タスク:
  Temperature = 0.0〜0.5（Greedyに近い）

創造的タスク:
  Temperature = 1.0〜1.5 + Top-p

組み合わせ:
  Temperature + Top-k/Top-p が最も一般的
```

### nanochatの戦略

```python
sample_next_token(
    logits,
    temperature=1.0,  # スケーリング
    top_k=50,         # 上位50個に絞る
)

柔軟性:
  - temperature=0 でGreedy
  - top_k=None でTemperatureのみ
  - 組み合わせで多様性を制御
```

### 次のステップ

- [数学03: Softmax関数](03-softmax.md) - 確率分布の計算
- [第7章: 推論エンジンとサンプリング](../07-inference.md) - 実装の詳細
- [数学04: 交差エントロピー損失](04-cross-entropy.md) - 訓練時の確率の扱い
