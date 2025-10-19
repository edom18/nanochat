# 数学02: 行列演算

## 目次
- [2.1 はじめに](#21-はじめに)
- [2.2 アダマール積（要素ごとの積）](#22-アダマール積要素ごとの積)
- [2.3 行列積の性質](#23-行列積の性質)
  - [2.3.1 交換法則は成り立たない](#231-交換法則は成り立たない)
  - [2.3.2 結合法則](#232-結合法則)
  - [2.3.3 分配法則](#233-分配法則)
- [2.4 単位行列](#24-単位行列)
- [2.5 行列のノルム](#25-行列のノルム)
  - [2.5.1 フロベニウスノルム](#251-フロベニウスノルム)
  - [2.5.2 スペクトルノルム](#252-スペクトルノルム)
- [2.6 トレース（対角和）](#26-トレース対角和)
- [2.7 バッチ行列積](#27-バッチ行列積)
- [2.8 Einstein記法とEinsum](#28-einstein記法とeinsum)
- [2.9 特異値分解（SVD）](#29-特異値分解svd)
- [2.10 nanochatでの使用例](#210-nanochatでの使用例)
- [2.11 まとめ](#211-まとめ)

---

## 2.1 はじめに

[前章](01-vectors-and-matrices.md)では、ベクトルと行列の基礎、および基本的な演算（行列積、転置）を学びました。

この章では、より高度な行列演算について、nanochatプロジェクトでの使用例とともに学びます。

---

## 2.2 アダマール積（要素ごとの積）

**アダマール積**（Hadamard product）は、同じ形状の行列の対応する要素同士を掛け合わせる演算です。記号は `⊙` です。

```
A ⊙ B

例:
    ┌        ┐       ┌        ┐       ┌          ┐
    │ 1  2   │   ⊙   │ 5  6   │   =   │  5  12   │
    │ 3  4   │       │ 7  8   │       │ 21  32   │
    └        ┘       └        ┘       └          ┘

計算:
  [1*5  2*6]   [5  12]
  [3*7  4*8] = [21 32]
```

**行列積との違い**：

```python
import torch

A = torch.tensor([[1, 2],
                  [3, 4]])
B = torch.tensor([[5, 6],
                  [7, 8]])

# アダマール積（要素ごとの積）
hadamard = A * B
print(hadamard)
# tensor([[ 5, 12],
#         [21, 32]])

# 行列積
matmul = A @ B
print(matmul)
# tensor([[19, 22],
#         [43, 50]])
```

### nanochatでの使用例

```python
# gpt.py:180-181（Attention計算）
# Attention重みの計算後、マスクを適用

# attn: (B, H, T, T) - Attention重み
# mask: (T, T) - Causalマスク（下三角）

attn = attn.masked_fill(mask == 0, float('-inf'))
# これは mask との要素ごとの操作
```

---

## 2.3 行列積の性質

### 2.3.1 交換法則は成り立たない

行列積は**非可換**です：`A @ B ≠ B @ A`

```python
A = torch.tensor([[1, 2],
                  [3, 4]])
B = torch.tensor([[5, 6],
                  [7, 8]])

print(A @ B)
# tensor([[19, 22],
#         [43, 50]])

print(B @ A)
# tensor([[23, 34],
#         [31, 46]])
```

**理由**：
- 行列積は「Aの行」と「Bの列」の内積
- 順序を変えると、全く異なる計算になる

### 2.3.2 結合法則

行列積は**結合法則**を満たします：`(A @ B) @ C = A @ (B @ C)`

```python
A = torch.randn(2, 3)
B = torch.randn(3, 4)
C = torch.randn(4, 5)

result1 = (A @ B) @ C  # (2, 5)
result2 = A @ (B @ C)  # (2, 5)

print(torch.allclose(result1, result2))  # True
```

**計算効率の違い**：
```
例: A(10, 100), B(100, 10), C(10, 1000)

(A @ B) @ C:
  A @ B: 10 × 100 × 10 = 10,000 演算 → (10, 10)
  結果 @ C: 10 × 10 × 1000 = 100,000 演算
  合計: 110,000 演算

A @ (B @ C):
  B @ C: 100 × 10 × 1000 = 1,000,000 演算 → (100, 1000)
  A @ 結果: 10 × 100 × 1000 = 1,000,000 演算
  合計: 2,000,000 演算
```

適切な括弧の付け方で、計算量を大幅に削減できます！

### 2.3.3 分配法則

行列積は**分配法則**を満たします：

```
A @ (B + C) = A @ B + A @ C  （右分配法則）
(A + B) @ C = A @ C + B @ C  （左分配法則）
```

```python
A = torch.randn(2, 3)
B = torch.randn(3, 4)
C = torch.randn(3, 4)

# 右分配法則
result1 = A @ (B + C)
result2 = A @ B + A @ C
print(torch.allclose(result1, result2))  # True
```

---

## 2.4 単位行列

**単位行列**（Identity matrix）は、対角成分が1、それ以外が0の正方行列です。記号は `I` です。

```
3×3の単位行列:
    ┌          ┐
    │ 1  0  0  │
I = │ 0  1  0  │
    │ 0  0  1  │
    └          ┘
```

**性質**：
```
A @ I = I @ A = A  （単位元）

任意の行列Aに対して、単位行列を掛けてもAは変わらない
```

```python
import torch

# 3×3の単位行列
I = torch.eye(3)
print(I)
# tensor([[1., 0., 0.],
#         [0., 1., 0.],
#         [0., 0., 1.]])

A = torch.randn(3, 3)

# A @ I = A
print(torch.allclose(A @ I, A))  # True

# I @ A = A
print(torch.allclose(I @ A, A))  # True
```

---

## 2.5 行列のノルム

**ノルム**は、ベクトルや行列の「大きさ」を測る尺度です。

### 2.5.1 フロベニウスノルム

**フロベニウスノルム**は、全要素の二乗和の平方根です。

```
‖A‖_F = √(Σ_i Σ_j |a_ij|²)
```

```python
A = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])

# フロベニウスノルム
frobenius_norm = torch.norm(A, p='fro')
print(frobenius_norm)  # tensor(5.4772)

# 手動計算
manual = torch.sqrt((A ** 2).sum())
print(manual)  # tensor(5.4772)
# √(1² + 2² + 3² + 4²) = √30 ≈ 5.477
```

### 2.5.2 スペクトルノルム

**スペクトルノルム**は、行列の最大特異値です。

```python
A = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])

# スペクトルノルム
spectral_norm = torch.norm(A, p=2)
print(spectral_norm)  # tensor(5.4650)
```

**nanochatでの使用例**：

```python
# muon.py:27（Newton-Schulz反復）
# スペクトルノルムを1以下に正規化

X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
```

---

## 2.6 トレース（対角和）

**トレース**は、正方行列の対角成分の和です。

```
    ┌          ┐
    │ 1  2  3  │
A = │ 4  5  6  │
    │ 7  8  9  │
    └          ┘

tr(A) = 1 + 5 + 9 = 15
```

```python
A = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

trace = torch.trace(A)
print(trace)  # tensor(15)
```

**性質**：
```
tr(A + B) = tr(A) + tr(B)
tr(cA) = c · tr(A)
tr(A^T) = tr(A)
tr(AB) = tr(BA)  （巡回性）
```

---

## 2.7 バッチ行列積

深層学習では、**バッチ**（複数のサンプル）を同時に処理します。PyTorchは高次元テンソルの行列積をサポートしています。

```python
# バッチサイズ32、各行列は (3, 4)
A = torch.randn(32, 3, 4)

# バッチサイズ32、各行列は (4, 5)
B = torch.randn(32, 4, 5)

# バッチ行列積
C = A @ B  # (32, 3, 5)

# 各バッチで独立に行列積が計算される
# C[i] = A[i] @ B[i] for i in range(32)
```

**形状の規則**：
```
(B, m, n) @ (B, n, p) → (B, m, p)
     ↑↑↑      ↑↑↑
  一致する    一致する
```

**ブロードキャスティング**も適用されます：

```python
# Aはバッチ、Bは単一行列
A = torch.randn(32, 3, 4)
B = torch.randn(4, 5)  # バッチ次元なし

C = A @ B  # (32, 3, 5)
# Bが全バッチで共有される
```

### nanochatでの使用例

```python
# gpt.py:115-123（Attention計算）
# Q, K, V: (B, H, T, d)
# B: バッチサイズ、H: ヘッド数、T: シーケンス長、d: ヘッド次元

# Q @ K^T: (B, H, T, d) @ (B, H, d, T) → (B, H, T, T)
attn = Q @ K.transpose(-2, -1)

# attn @ V: (B, H, T, T) @ (B, H, T, d) → (B, H, T, d)
output = attn @ V
```

各バッチ、各ヘッドで独立に行列積が計算されます。

---

## 2.8 Einstein記法とEinsum

**Einstein記法**（アインシュタイン記法）は、テンソルの縮約を簡潔に表現する記法です。

PyTorchの`torch.einsum`は、この記法でテンソル演算を記述できます。

### 基本的な使用例

```python
import torch

# 行列積: A @ B
A = torch.randn(3, 4)
B = torch.randn(4, 5)

# 明示的な行列積
C1 = A @ B  # (3, 5)

# einsumで同じ計算
C2 = torch.einsum('ij,jk->ik', A, B)  # (3, 5)
print(torch.allclose(C1, C2))  # True
```

**記法の意味**：
```
'ij,jk->ik'
 ↑↑ ↑↑  ↑↑
 A  B   C

- 'ij': Aの形状 (i, j)
- 'jk': Bの形状 (j, k)
- '->ik': 出力の形状 (i, k)
- 同じ文字（j）が縮約される（和を取る）
```

### より複雑な例

```python
# バッチ行列積
A = torch.randn(32, 3, 4)  # (B, i, j)
B = torch.randn(32, 4, 5)  # (B, j, k)

C = torch.einsum('bij,bjk->bik', A, B)  # (32, 3, 5)
```

```python
# トレース
A = torch.randn(5, 5)
trace = torch.einsum('ii->', A)  # スカラー
# 同じ: torch.trace(A)
```

```python
# Attention計算（簡略版）
Q = torch.randn(2, 8, 10, 64)  # (B, H, T, d)
K = torch.randn(2, 8, 10, 64)  # (B, H, T, d)

# Q @ K^T
attn = torch.einsum('bhid,bhjd->bhij', Q, K)  # (B, H, T, T)
#                    ↑↑     ↑↑    ↑↑
#                    同じ  同じ  縮約
```

**利点**：
- 明示的で分かりやすい
- 複雑なテンソル演算を1行で表現
- 最適化された実装

---

## 2.9 特異値分解（SVD）

**特異値分解**（Singular Value Decomposition, SVD）は、任意の行列を3つの行列の積に分解する手法です。

```
A = U Σ V^T

- U: 左特異ベクトル（直交行列）
- Σ: 特異値（対角行列）
- V: 右特異ベクトル（直交行列）
```

```python
A = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])  # (2, 3)

U, S, V = torch.svd(A)
# U: (2, 2)
# S: (2,) - 特異値
# V: (3, 2)

# 再構成
Sigma = torch.diag(S)
A_reconstructed = U @ Sigma @ V.T
print(torch.allclose(A, A_reconstructed))  # True
```

### SVDの応用

1. **低ランク近似**
   - 小さい特異値を無視して、データを圧縮

2. **主成分分析（PCA）**
   - データの主要な変動方向を抽出

3. **Muon最適化**（nanochat）
   - 直交行列の計算（Newton-Schulz反復の理論的基礎）

**nanochatでの関連**：

```python
# muon.py:10-36
# Newton-Schulz反復は、SVDの近似計算
# G = U Σ V^T → 直交行列 U V^T を求める
```

---

## 2.10 nanochatでの使用例

### 1. Attention計算（バッチ行列積）

```python
# gpt.py:115-123
Q = self.wq(x)  # (B, T, D) → (B, T, H*d) → (B, H, T, d)
K = self.wk(x)
V = self.wv(x)

# Q @ K^T: (B, H, T, d) @ (B, H, d, T) → (B, H, T, T)
attn = Q @ K.transpose(-2, -1) / math.sqrt(d)

# softmax
attn = F.softmax(attn, dim=-1)

# attn @ V: (B, H, T, T) @ (B, H, T, d) → (B, H, T, d)
output = attn @ V
```

### 2. 重み初期化（ノルム）

```python
# gpt.py:171-175
# 重みを初期化後、ノルムを調整

for pn, p in self.named_parameters():
    if pn.endswith('c_proj.weight'):
        # 標準偏差を調整
        torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
```

### 3. Newton-Schulz反復（スペクトルノルム）

```python
# muon.py:27
# 行列のスペクトルノルムを1以下に正規化

X = G.bfloat16()
if G.size(-2) > G.size(-1):
    X = X.mT

# スペクトルノルムを1以下に
X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
```

### 4. 分散計算（トレース）

```python
# RMSNormの分散計算（概念）
# x: (B, T, D)

variance = (x ** 2).mean(dim=-1, keepdim=True)  # (B, T, 1)
# 各ベクトルのノルムの二乗の平均
```

---

## 2.11 まとめ

この章では、行列演算の詳細について学びました。

### 主要な概念

1. **アダマール積**
   - 要素ごとの積: `A * B`
   - 同じ形状の行列のみ

2. **行列積の性質**
   - 非可換: `A @ B ≠ B @ A`
   - 結合法則: `(A @ B) @ C = A @ (B @ C)`
   - 分配法則: `A @ (B + C) = A @ B + A @ C`

3. **単位行列**
   - `torch.eye(n)`
   - `A @ I = I @ A = A`

4. **ノルム**
   - フロベニウスノルム: `torch.norm(A, p='fro')`
   - スペクトルノルム: `torch.norm(A, p=2)`

5. **トレース**
   - 対角成分の和: `torch.trace(A)`

6. **バッチ行列積**
   - `(B, m, n) @ (B, n, p) → (B, m, p)`
   - 各バッチで独立に計算

7. **Einsum**
   - Einstein記法で複雑なテンソル演算を表現
   - `torch.einsum('ij,jk->ik', A, B)`

8. **SVD**
   - `A = U Σ V^T`
   - 直交行列の計算、低ランク近似

### nanochatでの主な使用箇所

| 演算 | コード位置 | 説明 |
|------|----------|------|
| バッチ行列積 | `gpt.py:115-123` | Attention: `Q @ K^T`, `attn @ V` |
| スペクトルノルム | `muon.py:27` | Newton-Schulz反復の正規化 |
| 要素ごとの積 | `gpt.py:180` | Attentionマスク適用 |

### 次のステップ

次の数学ドキュメントでは、以下を学びます：
- **Softmax関数**: logitsを確率分布に変換
- **交差エントロピー損失**: 分類タスクの損失関数
- **Attention機構の数式**: Q, K, Vの詳細な計算

行列演算の知識があれば、これらの概念をより深く理解できます。

---

**関連ドキュメント**:
- [数学01: ベクトルと行列の基礎](01-vectors-and-matrices.md)
- [数学03: Softmax関数](03-softmax.md)
- [数学07: Attention機構の数式](07-attention-math.md)
- [第6章: 最適化手法（Muon, AdamW）](../06-optimization.md)
