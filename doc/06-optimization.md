# 第6章: 最適化手法（Muon, AdamW）

## 目次
- [6.1 最適化の概要](#61-最適化の概要)
- [6.2 なぜ異なるオプティマイザーを使うのか](#62-なぜ異なるオプティマイザーを使うのか)
- [6.3 Muonオプティマイザー](#63-muonオプティマイザー)
  - [6.3.1 Muonの基本原理](#631-muonの基本原理)
  - [6.3.2 Newton-Schulz反復法](#632-newton-schulz反復法)
  - [6.3.3 直交化処理の詳細](#633-直交化処理の詳細)
  - [6.3.4 Muonの実装](#634-muonの実装)
  - [6.3.5 使用すべきパラメータと避けるべきパラメータ](#635-使用すべきパラメータと避けるべきパラメータ)
- [6.4 AdamWオプティマイザー](#64-adamwオプティマイザー)
  - [6.4.1 Adamの基本原理](#641-adamの基本原理)
  - [6.4.2 重み減衰（Weight Decay）](#642-重み減衰weight-decay)
  - [6.4.3 AdamWの実装](#643-adamwの実装)
  - [6.4.4 埋め込み層への適用](#644-埋め込み層への適用)
- [6.5 分散最適化（Distributed Optimization）](#65-分散最適化distributed-optimization)
  - [6.5.1 ZeRO-2スタイルのシャーディング](#651-zero-2スタイルのシャーディング)
  - [6.5.2 DistMuonの実装](#652-distmuonの実装)
  - [6.5.3 DistAdamWの実装](#653-distadamwの実装)
- [6.6 nanochatにおける最適化設定](#66-nanochatにおける最適化設定)
  - [6.6.1 パラメータグループ分割](#661-パラメータグループ分割)
  - [6.6.2 学習率の設定とスケーリング](#662-学習率の設定とスケーリング)
  - [6.6.3 ハイパーパラメータスケジューリング](#663-ハイパーパラメータスケジューリング)
  - [6.6.4 勾配クリッピング](#664-勾配クリッピング)
- [6.7 訓練ステップの詳細](#67-訓練ステップの詳細)
- [6.8 まとめ](#68-まとめ)

---

## 6.1 最適化の概要

深層学習における**最適化**とは、モデルの損失関数を最小化するようにパラメータを更新するプロセスです。

### 基本的な流れ

```
1. 順伝播 (Forward Pass)
   入力データからモデルで予測を計算
   ↓
2. 損失計算 (Loss Computation)
   予測と正解の差を損失関数で評価
   ↓
3. 逆伝播 (Backward Pass)
   損失に対する各パラメータの勾配を計算
   ↓
4. パラメータ更新 (Optimizer Step)
   オプティマイザーが勾配を使ってパラメータを更新
   ↓
1に戻る（次のバッチで繰り返し）
```

### 勾配降下法の基本

最もシンプルな最適化手法は**勾配降下法（Gradient Descent）**です：

```
θ_new = θ_old - η * ∇L(θ)
```

- `θ`: パラメータ
- `η`: 学習率（Learning Rate）
- `∇L(θ)`: 損失関数Lのパラメータθに対する勾配

しかし、実際の深層学習ではこれだけでは不十分で、より高度なオプティマイザーが使われます。

> 詳細な数学的解説は [`doc/math/05-gradient-descent.md`](math/05-gradient-descent.md) を参照してください。

---

## 6.2 なぜ異なるオプティマイザーを使うのか

nanochatプロジェクトでは、**パラメータの種類に応じて異なるオプティマイザー**を使用しています：

```python
# gpt.py:228-257
def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
    # パラメータを3つのグループに分割
    matrix_params = list(self.transformer.h.parameters())        # Transformerブロックの重み行列
    embedding_params = list(self.transformer.wte.parameters())   # 埋め込み層
    lm_head_params = list(self.lm_head.parameters())            # 出力層

    # AdamW: 埋め込み層と出力層用
    adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)

    # Muon: Transformerブロックの重み行列用
    muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)

    return [adamw_optimizer, muon_optimizer]
```

### なぜこの分割なのか？

| パラメータの種類 | 次元 | 特性 | 適したオプティマイザー |
|----------------|------|------|---------------------|
| **重み行列** (Linear層) | 2D | 高次元、密な相互作用 | **Muon** |
| **埋め込み層** | 2D (語彙×次元) | スパース更新、異なる更新頻度 | **AdamW** |
| **出力層** | 2D | スパース更新、語彙サイズに依存 | **AdamW** |
| **正規化層のパラメータ** | 1D | 低次元、スケール/バイアス | **AdamW** |

**重要な原則**：
- **Muon**は2D行列パラメータ（重み行列）に特化
- **AdamW**は0D/1D/埋め込み層に適している
- 各パラメータの特性に合わせた最適化が訓練を効率化

---

## 6.3 Muonオプティマイザー

### 6.3.1 Muonの基本原理

**Muon**（MomentUm Orthogonalized by Newton-schulz）は、Keller et al.によって開発された最適化手法です。

公式ブログ: https://kellerjordan.github.io/posts/muon/

Muonの特徴：
1. 内部でSGD + モメンタムを使用
2. 更新ベクトルを**直交行列で近似**（Orthogonalization）
3. Newton-Schulz反復法を使って効率的に直交化
4. bfloat16で安定して動作

```
┌─────────────────────────────────────────┐
│           Muon Algorithm Flow            │
└─────────────────────────────────────────┘

入力: パラメータ P, 勾配 G
  ↓
[1] SGD + Momentum
  モメンタムバッファ buf を更新
  buf ← β·buf + (1-β)·G
  ↓
[2] Nesterov (オプション)
  g ← lerp(G, buf, β)
  ↓
[3] Newton-Schulz Orthogonalization
  g ← zeropower_via_newtonschulz5(g)
  最も近い直交行列に変換
  ↓
[4] アスペクト比スケーリング
  scale = √(max(1, rows/cols))
  P ← P - lr × scale × g
  ↓
出力: 更新されたパラメータ P
```

### 6.3.2 Newton-Schulz反復法

**Newton-Schulz反復法**は、行列の**ゼロ乗（直交化）**を計算するアルゴリズムです。

通常、行列Gの特異値分解（SVD）を使って直交行列を得るには：
```
G = U·Σ·V^T  (SVD)
直交行列 = U·V^T  (Σを無視)
```

しかし、SVDは計算コストが高いため、Newton-Schulz反復を使って近似します。

#### 実装: `nanochat/muon.py:10-36`

```python
@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz反復でGの直交化（ゼロ乗）を計算
    5次の反復式を使用し、収束を高速化
    """
    assert G.ndim >= 2  # 2D以上の行列

    # 5次反復の係数（ゼロでの傾きを最大化するよう選ばれている）
    a, b, c = (3.4445, -4.7750, 2.0315)

    X = G.bfloat16()

    # 行数 > 列数の場合、転置して計算（効率化）
    if G.size(-2) > G.size(-1):
        X = X.mT

    # スペクトルノルムを1以下に正規化
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    # Newton-Schulz反復（デフォルト5ステップ）
    for _ in range(steps):
        A = X @ X.mT                    # X·X^T
        B = b * A + c * A @ A           # 5次項の計算
        X = a * X + B @ X               # 更新式

    # 転置していた場合は元に戻す
    if G.size(-2) > G.size(-1):
        X = X.mT

    return X
```

**ポイント**：
- **5次反復**を使用（通常の3次より速い収束）
- **bfloat16**で安定動作（GPUで高速）
- **スペクトルノルム正規化**で数値安定性を確保

> Newton-Schulz反復の数学的詳細は [`doc/math/10-optimization-algorithms.md`](math/10-optimization-algorithms.md#muon) を参照してください。

### 6.3.3 直交化処理の詳細

なぜ直交化が有効なのか？

1. **勾配の方向を正規化**
   - 直交行列は単位ノルムを保つ
   - 各次元が独立した更新を受ける

2. **条件数の改善**
   - 行列の条件数が改善され、訓練が安定化
   - 勾配消失/爆発を防ぐ

3. **効率的な探索**
   - パラメータ空間をより効率的に探索できる

### 6.3.4 Muonの実装

#### 基本的なMuonオプティマイザー: `nanochat/muon.py:38-84`

```python
class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    引数:
        lr: 学習率（デフォルト: 0.02）
        momentum: モメンタム係数（デフォルト: 0.95）
        nesterov: Nesterovモメンタムを使用するか（推奨）
        ns_steps: Newton-Schulz反復のステップ数（デフォルト: 5）
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params: list[Tensor] = [*params]

        # パラメータを要素数ごとにグループ化（効率化のため）
        param_groups = []
        for size in {p.numel() for p in params}:
            group = dict(params=[p for p in params if p.numel() == size])
            param_groups.append(group)

        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            for p in params:
                g = p.grad
                assert g is not None
                state = self.state[p]

                # モメンタムバッファの初期化
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf: Tensor = state["momentum_buffer"]

                # モメンタム更新: buf ← β·buf + (1-β)·g
                buf.lerp_(g, 1 - group["momentum"])

                # Nesterovモメンタム（推奨）
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf

                # Newton-Schulzで直交化
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

                # アスペクト比スケーリングを適用して更新
                # √(max(1, rows/cols)) でスケール
                p.add_(g, alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)
```

**重要な点**：
- **アスペクト比スケーリング**: `√(max(1, rows/cols))`
  - 行列の形状に応じて学習率を調整
  - 細長い行列（rows >> cols）は大きなスケールを受ける
  - これにより、異なる形状の重み行列が均一に学習される

### 6.3.5 使用すべきパラメータと避けるべきパラメータ

#### ✅ Muonを使用すべきパラメータ

- **Linear層の重み行列** (2D)
  - `nn.Linear`の`.weight`
  - CausalSelfAttentionの`c_attn`, `c_proj`
  - MLPの`c_fc1`, `c_fc2`, `c_proj`

#### ❌ Muonを避けるべきパラメータ

- **埋め込み層** (`wte.weight`)
  - スパース更新（全語彙のうち一部しか更新されない）
  - 直交化が意味を持たない

- **出力層** (`lm_head.weight`)
  - 埋め込み層と同様の理由

- **0D/1Dパラメータ**
  - バイアス項（存在する場合）
  - LayerNormのgain/bias
  - 直交化は2D行列にのみ適用可能

**公式ドキュメントの警告** (`muon.py:49-52`):
> - This optimizer should not be used for the embedding layer, the final fully connected layer,
>   or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
> - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

---

## 6.4 AdamWオプティマイザー

### 6.4.1 Adamの基本原理

**Adam**（Adaptive Moment Estimation）は、各パラメータに対して**適応的な学習率**を使用するオプティマイザーです。

Adamの特徴：
1. **1次モーメント（平均）**と**2次モーメント（分散）**を追跡
2. パラメータごとに学習率を自動調整
3. スパース勾配に強い（埋め込み層に適している）

#### Adamの更新式

```
# 1次モーメント（移動平均）
m_t = β1 * m_{t-1} + (1 - β1) * g_t

# 2次モーメント（移動平均の二乗）
v_t = β2 * v_{t-1} + (1 - β2) * g_t²

# バイアス補正
m̂_t = m_t / (1 - β1^t)
v̂_t = v_t / (1 - β2^t)

# パラメータ更新
θ_t = θ_{t-1} - η * m̂_t / (√v̂_t + ε)
```

- `β1`: 1次モーメント減衰率（通常0.9）
- `β2`: 2次モーメント減衰率（通常0.999）
- `ε`: 数値安定性のための小さな値

### 6.4.2 重み減衰（Weight Decay）

**AdamW**は、Adamに**重み減衰（Weight Decay）**を正しく適用したバージョンです。

通常のAdamでは、重み減衰を勾配に加えますが、AdamWでは**パラメータに直接**適用します：

```python
# AdamW: パラメータに直接適用（正しい方法）
θ = θ * (1 - wd * lr)  # 重み減衰
θ = θ - lr * update     # Adam更新
```

これにより、重み減衰が適応的学習率の影響を受けなくなり、より効果的な正則化が可能になります。

> 重み減衰の数学的詳細は [`doc/math/10-optimization-algorithms.md`](math/10-optimization-algorithms.md#adamw) を参照してください。

### 6.4.3 AdamWの実装

nanochatでは、通常の訓練では**PyTorchのfused AdamW**を使用し、分散訓練では**DistAdamW**を使用します。

#### PyTorchのAdamW使用例: `gpt.py:241-247`

```python
adam_groups = [
    dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
    dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
]
adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
```

**パラメータ設定**：
- `betas=(0.8, 0.95)`: β1=0.8, β2=0.95（標準的な(0.9, 0.999)より低め）
- `eps=1e-10`: 非常に小さなイプシロン（数値安定性）
- `fused=True`: PyTorchの最適化されたカーネルを使用

### 6.4.4 埋め込み層への適用

なぜ埋め込み層にAdamWが適しているのか？

1. **スパース更新**
   - 各バッチで使われる語彙は全体のごく一部
   - 更新されないトークンの埋め込みベクトルは変化しない
   - Adamは各パラメータごとに更新状態を保持するため、これに適している

2. **適応的学習率**
   - 頻繁に更新されるトークン（"the", "a"など）は小さな学習率
   - まれなトークンは大きな学習率
   - 自動的にバランスが取れる

3. **語彙サイズに依存**
   - 語彙サイズが大きい（nanochatでは50,257）
   - Muonの直交化は語彙次元で意味を持たない

---

## 6.5 分散最適化（Distributed Optimization）

大規模モデルの訓練では、複数のGPUを使った**分散訓練**が不可欠です。nanochatでは**ZeRO-2スタイル**の最適化を実装しています。

### 6.5.1 ZeRO-2スタイルのシャーディング

**ZeRO**（Zero Redundancy Optimizer）は、Microsoftが開発した分散最適化手法です。

ZeRO-2では：
- **勾配**をランク間で分散（reduce_scatter）
- **オプティマイザーの状態**を各ランクでシャーディング
- **パラメータ**は全ランクで複製（all_gather）

```
┌──────────────────────────────────────────────────┐
│        ZeRO-2 Style Distributed Optimization      │
└──────────────────────────────────────────────────┘

        Rank 0          Rank 1          Rank 2
          │               │               │
  ┌───────▼───────┐ ┌────▼────┐  ┌──────▼──────┐
  │  Full Grads   │ │ Full G  │  │  Full Grads │
  │  [g0,g1,g2]   │ │[g0,g1,g2]│  │ [g0,g1,g2]  │
  └───────┬───────┘ └────┬────┘  └──────┬──────┘
          │               │               │
          └───────┐       │       ┌───────┘
                  │       │       │
  [1] reduce_scatter(AVG) - 勾配を平均して分散
                  │       │       │
          ┌───────▼───┐ ┌─▼──┐ ┌─▼───────┐
          │   g0      │ │ g1 │ │   g2    │
          └───────┬───┘ └─┬──┘ └─┬───────┘
                  │       │       │
  [2] 各ランクが自分のシャードを更新（optimizer.step()）
                  │       │       │
          ┌───────▼───┐ ┌─▼──┐ ┌─▼───────┐
          │  p0(new)  │ │p1(n)│ │ p2(new) │
          └───────┬───┘ └─┬──┘ └─┬───────┘
                  │       │       │
                  └───────┐       │       ┌───────
                          │       │       │
  [3] all_gather - 更新されたパラメータを全ランクに複製
                          │       │       │
                  ┌───────▼───────▼───────▼───┐
                  │   Full Params (replicated) │
                  │      [p0, p1, p2]          │
                  └────────────────────────────┘
```

**メリット**：
- オプティマイザー状態のメモリ使用量を`1/N`に削減（Nはランク数）
- 通信量は適切にオーバーラップ可能
- スケーラビリティが高い

### 6.5.2 DistMuonの実装

#### 全体構造: `nanochat/muon.py:86-188`

```python
class DistMuon(torch.optim.Optimizer):
    """
    分散Muonオプティマイザー（ZeRO-2スタイル）

    - reduce_scatter(AVG)で勾配を平均化
    - all_gatherで更新後のパラメータを複製
    """
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5):
        params = list(params)
        assert all(p.ndim == 2 for p in params), "Muon expects 2D parameters only"

        rank = dist.get_rank()

        # 全パラメータを形状ごとにグループ化（決定的な順序で）
        shapes = sorted({p.shape for p in params})
        param_groups = []
        for shape in shapes:
            group_params = [p for p in params if p.shape == shape]
            device, dtype = group_params[0].device, group_params[0].dtype

            if rank == 0:
                print(f"Muon: Grouping {len(group_params)} params of shape {shape}")

            # ゼロバッファ（パディング用）
            param_groups.append(dict(
                params=group_params,
                zero_buffer=torch.zeros_like(group_params[0])
            ))

        super().__init__(param_groups, defaults)
```

#### ステップ処理: `muon.py:126-187`

```python
@torch.no_grad()
def step(self):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # [1] すべての勾配が存在することを確認
    assert all(p.grad is not None for group in self.param_groups for p in group["params"])

    # [2] reduce_scatter操作を開始（勾配の平均化）
    all_reduce_futures = []
    for group in self.param_groups:
        params = group["params"]
        zero_buffer = group["zero_buffer"]

        # world_sizeごとにパラメータをグループ化
        for base_i in range(0, len(params), world_size):
            owner_idx = base_i + rank  # このランクが所有するパラメータ

            # world_size個の勾配をスタック
            rs_input = [p.grad for p in params[base_i:base_i + world_size]]
            rs_input.extend([zero_buffer] * (world_size - len(rs_input)))  # パディング

            # 出力バッファ（このランクが所有するパラメータの勾配）
            rs_output = params[owner_idx].grad if owner_idx < len(params) else torch.empty_like(zero_buffer)

            # reduce_scatter: 勾配を平均化して分散
            work = dist.reduce_scatter(rs_output, rs_input, op=dist.ReduceOp.AVG, async_op=True).get_future()
            all_reduce_futures.append(work)

    # [3] 各ランクが自分の担当パラメータを更新
    future_idx = 0
    all_gather_futures = []
    for group in self.param_groups:
        params = group["params"]
        zero_buffer = group["zero_buffer"]

        for base_i in range(0, len(params), world_size):
            owner_idx = base_i + rank

            # reduce_scatterの完了を待つ
            all_reduce_futures[future_idx].wait()
            future_idx += 1

            # オーナーランクがMuon更新を実行
            if owner_idx < len(params):
                p = params[owner_idx]
                g = p.grad  # 平均化済みの勾配
                state = self.state[p]

                # モメンタムバッファ
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf: Tensor = state["momentum_buffer"]

                # モメンタム更新
                buf.lerp_(g, 1.0 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf

                # Newton-Schulz直交化
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

                # アスペクト比スケーリング
                scale = (max(1.0, p.size(-2) / p.size(-1)) ** 0.5)
                p.add_(g, alpha=-group["lr"] * scale)

            # [4] all_gather: 更新されたパラメータを全ランクに複製
            ag_input = params[owner_idx] if owner_idx < len(params) else zero_buffer
            ag_output = params[base_i:base_i + world_size]
            ag_output.extend([torch.empty_like(zero_buffer) for _ in range(world_size - len(ag_output))])

            work = dist.all_gather(ag_output, ag_input, async_op=True).get_future()
            all_gather_futures.append(work)

    # すべての非同期操作の完了を待つ
    torch.futures.collect_all(all_gather_futures).wait()
```

**重要な仕組み**：
1. **ブロック巡回割り当て**（Block-cyclic assignment）
   - パラメータをworld_sizeごとにグループ化
   - 各ランクが`rank % world_size`のインデックスを担当

2. **非同期通信**
   - `async_op=True`ですべての通信を非同期化
   - 計算と通信をオーバーラップさせて効率化

3. **モメンタムバッファのシャーディング**
   - 各ランクは自分が担当するパラメータのモメンタムのみを保持
   - メモリ効率が高い

### 6.5.3 DistAdamWの実装

#### 全体構造: `nanochat/adamw.py:10-78`

```python
class DistAdamW(torch.optim.Optimizer):
    """
    分散AdamWオプティマイザー（ZeRO-2スタイル）
    """
    def __init__(self, param_groups, lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(param_groups, defaults)

    @torch.compile
    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # [1] reduce_scatter: 勾配をスライスごとに平均化
        reduce_scatter_futures = []
        grad_slices = []
        for group in self.param_groups:
            params = group["params"]
            for base_i in range(len(params)):
                grad = params[base_i].grad
                rank_size = grad.shape[0] // world_size  # 各ランクのスライスサイズ
                grad_slice = torch.empty_like(grad[:rank_size])

                # 勾配をスライスごとに平均化
                reduce_scatter_futures.append(
                    dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                )
                grad_slices.append(grad_slice)

        # [2] AdamW更新（各ランクが自分のスライスを更新）
        idx = 0
        all_reduce_futures = []
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            params = group['params']

            for base in range(len(params)):
                # reduce_scatterの完了を待つ
                reduce_scatter_futures[idx].wait()

                p = params[base]
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]  # このランクのスライス

                lr = group['lr'] * getattr(p, "lr_mul", 1.0)
                state = self.state[p]
                g_slice = grad_slices[idx]

                # オプティマイザー状態の初期化（スライスのみ）
                if not state:
                    state['step'] = torch.tensor(0, dtype=torch.int64, device=p.device)
                    state['exp_avg'] = torch.zeros_like(p_slice)      # 1次モーメント
                    state['exp_avg_sq'] = torch.zeros_like(p_slice)   # 2次モーメント

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1
                t = state['step']

                # 重み減衰（パラメータに直接適用）
                if wd != 0:
                    eff_weight_decay = lr * wd * getattr(p, "wd_mul", 1.0)
                    p_slice.mul_(1 - eff_weight_decay)

                # モーメントの更新
                exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)           # m_t
                exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)  # v_t

                # バイアス補正
                bias1 = 1 - beta1 ** t
                bias2 = 1 - beta2 ** t

                # 更新ステップの計算
                denom = exp_avg_sq.sqrt().add_(eps)
                step_size = lr * (torch.sqrt(bias2) / bias1)
                update = exp_avg.div(denom).mul_(step_size)

                # パラメータ更新
                p_slice.add_(other=update, alpha=-1.0)

                idx += 1

                # [3] all_gather: 更新されたスライスを全ランクに複製
                all_reduce_futures.append(
                    dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future()
                )

        # すべてのall_gatherの完了を待つ
        torch.futures.collect_all(all_reduce_futures).wait()
```

**DistAdamWの特徴**：
1. **パラメータを第0次元でスライス**
   - 各ランクが`[rank * rank_size : (rank+1) * rank_size]`を担当
   - exp_avg, exp_avg_sqもスライスのみを保持

2. **@torch.compile**
   - PyTorchのコンパイラで最適化
   - カーネル融合で高速化

3. **非同期通信**
   - reduce_scatterとall_gatherを非同期実行
   - 計算と通信のオーバーラップ

---

## 6.6 nanochatにおける最適化設定

### 6.6.1 パラメータグループ分割

nanochatでは、GPTモデルのパラメータを**3つのグループ**に分割します。

#### gpt.py:228-257

```python
def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
    model_dim = self.config.n_embd
    ddp, rank, local_rank, world_size = get_dist_info()

    # [1] パラメータを3グループに分割
    matrix_params = list(self.transformer.h.parameters())      # Transformerブロック
    embedding_params = list(self.transformer.wte.parameters()) # 埋め込み層
    lm_head_params = list(self.lm_head.parameters())          # 出力層

    # 全パラメータが含まれていることを確認
    assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params)

    # [2] AdamWオプティマイザーの作成（埋め込みと出力層用）
    dmodel_lr_scale = (model_dim / 768) ** -0.5  # モデルサイズに応じたLRスケーリング

    if rank == 0:
        print(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

    adam_groups = [
        dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
        dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
    ]
    adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
    AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
    adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)

    # [3] Muonオプティマイザーの作成（重み行列用）
    muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
    MuonFactory = DistMuon if ddp else Muon
    muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)

    # [4] 2つのオプティマイザーを1つのリストにまとめる
    optimizers = [adamw_optimizer, muon_optimizer]

    # 各オプティマイザーのinitial_lrを保存（スケジューリング用）
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    return optimizers
```

### 6.6.2 学習率の設定とスケーリング

#### デフォルト学習率: `scripts/base_train.py:40-43`

```python
embedding_lr = 0.2      # 埋め込み層の学習率（Adam）
unembedding_lr = 0.004  # 出力層の学習率（Adam）
matrix_lr = 0.02        # 重み行列の学習率（Muon）
weight_decay = 0.0      # 重み減衰（Adam）
```

**なぜこの値？**

| パラメータ | 学習率 | 理由 |
|----------|--------|------|
| embedding_lr | 0.2 | 埋め込み層は大きな学習率で速く学習 |
| unembedding_lr | 0.004 | 出力層は小さめ（語彙サイズ大、慎重に更新） |
| matrix_lr | 0.02 | Muonは直交化があるため適度な学習率 |

#### モデルサイズに応じたスケーリング

```python
# gpt.py:238-240
dmodel_lr_scale = (model_dim / 768) ** -0.5
```

これは**√dmodelスケーリング**と呼ばれます：
- モデル次元が大きくなると学習率を小さくする
- 768次元でチューニングされたLRを基準にスケール
- 例: `model_dim=1280`の場合、`scale = (1280/768)^-0.5 ≈ 0.77`

### 6.6.3 ハイパーパラメータスケジューリング

#### 学習率スケジューラー: `base_train.py:143-157`

```python
warmup_ratio = 0.0      # ウォームアップなし
warmdown_ratio = 0.2    # 最後の20%でLRを減衰
final_lr_frac = 0.0     # 最終LRは0（完全に減衰）

def get_lr_multiplier(it):
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)

    if it < warmup_iters:
        # ウォームアップフェーズ（線形増加）
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        # 定常フェーズ（LR = 1.0）
        return 1.0
    else:
        # ウォームダウンフェーズ（線形減衰）
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac
```

**学習率の変化**：
```
LR Multiplier
    1.0 ┤─────────────────────────────────┐
        │                                 │
        │                                 │
        │                                 ╲
        │                                  ╲
        │                                   ╲
    0.0 ┤                                    ╲─────
        └───────────────────────────────────────────> Iteration
        0%                  80%            100%
              定常フェーズ        ウォームダウン
```

#### Muonモメンタムスケジューラー: `base_train.py:159-163`

```python
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum
```

最初の300イテレーションで0.85から0.95に線形増加：
- 初期: 小さなモメンタムで探索
- 後期: 大きなモメンタムで安定した更新

#### スケジューラーの適用: `base_train.py:268-276`

```python
# 学習率の更新
lrm = get_lr_multiplier(step)
for opt in optimizers:
    for group in opt.param_groups:
        group["lr"] = group["initial_lr"] * lrm  # initial_lrに乗算

# Muonモメンタムの更新
muon_momentum = get_muon_momentum(step)
for group in muon_optimizer.param_groups:
    group["momentum"] = muon_momentum

# オプティマイザーのステップ
for opt in optimizers:
    opt.step()
```

### 6.6.4 勾配クリッピング

**勾配クリッピング**は、勾配のノルムが大きくなりすぎるのを防ぐ手法です。

#### base_train.py:264-266

```python
grad_clip = 1.0  # 勾配クリッピング値（0.0 = 無効）

# 勾配クリッピング
if grad_clip > 0.0:
    torch.nn.utils.clip_grad_norm_(orig_model.parameters(), grad_clip)
```

**仕組み**：
```
全パラメータの勾配ノルム g_norm を計算
if g_norm > grad_clip:
    すべての勾配をスケール: g ← g * (grad_clip / g_norm)
```

これにより、勾配爆発を防ぎ、訓練を安定化します。

---

## 6.7 訓練ステップの詳細

1回の訓練ステップの流れ: `base_train.py:252-281`

```python
# [1] 勾配累積ループ
for micro_step in range(grad_accum_steps):
    with autocast_ctx:  # bfloat16混合精度
        loss = model(x, y)

    train_loss = loss.detach()  # ロギング用
    loss = loss / grad_accum_steps  # 正規化（.backward()は勾配を加算するため）
    loss.backward()  # 逆伝播

    x, y = next(train_loader)  # 次のバッチをプリフェッチ

# [2] 勾配クリッピング
if grad_clip > 0.0:
    torch.nn.utils.clip_grad_norm_(orig_model.parameters(), grad_clip)

# [3] 学習率の更新
lrm = get_lr_multiplier(step)
for opt in optimizers:
    for group in opt.param_groups:
        group["lr"] = group["initial_lr"] * lrm

# [4] Muonモメンタムの更新
muon_momentum = get_muon_momentum(step)
for group in muon_optimizer.param_groups:
    group["momentum"] = muon_momentum

# [5] オプティマイザーのステップ
for opt in optimizers:
    opt.step()

# [6] 勾配のゼロ化
model.zero_grad(set_to_none=True)
```

**重要なポイント**：
1. **勾配累積**（Gradient Accumulation）
   - `grad_accum_steps`回の順伝播/逆伝播を実行
   - 勾配は`.backward()`で累積される
   - 損失を`grad_accum_steps`で割って正規化

2. **混合精度訓練**（Mixed Precision）
   - `autocast_ctx`でbfloat16を使用
   - 順伝播/逆伝播を高速化
   - メモリ使用量を削減

3. **set_to_none=True**
   - 勾配をゼロで埋めず、Noneに設定
   - メモリ効率が良い

---

## 6.8 まとめ

この章では、nanochatプロジェクトで使用される最適化手法について学びました。

### 主要な概念

1. **異なるパラメータには異なるオプティマイザー**
   - **Muon**: 2D重み行列用（直交化で効率的な学習）
   - **AdamW**: 埋め込み層/出力層用（適応的学習率でスパース更新に対応）

2. **Muonオプティマイザー**
   - SGD + モメンタム + Newton-Schulz直交化
   - アスペクト比スケーリングで形状の異なる行列を均一に学習
   - bfloat16で安定動作

3. **AdamWオプティマイザー**
   - 1次/2次モーメント推定で適応的学習率
   - 重み減衰をパラメータに直接適用（正しい正則化）
   - スパース勾配に強い

4. **分散最適化（ZeRO-2スタイル）**
   - reduce_scatterで勾配を平均化・分散
   - 各ランクが担当パラメータのオプティマイザー状態のみを保持
   - all_gatherで更新後のパラメータを複製
   - メモリ効率と通信効率の両立

5. **ハイパーパラメータスケジューリング**
   - 学習率: ウォームダウン（最後の20%で減衰）
   - Muonモメンタム: 0.85→0.95に増加（最初の300ステップ）
   - モデルサイズに応じた√dmodelスケーリング

6. **勾配累積と混合精度**
   - 勾配累積で大きなバッチサイズを実現
   - bfloat16混合精度で高速化とメモリ削減

### コード参照

| 概念 | ファイル | 行番号 |
|------|---------|--------|
| Muonオプティマイザー | `nanochat/muon.py` | 38-84 |
| DistMuon | `nanochat/muon.py` | 86-188 |
| Newton-Schulz反復 | `nanochat/muon.py` | 10-36 |
| DistAdamW | `nanochat/adamw.py` | 10-78 |
| オプティマイザー設定 | `nanochat/gpt.py` | 228-257 |
| 訓練ステップ | `scripts/base_train.py` | 252-281 |
| LRスケジューラー | `scripts/base_train.py` | 143-157 |
| Muonモメンタムスケジューラー | `scripts/base_train.py` | 159-163 |

### 次のステップ

第7章では、訓練されたモデルを使って**推論**を行う方法について学びます：
- 推論エンジンの実装
- KVキャッシュの仕組み
- サンプリング手法（Top-k, Temperature）
- ストリーミング生成
- チャットインターフェースの実装

最適化手法の数学的詳細については、以下のドキュメントを参照してください：
- [`doc/math/05-gradient-descent.md`](math/05-gradient-descent.md): 勾配降下法
- [`doc/math/10-optimization-algorithms.md`](math/10-optimization-algorithms.md): 最適化アルゴリズム（SGD, Adam, AdamW, Muon）

---

**前の章**: [第5章: データパイプラインと訓練プロセス](05-training-pipeline.md)
**次の章**: [第7章: 推論エンジンとサンプリング](07-inference.md)
**目次に戻る**: [ドキュメントTOP](01-project-overview.md)
