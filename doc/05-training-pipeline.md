# 第5章: データパイプラインと訓練プロセス

## 目次
1. [訓練パイプライン全体像](#訓練パイプライン全体像)
2. [データセット管理（dataset.py）](#データセット管理datasetpy)
3. [データローダー（dataloader.py）](#データローダーdataloader)
4. [訓練スクリプト（base_train.py）](#訓練スクリプトbase_trainpy)
5. [訓練ループの詳細](#訓練ループの詳細)
6. [分散訓練（DDP）](#分散訓練ddp)
7. [評価と検証](#評価と検証)
8. [次章への導入](#次章への導入)

---

## 訓練パイプライン全体像

nanochatの訓練パイプラインは、データのダウンロードからモデルの更新まで、一貫した流れで構成されています。

### データフロー

```
[HuggingFace fineweb-edu]
    ↓ ダウンロード（dataset.py）
[Parquetファイル（~/.cache/nanochat/base_data/）]
    ↓ ストリーミング読み込み（dataset.py）
[ドキュメントバッチ（テキスト）]
    ↓ トークン化（dataloader.py + tokenizer）
[トークンバッファ]
    ↓ バッチ化
[入力テンソル (B, T)]、[ターゲットテンソル (B, T)]
    ↓ GPU転送
[訓練ループ（base_train.py）]
    ↓ Forward
[損失計算]
    ↓ Backward
[勾配計算]
    ↓ Optimizer Step
[パラメータ更新]
```

### 主要コンポーネント

1. **dataset.py**: Parquetファイルの管理
   - ダウンロード
   - イテレーション

2. **dataloader.py**: データローダー
   - トークン化
   - バッチ生成
   - GPU転送

3. **base_train.py**: 訓練スクリプト
   - モデル初期化
   - オプティマイザー設定
   - 訓練ループ
   - 評価

---

## データセット管理（dataset.py）

**dataset.py**は、HuggingFaceのfineweb-eduデータセットをParquet形式で管理します。

### データセットの仕様（dataset.py:20-28）

```python
BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
MAX_SHARD = 1822  # 最後のシャードはshard_01822.parquet
index_to_filename = lambda index: f"shard_{index:05d}.parquet"
DATA_DIR = os.path.join(base_dir, "base_data")
```

**データセット詳細**:
- **名前**: fineweb-edu-100b-shuffle
- **総シャード数**: 1,823（shard_00000 〜 shard_01822）
- **1シャードあたり**: 約250M文字（約100MBの圧縮Parquetファイル）
- **総容量**: 約180GB（圧縮）
- **内容**: 高品質な教育コンテンツ（webから収集、フィルタリング済み）

### Parquetファイルのダウンロード（dataset.py:60-110）

```python
def download_single_file(index):
    """1つのシャードをダウンロード（リトライ機能付き）"""
    filename = index_to_filename(index)
    filepath = os.path.join(DATA_DIR, filename)

    # すでに存在する場合はスキップ
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    # URLを構築
    url = f"{BASE_URL}/{filename}"

    # 最大5回リトライ
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            # 一時ファイルに書き込み
            temp_path = filepath + f".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MBチャンク
                    if chunk:
                        f.write(chunk)

            # 成功したら一時ファイルを本番ファイルにリネーム
            os.rename(temp_path, filepath)
            return True

        except (requests.RequestException, IOError) as e:
            # 指数バックオフでリトライ
            wait_time = 2 ** attempt
            time.sleep(wait_time)

    return False
```

**並列ダウンロード**:
```python
# speedrun.shでの使用例
python -m nanochat.dataset -n 240  # 240シャードをダウンロード
```

内部では`multiprocessing.Pool`で並列ダウンロード（デフォルト4ワーカー）。

### Parquetファイルのイテレーション（dataset.py:43-57）

```python
def parquets_iter_batched(split, start=0, step=1):
    """
    Parquetファイルをバッチで読み込み、ドキュメントを生成
    - split: "train" または "val"（最後の1シャードがval）
    - start/step: DDP用のオフセット（例: start=rank, step=world_size）
    """
    assert split in ["train", "val"]
    parquet_paths = list_parquet_files()
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        # Row groupごとに読み込み（効率的）
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts
```

**Parquet Row Group**:
- Parquetファイルは内部的に「Row Group」に分割されている
- 1 Row Group = 通常1024行程度
- Row Groupごとに読み込むことで、メモリ効率が向上

**Train/Val Split**:
- **Train**: 最後の1シャード以外（shard_00000 〜 shard_01821）
- **Val**: 最後の1シャードのみ（shard_01822）

**DDP対応**:
- `start=rank, step=world_size`で各GPUが異なるRow Groupを処理
- 例: 8 GPU → GPU0はrow_group 0, 8, 16, ...、GPU1はrow_group 1, 9, 17, ...

---

## データローダー（dataloader.py）

**dataloader.py**は、ドキュメントをトークン化し、訓練用のバッチを生成します。

### tokenizing_distributed_data_loader（dataloader.py:9-49）

```python
def tokenizing_distributed_data_loader(B, T, split, tokenizer_threads=4, tokenizer_batch_size=128):
    """テキストをストリーミングでトークン化し、訓練バッチを生成"""
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    needed_tokens = B * T + 1  # +1はターゲット用

    # トークナイザーとBOSトークンを取得
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()

    # トークンバッファ（deque）
    token_buffer = deque()

    # スクラッチバッファ（GPU転送用）
    scratch = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)

    # ... (続く)
```

**パラメータ**:
- `B`: バッチサイズ
- `T`: シーケンス長
- `split`: "train" または "val"
- `tokenizer_threads`: トークン化の並列スレッド数
- `tokenizer_batch_size`: トークン化時のバッチサイズ

### データフロー

```python
# 無限イテレータでドキュメントバッチを生成
def document_batches():
    while True:
        for batch in parquets_iter_batched(split=split, start=ddp_rank, step=ddp_world_size):
            # トークナイザーバッチサイズに分割
            for i in range(0, len(batch), tokenizer_batch_size):
                yield batch[i:i+tokenizer_batch_size]

batches = document_batches()

# メインループ
while True:
    # 1. 必要なトークン数が集まるまでバッファに蓄積
    while len(token_buffer) < needed_tokens:
        doc_batch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
        for tokens in token_lists:
            token_buffer.extend(tokens)

    # 2. バッファからトークンを取り出してスクラッチに移動
    for i in range(needed_tokens):
        scratch[i] = token_buffer.popleft()

    # 3. 入力とターゲットを作成
    inputs_cpu = scratch[:-1].to(dtype=torch.int32)
    targets_cpu = scratch[1:]

    # 4. 2Dに整形してGPUに非同期転送
    inputs = inputs_cpu.view(B, T).to(device="cuda", dtype=torch.int32, non_blocking=True)
    targets = targets_cpu.view(B, T).to(device="cuda", dtype=torch.int64, non_blocking=True)

    yield inputs, targets
```

### トークンバッファの役割

```
ドキュメント1: [BOS, 10, 20, 30, ...]
ドキュメント2: [BOS, 15, 25, 35, ...]
    ↓ バッファに追加
token_buffer: [BOS, 10, 20, 30, ..., BOS, 15, 25, 35, ...]
    ↓ 必要数（B×T+1）取り出し
[BOS, 10, 20, ..., 15]  ← needed_tokens個
    ↓ 入力/ターゲットに分割
inputs:  [BOS, 10, 20, ..., 15][:-1]
targets: [BOS, 10, 20, ..., 15][1:]
```

**重要**: ドキュメント境界を跨いでバッチを作成します。これにより、ドキュメントの長さに関わらず、常に`B×T`トークンのバッチが生成されます。

### Pin Memory

```python
scratch = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
```

**Pin Memory**:
- CPUメモリをページング不可にする
- CPU→GPU転送が高速化
- `non_blocking=True`と組み合わせて非同期転送

---

## 訓練スクリプト（base_train.py）

**base_train.py**は、事前訓練（pretraining）を実行するメインスクリプトです。

### ハイパーパラメータ（base_train.py:28-52）

```python
# モデルアーキテクチャ
depth = 20                    # Transformerの層数
max_seq_len = 2048            # 最大シーケンス長

# 訓練期間
target_param_data_ratio = 20  # データ:パラメータ比（Chinchilla）

# 最適化
device_batch_size = 32        # デバイスあたりバッチサイズ
total_batch_size = 524288     # 総バッチサイズ（トークン数）
embedding_lr = 0.2            # 埋め込み層の学習率
unembedding_lr = 0.004        # 出力層の学習率
matrix_lr = 0.02              # 行列パラメータの学習率（Muon）
grad_clip = 1.0               # 勾配クリッピング

# 評価
eval_every = 250              # 検証損失の評価頻度
eval_tokens = 20*524288       # 検証に使うトークン数
core_metric_every = 2000      # COREメトリクス評価頻度
```

### モデルの初期化（base_train.py:74-106）

```python
# モデル次元の計算
num_layers = depth
model_dim = depth * 64        # アスペクト比64
num_heads = max(1, (model_dim + 127) // 128)  # ヘッド次元128
num_kv_heads = num_heads      # 1:1 MQA比率

# Meta deviceでモデルを初期化（メモリ効率化）
with torch.device("meta"):
    model_config = GPTConfig(
        sequence_len=max_seq_len,
        vocab_size=vocab_size,
        n_layer=num_layers,
        n_head=num_heads,
        n_kv_head=num_kv_heads,
        n_embd=model_dim
    )
    model = GPT(model_config)

# CUDAデバイスに移動して重みを初期化
model.to_empty(device="cuda")
model.init_weights()

# モデルのコンパイル（高速化）
orig_model = model
model = torch.compile(model, dynamic=False)
```

**Meta Device**:
- 実際のメモリを割り当てずにモデルを構築
- `to_empty()`で実メモリを割り当て
- メモリ効率的な初期化

**torch.compile**:
- PyTorch 2.0の機能
- 動的グラフを静的グラフにコンパイル
- 実行速度が向上

### 訓練期間の計算（base_train.py:108-126）

```python
# Chinchillaルール: トークン数 = 20 × パラメータ数
target_tokens = target_param_data_ratio * num_params
num_iterations = target_tokens // total_batch_size

print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Params ratio: {total_batch_size * num_iterations / num_params:.2f}")
```

**d20モデル（561Mパラメータ）の例**:
```
パラメータ数: 561,000,000
Chinchilla比率: 20
目標トークン数: 561M × 20 = 11.22B
総バッチサイズ: 524,288
訓練ステップ数: 11.22B / 524,288 ≈ 21,400ステップ
```

### 勾配累積の計算（base_train.py:85-92）

```python
tokens_per_fwdbwd = device_batch_size * max_seq_len  # 1回のforward/backwardあたりのトークン数
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size  # 全ランク合計
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
```

**例（8 GPU、device_batch_size=32、max_seq_len=2048）**:
```
tokens_per_fwdbwd = 32 × 2048 = 65,536（1 GPUあたり）
world_tokens_per_fwdbwd = 65,536 × 8 = 524,288（全GPU合計）
grad_accum_steps = 524,288 / 524,288 = 1（勾配累積なし）
```

もしGPUが1つなら:
```
world_tokens_per_fwdbwd = 65,536
grad_accum_steps = 524,288 / 65,536 = 8（8回累積）
```

---

## 訓練ループの詳細

訓練ループは、forward、backward、optimizerステップを繰り返します。

### メインループ（base_train.py:172以降）

```python
for step in range(num_iterations + 1):
    last_step = step == num_iterations

    # 1. 定期的に検証損失を評価
    if last_step or step % eval_every == 0:
        model.eval()
        val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        model.train()

    # 2. 定期的にCOREメトリクスを評価
    if last_step or (step > 0 and step % core_metric_every == 0):
        model.eval()
        results = evaluate_model(orig_model, tokenizer, device)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        model.train()

    # 3. 定期的にサンプル生成
    if last_step or (step > 0 and step % sample_every == 0):
        model.eval()
        engine = Engine(orig_model, tokenizer, device)
        # ... サンプル生成 ...
        model.train()

    # 4. 定期的にチェックポイント保存
    if last_step or (step > 0 and step % save_every == 0):
        save_checkpoint(...)

    # 5. 最後のステップならループを抜ける
    if last_step:
        break

    # 6. 訓練ステップ
    model.train()
    model.require_backward_grad_sync = False  # 勾配累積中は同期しない

    for micro_step in range(grad_accum_steps):
        # 最後のmicro-stepでのみ勾配を同期
        if micro_step == grad_accum_steps - 1:
            model.require_backward_grad_sync = True

        # Forward + Backward
        with autocast_ctx:  # bfloat16
            loss = model(x, y)
            loss = loss / grad_accum_steps  # 勾配累積のためスケーリング

        loss.backward()
        x, y = next(train_loader)  # 次のバッチを非同期でロード

    # 7. 勾配クリッピング
    if grad_clip > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # 8. 学習率スケジューリング
    lr_mult = get_lr_multiplier(step)
    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['initial_lr'] * lr_mult

    # 9. Muonのモーメンタム調整
    muon_momentum = get_muon_momentum(step)
    muon_optimizer.param_groups[0]['momentum'] = muon_momentum

    # 10. オプティマイザーステップ
    for optimizer in optimizers:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    # 11. 訓練損失をログ
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * loss.item()
    if step % 10 == 0:
        print0(f"Step {step:05d} | train loss {smooth_train_loss:.4f}")
```

### 重要なテクニック

#### 1. Mixed Precision Training（bfloat16）

```python
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

with autocast_ctx:
    loss = model(x, y)
```

**bfloat16**:
- 16ビット浮動小数点（float16と異なる指数部の範囲）
- メモリ使用量半減
- 計算速度向上
- float32と同じ範囲をカバー（オーバーフロー/アンダーフローに強い）

#### 2. 勾配累積

```python
for micro_step in range(grad_accum_steps):
    loss = model(x, y)
    loss = loss / grad_accum_steps  # スケーリング
    loss.backward()  # 勾配を累積

# 累積した勾配でパラメータ更新
optimizer.step()
```

**効果**: GPUメモリが足りなくても大きなバッチサイズを実現

#### 3. 勾配同期の制御（DDP）

```python
model.require_backward_grad_sync = False  # 勾配累積中は同期しない

for micro_step in range(grad_accum_steps):
    if micro_step == grad_accum_steps - 1:
        model.require_backward_grad_sync = True  # 最後のみ同期

    loss.backward()
```

**効果**: 通信オーバーヘッドを削減

#### 4. 学習率スケジューリング

```python
def get_lr_multiplier(it):
    warmup_iters = round(0.0 * num_iterations)       # Warmup: 0%
    warmdown_iters = round(0.2 * num_iterations)     # Warmdown: 20%

    if it < warmup_iters:
        return (it + 1) / warmup_iters               # 線形増加
    elif it <= num_iterations - warmdown_iters:
        return 1.0                                    # 一定
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * 0.0  # 線形減少
```

**スケジュール**:
```
LR = initial_lr × multiplier

0%          80%                    100%
|-----------|----------------------|
1.0         1.0 → 0.0 (線形減少)
```

---

## 分散訓練（DDP）

nanochatは**PyTorchのDistributedDataParallel（DDP）**を使用します。

### 初期化（common.py）

```python
def compute_init():
    ddp = int(os.environ.get('RANK', -1)) != -1  # torchrunで起動されたか
    if ddp:
        torch.distributed.init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        device = 'cuda:0'

    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device
```

### DDPモデルのラップ

```python
if ddp:
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[ddp_local_rank]
    )
```

### データの分散

```python
# 各ランクが異なるデータを処理
train_loader = tokenizing_distributed_data_loader(
    B, T, split="train",
    # 内部でstart=ddp_rank, step=ddp_world_sizeを使用
)
```

### 勾配の同期

```python
# Backwardで自動的に勾配がAll-reduce（平均）される
loss.backward()

# 勾配累積時は同期を制御
model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
```

### 実行コマンド

```bash
# 単一GPU
python scripts/base_train.py

# 8 GPU（DDP）
torchrun --standalone --nproc_per_node=8 -m scripts.base_train
```

---

## 評価と検証

### Bits per Byte（BPB）評価

```python
def evaluate_bpb(model, val_loader, eval_steps, token_bytes):
    """検証損失をBits per Byteで評価"""
    total_loss = 0
    total_bytes = 0

    for _ in range(eval_steps):
        x, y = next(val_loader)
        with torch.no_grad():
            loss = model(x, y)  # Cross-entropy loss
        total_loss += loss.item()

        # トークンIDからバイト数を取得
        mask = (y != -1)
        num_bytes = token_bytes[y[mask]].sum().item()
        total_bytes += num_bytes

    avg_loss = total_loss / eval_steps
    bpb = avg_loss / math.log(2) * total_bytes / (eval_steps * B * T)
    return bpb
```

**Bits per Byte**:
- 語彙サイズに依存しない評価指標
- 各トークンの実際のバイト数を考慮
- 圧縮効率を反映

### COREメトリクス

```python
results = evaluate_model(model, tokenizer, device, max_per_task=500)
print0(f"CORE metric: {results['core_metric']:.4f}")
```

**CORE（Comprehensive Robustness Evaluation）**:
- DCLMペーパーで提案された総合評価指標
- 複数のタスクの平均スコア
- 詳細は第8章で解説

### サンプル生成

```python
engine = Engine(model, tokenizer, device)
prompt = "Once upon a time"
prompt_tokens = tokenizer.encode(prompt)

for token in engine.generate(prompt_tokens, max_tokens=100, temperature=1.0):
    print(tokenizer.decode([token]), end='', flush=True)
```

---

## まとめ：訓練パイプラインの重要ポイント

### 1. データパイプライン
- HuggingFaceからParquetファイルをダウンロード
- Row Groupごとにストリーミング読み込み
- トークンバッファで効率的にバッチ化

### 2. 訓練スクリプト
- Meta deviceでメモリ効率的に初期化
- Chinchillaルールで訓練期間を決定
- torch.compileで高速化

### 3. 訓練ループ
- Mixed precision（bfloat16）
- 勾配累積
- 学習率スケジューリング

### 4. 分散訓練
- PyTorch DDP
- データの自動分散
- 勾配の自動同期

### 5. 評価
- Bits per Byte（語彙サイズ非依存）
- COREメトリクス（総合評価）
- サンプル生成（品質確認）

---

## 次章への導入

第5章では、データの準備から訓練ループの実行まで、訓練パイプライン全体を学びました。

### これまでに学んだこと
- fineweb-eduデータセットの管理
- 効率的なデータローダーの実装
- 訓練スクリプトの構成
- 分散訓練の仕組み
- 評価指標

### 次章で学ぶこと

**第6章: 最適化手法（Muon, AdamW）**
- Muonオプティマイザーの詳細
- AdamWオプティマイザーの詳細
- ZeRO-2スタイルの分散最適化
- 直交化（Orthogonalization）の数学的背景
- パラメータグループごとの最適化戦略

訓練パイプラインの流れを理解したので、次はモデルのパラメータを効率的に更新する最適化手法を詳しく見ていきます。MuonとAdamWという2つの異なるオプティマイザーがどのように連携するのかを学びます。

---

**参照ドキュメント**:
- [nanochat/dataset.py](/Users/edom18/MyDesktop/PythonProjects/nanochat/nanochat/dataset.py:1)
- [nanochat/dataloader.py](/Users/edom18/MyDesktop/PythonProjects/nanochat/nanochat/dataloader.py:1)
- [scripts/base_train.py](/Users/edom18/MyDesktop/PythonProjects/nanochat/scripts/base_train.py:1)

**関連する数学ドキュメント**:
- [交差エントロピー損失](../doc/math/04-cross-entropy.md)（作成予定）
- [勾配降下法](../doc/math/05-gradient-descent.md)（作成予定）

---

**前へ**: [第4章: モデルの詳細実装](04-model-implementation.md)
**次へ**: [第6章: 最適化手法](06-optimization.md)
**戻る**: [ドキュメント作成計画](../todo/documentation-plan.md)
