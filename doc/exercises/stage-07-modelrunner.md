# 阶段 7 · ModelRunner（单卡 eager 版）

## 1. 结论

**实现一个把 `list[Sequence]` 翻译成 GPU 张量、跑模型、采样、再返回 token_ids 的执行器；先做 enforce_eager 的单卡版本。**

## 2. 为什么这么做

- ModelRunner 是**模型层与调度层之间的桥**：上面是面向「请求/序列」的语义，下面是面向「张量/批次」的执行。
- 它解决了 4 件杂事：初始化分布式、加载模型、按需分配 KV Cache、把 schedule 出来的 batch 翻译成可执行的张量。
- 先做单卡 eager 版，是为了在阶段 9（TP）和阶段 10（CUDA Graph）之前，建立一个**正确性可验证的基线**。

## 3. 三大支柱任务

### 3.1 初始化与生命周期
`__init__`（init_process_group + 创建模型 + 加载权重 + warmup + allocate_kv_cache）、`exit`（清理）。

### 3.2 输入张量构造
`prepare_prefill / prepare_decode / prepare_block_tables / prepare_sample`：把 Sequence 列表转成 input_ids / positions / cu_seqlens / slot_mapping / block_tables / temperatures。

### 3.3 单步执行
`run(seqs, is_prefill)`：调 prepare_xxx → 设置 Context → run_model → 采样 → reset_context → 返回 token_ids。

## 4. 验收标准

- [ ] `Sequence([1,2,3,4,5,6,7,8])` 走完 `run([seq], is_prefill=True)` 能返回一个合理的 token_id（与 HF 推理 top-1 一致）
- [ ] 紧接着调 `run([seq], is_prefill=False)` 多次能持续生成，结果稳定
- [ ] `allocate_kv_cache` 后能正确拿到 `num_kvcache_blocks > 0`，且 KV Cache 显存占用接近 `gpu_memory_utilization` 上限

---

## 5. 练习题

### 练习 7.1（__init__：分布式与模型加载）
实现 [`engine/model_runner.py`](nanovllm/engine/model_runner.py) 的 `__init__(config, rank, event)` 单卡版（rank=0, world_size=1）：
1. `dist.init_process_group("nccl", "tcp://localhost:2333", world_size=1, rank=0)` —— 即使单卡也初始化，让 layers 里的 `dist.get_rank() / get_world_size()` 工作
2. `torch.cuda.set_device(rank)`
3. **dtype 切换**（关键）：
   ```python
   default_dtype = torch.get_default_dtype()
   torch.set_default_dtype(hf_config.dtype)
   torch.set_default_device("cuda")
   self.model = Qwen3ForCausalLM(hf_config)   # 此时所有 nn.Parameter 都默认 bf16+cuda
   load_model(self.model, config.model)
   ```
4. `self.sampler = Sampler()`
5. `self.warmup_model()` —— 先用最大 batch 跑一次 prefill 触发 torch.compile
6. `self.allocate_kv_cache()` —— 算可用块数并分配
7. 恢复 default device/dtype 到 cpu/float（避免影响外面）

### 练习 7.2（warmup_model）
```python
def warmup_model(self):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    seq_len = min(self.config.max_num_batched_tokens, self.config.max_model_len)
    num_seqs = min(self.config.max_num_batched_tokens // seq_len, self.config.max_num_seqs)
    seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
    for seq in seqs:
        seq.num_scheduled_tokens = seq_len
    self.run(seqs, True)               # 预跑一次 prefill
    torch.cuda.empty_cache()
```
- 目的 1：触发 `@torch.compile` 编译（首次会慢，提前编译避免线上冷启动）
- 目的 2：精准测出模型与中间 buffer 的峰值显存，给 allocate_kv_cache 算可用空间

### 练习 7.3（allocate_kv_cache）
计算可用显存 → 反推可用块数 → 分配 5D 大张量 → 切片塞回每层 Attention：
```python
def allocate_kv_cache(self):
    free, total = torch.cuda.mem_get_info()
    used = total - free
    peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
    current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
    
    num_kv_heads = hf_config.num_key_value_heads // self.world_size
    head_dim = getattr(hf_config, "head_dim", hidden_size // num_attention_heads)
    block_bytes = 2 * num_layers * block_size * num_kv_heads * head_dim * dtype_size
    
    available = total * gpu_memory_utilization - used - peak + current
    config.num_kvcache_blocks = int(available) // block_bytes
    
    self.kv_cache = torch.empty(2, num_layers, num_blocks, block_size, num_kv_heads, head_dim)
    
    # 把每层 Attention 的 k_cache / v_cache 指向对应切片
    layer_id = 0
    for module in self.model.modules():
        if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
            module.k_cache = self.kv_cache[0, layer_id]
            module.v_cache = self.kv_cache[1, layer_id]
            layer_id += 1
```
- `peak - current` 是「会用峰值多少显存」，从可用空间扣掉

### 练习 7.4（prepare_prefill）
拼出 prefill 路径需要的 6 个张量：
- `input_ids`: list 拼接，shape `[total_q]`
- `positions`: 每个序列的 [start, start+seqlen_q) 拼接
- `cu_seqlens_q`: 累计前缀和，shape `[B+1]`
- `cu_seqlens_k`: 同上但用 seqlen_k = num_cached + num_scheduled = end
- `max_seqlen_q / max_seqlen_k`: int
- `slot_mapping`: 每个 token 在 KV Cache 中的 1D 偏移
- `block_tables`: 仅当存在前缀缓存（`cu_seqlens_k[-1] > cu_seqlens_q[-1]`）时构造

特别留意 slot_mapping 的循环（参考阶段 3 的解答 3.2）：
```python
for i in range(start_block, end_block):
    slot_start = block_table[i] * block_size
    if i == start_block: slot_start += start % block_size
    slot_end = block_table[i] * block_size + (block_size if i != end_block-1 else end - i*block_size)
    slot_mapping.extend(range(slot_start, slot_end))
```
最后用 `torch.tensor(..., pin_memory=True).cuda(non_blocking=True)` 异步搬到 GPU，再 `set_context(...)`。

### 练习 7.5（prepare_decode）
更简单：每序列 1 token：
- `input_ids = [seq.last_token for seq in seqs]`
- `positions = [len(seq) - 1 for seq in seqs]`
- `slot_mapping = [block_table[-1] * block_size + last_block_num_tokens - 1 for seq in seqs]`
- `context_lens = [len(seq) for seq in seqs]`
- `block_tables = prepare_block_tables(seqs)`（pad 到 max_len）

### 练习 7.6（prepare_block_tables 与 prepare_sample）
- block_tables：对每个 seq 把 block_table 用 `-1` pad 到 max_len，转 int32
- prepare_sample：`temperatures = torch.tensor([seq.temperature for seq in seqs])`

### 练习 7.7（run / run_model）
最小可用版（不做 cudagraph 加速）：
```python
@torch.inference_mode()
def run_model(self, input_ids, positions, is_prefill):
    return self.model.compute_logits(self.model(input_ids, positions))

def run(self, seqs, is_prefill):
    input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
    temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
    logits = self.run_model(input_ids, positions, is_prefill)
    token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
    reset_context()
    return token_ids
```

### 练习 7.8（端到端 demo）
不接 LLMEngine、Scheduler，先验证 ModelRunner 自己能用：
```python
config = Config(model="...", enforce_eager=True)
runner = ModelRunner(config, 0, [])
seq = Sequence(tokenizer.encode("Hello"))
seq.num_scheduled_tokens = len(seq)
# 手动给 seq 分配 block_table —— 临时用 BlockManager
bm = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
n = bm.can_allocate(seq); bm.allocate(seq, n)

token = runner.run([seq], is_prefill=True)[0]
seq.append_token(token)
for _ in range(20):
    if not bm.can_append(seq): break
    bm.may_append(seq)
    token = runner.run([seq], is_prefill=False)[0]
    seq.append_token(token)
print(tokenizer.decode(seq.completion_token_ids))
```

---

## 6. 解答

> 参考实现：[`engine/model_runner.py`](nanovllm/engine/model_runner.py)

### 解答 7.1 关键点
- `torch.set_default_dtype(hf_config.dtype)` + `set_default_device("cuda")` 让 `torch.empty(...)` 默认产生 bf16+cuda 张量 —— Qwen3 的 Parameter 直接被分配在 GPU 上的 bf16，省掉 `.to(...)`。
- `init_process_group` 必须最早调，否则 layers 内的 `dist.get_rank()` 报错。
- 注意 init 完成后**复原 default device/dtype**到 cpu/float，避免外部代码受影响。

### 解答 7.2 关键点
- warmup 用零 token 跑 prefill 是 OK 的（forward 不会因为 token=0 出 NaN，只是结果无意义）。
- warmup 后立刻 `empty_cache`，让 allocator 把刚刚 peak 的临时 buffer 还回 caching pool —— 反映在 `peak - current` 的差里。

### 解答 7.3 关键点
- `block_bytes = 2 * num_layers * block_size * num_kv_heads * head_dim * dtype_size`：2 是 K/V，num_kv_heads 已经按 TP 切分。
- 找 Attention 模块用 `hasattr(module, "k_cache")`：Attention 类初始化时给 `k_cache = torch.tensor([])` 占位，这里就有这个属性。
- 切片赋值后，原 Attention 类内的 `k_cache.numel() > 0` 自动成立 → forward 里的 store_kvcache 会启用。

### 解答 7.4 关键点（prefill）
- 「前缀缓存命中」的判定：`cu_seqlens_k[-1] > cu_seqlens_q[-1]` 意味着 K 序列总长度超过 Q（有些 K 是已缓存的） → 必须把 block_tables 传给 flash-attn 让它知道去 KV Cache 里取。
- pin_memory + non_blocking：把 CPU tensor 用 pinned buffer 异步搬到 GPU，与下一行的张量构造重叠。
- warmup 时序列没分 block_table（`if not seq.block_table: continue`），所以 slot_mapping 跳过 —— 此时 store_kvcache 也不会生效（因为 k_cache.numel()==0）。

### 解答 7.5 关键点（decode）
- `slot = block_table[-1] * block_size + last_block_num_tokens - 1`：
  - 例：block_size=4, block_table=[2, 7], 当前 last_block_num_tokens=2（block 7 已用 2 slot），slot = 7*4 + 2 - 1 = 29 → 等等，这里要小心 -1 的位置。
  - 实际：last_block_num_tokens 是「填了几个」，新 token 写到第 (last_block_num_tokens) 位（0-indexed 是 last_block_num_tokens - 0），但 may_append 在 schedule 时已经追加过新块（如果跨块），所以 last_token 写到 last_block 的最末位 - 1 处？
  - 仔细看原代码：`slot = seq.block_table[-1] * block_size + seq.last_block_num_tokens - 1`。
  - 解释：当我们 prepare_decode 时，**这个 token 已经被 append_token**（postprocess 在上一步调过），num_tokens 已经包含它，所以 last_block_num_tokens 就是「该 token 在最后块中是第几个」（从 1 数起），减 1 转成 0-index 就是它的 slot 偏移。

### 解答 7.6 关键点
- `prepare_block_tables` 用 `-1` 作为 pad：FlashAttention 的 paged kernel 会把 -1 视作无效（不读 KV）。
- 温度只在 rank0 准备：因为采样发生在 rank0，其他 rank 不用。

### 解答 7.7 关键点
- `@torch.inference_mode()` 比 `no_grad` 多关掉 view tracking，更适合纯推理。
- 单卡 eager 路径不需要 cudagraph 分支，直接 `model(...).compute_logits(...)`。
- 注意 `set_context` 已经在 prepare_xxx 里调过，run_model 后 `reset_context` 清理。

### 解答 7.8 端到端
- 期望输出：tokenizer.decode 后是一段合理的接续文本（可能是乱的，因为温度=1.0 + 单 prompt）。
- 如果输出全是同一个 token：检查 sampler 的 `temperatures` 是否传成 GPU tensor、是否是 float32。

### 常见坑
1. **dtype 没切到 bf16 就建模型** → Parameter 是 fp32，加载 bf16 权重时 dtype 不匹配。
2. **忘了把 KV Cache 切片塞给每层 Attention** → store_kvcache 看到 numel=0 跳过，decode 时 attention 用空 cache → 结果全错。
3. **slot_mapping 的 int dtype 不对** → Triton kernel 用 int32 索引；用 int64 会触发隐式转换或报错。
4. **prepare_prefill 在 warmup 路径上构造 slot_mapping** → 必须有 `if not seq.block_table: continue` 保护，不然访问 `seq.block_table[i]` 会越界。

---

## 7. 自检提问

- [ ] 我能解释 `torch.set_default_dtype(bf16)` 在 `__init__` 里的作用
- [ ] 我能复述 allocate_kv_cache 计算可用块数的公式（total*util - used - peak + current）/ block_bytes
- [ ] 我能讲清 prefill 与 decode 的 slot_mapping 公式差别
- [ ] 我能解释 `cu_seqlens_k[-1] > cu_seqlens_q[-1]` 这个判断的含义
- [ ] 我能在不接 LLMEngine 的情况下手工跑通 ModelRunner 的 prefill+decode 循环