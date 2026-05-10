# 阶段 9 · Tensor Parallelism

## 1. 结论

**让单卡可正确运行的引擎升级为多卡 TP：每张卡只持有 1/N 模型权重与 1/N KV Cache，通过 `dist.all_reduce / gather` 协作完成 forward；rank0 通过共享内存广播命令给其他 worker。**

## 2. 为什么这么做

- Qwen3 0.6B 单卡能跑，但更大的模型（7B/14B/32B）必须 TP 才能放下。
- TP 的难点不在算子（线性代数已经被论文讲透），而在**工程**：进程间命令分发、权重 shard 加载、采样的位置选择、共享内存生命周期。
- 这一阶段要把 ModelRunner 拆成「主控 rank0 + 多个 worker rank>0」，是从「玩具」到「能用」的关键跨越。

## 3. 三大支柱任务

### 3.1 算子层 TP 升级
让 `ColumnParallelLinear / RowParallelLinear / VocabParallelEmbedding / ParallelLMHead` 在 tp_size>1 时正确 shard 与通信。

### 3.2 多进程启动 + 命令分发
LLMEngine 在 rank0 启动 N-1 个 worker 进程；rank0 与 workers 通过 `SharedMemory + Event` 同步执行同一方法。

### 3.3 KV Cache 与采样的 TP 适配
KV Cache 头数除以 TP；采样只在 rank0 做。

## 4. 验收标准

- [ ] tp_size=2 跑 example.py，输出与 tp_size=1 一致（除随机性差异，温度=0.6 时 top-1 应高度重合）
- [ ] tp_size=2 时显存占用 ≈ 单卡的一半（每卡只放半模型）
- [ ] Ctrl-C 后所有 worker 进程都能退出（无僵尸）

---

## 5. 练习题

### 练习 9.1（ColumnParallelLinear 完整版）
升级 [`layers/linear.py`](nanovllm/layers/linear.py) 的 `ColumnParallelLinear`：
- `__init__` 时 `output_size = divide(output_size, tp_size)` —— 每个 rank 只放 1/N
- `weight_loader(param, loaded_weight)`：
  - `shard_size = param.size(tp_dim)`
  - `start = tp_rank * shard_size`
  - `loaded_weight = loaded_weight.narrow(tp_dim, start, shard_size)`
  - `param.copy_(loaded_weight)`
- forward 不变（`F.linear(x, weight, bias)`）—— 输出维度已经是切分过的 1/N

### 练习 9.2（RowParallelLinear 完整版）
- `__init__` 时 `input_size = divide(input_size, tp_size)`
- weight_loader 沿 tp_dim=1（输入维度）切片
- forward：
  ```python
  y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
  if self.tp_size > 1:
      dist.all_reduce(y)
  return y
  ```
- 注意 bias 只加在 rank0：因为 all_reduce 后会把所有 rank 的 partial 和加起来，bias 多次相加就错了。

### 练习 9.3（QKVParallelLinear / MergedColumnParallelLinear）
- QKV：`weight_loader(param, loaded_weight, shard_id)`：
  - 计算 q/k/v 各自的 shard_size（已经按 tp_size 除过）
  - 在 param 上 narrow 到对应位置
  - 在 loaded_weight 上 `chunk(tp_size, dim=tp_dim)[tp_rank]` 切自己那份
  - 写入
- MergedColumnParallel：类似，但 shard_id 是整数（gate=0, up=1）

### 练习 9.4（VocabParallelEmbedding）
升级：
- `num_embeddings_per_partition = num_embeddings // tp_size`
- 每个 rank 持有 vocab 的 `[start, end)` 段
- forward：
  ```python
  if tp_size > 1:
      mask = (x >= vocab_start) & (x < vocab_end)
      x = mask * (x - vocab_start)        # 不在本 rank 的 token 强制查 0 行（后会被 mask 清零）
  y = F.embedding(x, weight)
  if tp_size > 1:
      y = mask.unsqueeze(1) * y           # 不在本 rank 的位置清零
      dist.all_reduce(y)                  # 所有 rank 汇总
  return y
  ```

### 练习 9.5（ParallelLMHead）
- 每个 rank 持有 1/N 的 vocab，forward 后 logits shape `[B, vocab/N]`
- 用 `dist.gather` 把所有 rank 的 logits 收集到 rank0：
  ```python
  if tp_size > 1:
      all_logits = [torch.empty_like(logits) for _ in range(tp_size)] if tp_rank == 0 else None
      dist.gather(logits, all_logits, 0)
      logits = torch.cat(all_logits, -1) if tp_rank == 0 else None
  return logits   # rank0 拿到完整 [B, vocab]，其他 rank 拿到 None
  ```

### 练习 9.6（多进程启动）
升级 [`engine/llm_engine.py`](nanovllm/engine/llm_engine.py)：
```python
ctx = mp.get_context("spawn")
self.ps, self.events = [], []
for i in range(1, tp_size):
    event = ctx.Event()
    p = ctx.Process(target=ModelRunner, args=(config, i, event))
    p.start()
    self.ps.append(p); self.events.append(event)
self.model_runner = ModelRunner(config, 0, self.events)   # rank0 持有所有 events
```
- 用 `spawn` 而非 fork，避免 CUDA 上下文复制问题
- 每个 worker 拿一个 Event，rank0 写完共享内存后 `event.set()` 通知

### 练习 9.7（共享内存命令通道）
升级 [`engine/model_runner.py`](nanovllm/engine/model_runner.py) 中的：
- `__init__`（rank0 vs rank>0 分支）：
  ```python
  if world_size > 1:
      if rank == 0:
          self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
          dist.barrier()                  # 让 worker 启动后能找到
      else:
          dist.barrier()
          self.shm = SharedMemory(name="nanovllm")
          self.loop()                     # worker 进入命令循环
  ```
- `write_shm(method, *args)`：rank0 把 pickled 命令写入 shm，前 4 字节存长度，然后 set 所有 event
- `read_shm()`：worker 等 event，从 shm 读命令，clear event
- `loop()`：worker 死循环 read_shm → call(method, *args)；method=="exit" 时退出
- `call(method, *args)`：rank0 时先 write_shm 给 workers，自己也跑 method；非 rank0 直接跑

### 练习 9.8（Sequence pickle 优化兑现）
回顾阶段 1.6 的 `__getstate__/__setstate__`：在 TP 模式下，每一步 schedule 都会把 seqs 通过 shm 发给 workers。
- decode 阶段每个 seq 只传 last_token + meta，省下 token_ids 的传输
- 写一个 micro-benchmark：分别测「pickle 完整 token_ids」与「优化版」的 throughput 差距

### 练习 9.9（TP 端到端）
跑通 tp_size=2 的 example.py：
```python
llm = LLM(path, tensor_parallel_size=2, enforce_eager=True)
```
对比 tp_size=1 输出是否一致（应一致；细微浮点差不影响 argmax）。

---

## 6. 解答

> 参考实现：[`layers/linear.py`](nanovllm/layers/linear.py)、[`layers/embed_head.py`](nanovllm/layers/embed_head.py)、[`engine/llm_engine.py`](nanovllm/engine/llm_engine.py)、[`engine/model_runner.py`](nanovllm/engine/model_runner.py)

### 解答 9.1~9.3（Linear TP）
- 通用准则：**ColumnParallel 切 output 维（dim=0）、RowParallel 切 input 维（dim=1）**。
- 为什么 RowParallel 要 all_reduce 而 ColumnParallel 不要？
  - Column：每个 rank 算自己的 1/N output，concat 起来才是完整的 → 但 Qwen3 中 ColumnParallel 后接的 RowParallel 直接吃这 1/N → 不需要 concat。
  - Row：每个 rank 拿到的是完整 hidden 的某 1/N 维 → 部分和 → 必须 all_reduce 求总和才得到 output。
- bias 只加 rank0：避免 all_reduce 时 bias 被加 N 次。

### 解答 9.4（VocabParallelEmbedding）
- mask 技巧：先把不在本 rank 范围的 token id 强制变成 0（避免越界），再用 mask 把这些位置的 embedding 输出清零，最后 all_reduce 汇总。
- all_reduce 后，每个不在本 rank 的位置由别人 rank 提供完整结果；本 rank 提供本 rank 的 → sum 等于完整 embedding。

### 解答 9.5（LMHead）
- 用 `gather` 而非 `all_gather`：因为只有 rank0 需要完整 logits 做采样，其他 rank 不需要。
- gather 的 dst=0 表示「集中到 rank0」，其他 rank 收到 None。
- rank0 收到 list 后 `cat(dim=-1)` 拼成 [B, vocab_size]。

### 解答 9.6（多进程）
- spawn vs fork：CUDA 上下文不能被 fork（CUDA 不支持），必须 spawn 重新初始化。
- worker 内部 ModelRunner.__init__ 自己 init_process_group，rank0 通过 dist.barrier 等所有人就位。

### 解答 9.7（共享内存）
- size=2^20 (1MB) 通常够装一个 step 的命令（seqs 列表 pickle 后几十 KB）。
- 用前 4 字节存 pickled 数据长度，后续读取时按长度截取。
- Event 机制：rank0 set → workers wait 唤醒 → workers 处理完后 clear → rank0 下一轮再 set。
- `dist.barrier()` 用来同步「shm 创建好了 vs workers 已 attach 了」这个时序：rank0 创建后 barrier 等 workers，workers attach 后 barrier 通过 → 双方都到这个点才继续。

### 解答 9.8（Sequence pickle）
- 阶段 1 写的优化在这里发挥价值：原始 token_ids 可能几千个 int，每步广播一次 = 浪费几 MB；优化后单个 seq 只 100 字节左右。
- 验证方式：用 `pickle.dumps(seq)` 算 size，对比 `is_prefill=True` 与 `is_prefill=False`。

### 解答 9.9（端到端）
- 期望：tp=2 时 KV Cache 块数大约**翻倍**（每卡放半模型 → 单卡剩余空间更多 → 块数更多）。
- 输出对比：固定 sampling_params(temperature=0.6) 时，因为采样有随机性，输出可能不完全一致；但 tp=1 与 tp=2 的吞吐与 token 分布应统计上相似。

### 常见坑
1. **dist.init_process_group 在 worker 里被调了两次** → ModelRunner 自己 init，但你又在 LLMEngine 里 init → 报错。规约：只让 ModelRunner 负责。
2. **shm 名字冲突** → 第二次启动时 `name="nanovllm"` 已存在 → 报错。在 exit 中正确 unlink，或加随机后缀。
3. **gather 后 rank>0 拿 None，但你又对 None 做操作** → 严格按 `if rank == 0:` 分支处理。
4. **dist.barrier() 时序错乱** → 创建 shm 与 attach shm 必须用 barrier 同步，否则可能 attach 时 shm 还没创建好。
5. **Sequence.counter 在 worker 里被 fork** → 用 spawn 就没问题，因为子进程是新解释器实例，counter 重置。但只在 rank0 add_request 即可保证唯一性。

---

## 7. 自检提问

- [ ] 我能解释 ColumnParallel 与 RowParallel 各自切哪一维、哪个需要 all_reduce
- [ ] 我能讲清 VocabParallelEmbedding 的 mask + all_reduce 妙处
- [ ] 我能复述 spawn 进程 + SharedMemory + Event 的命令分发流程
- [ ] 我能解释 dist.barrier() 在 shm 创建/attach 中的时序作用
- [ ] 我能算出 tp_size=2 时 KV Cache 块数大约相对单卡的变化（更多还是更少？为什么？）