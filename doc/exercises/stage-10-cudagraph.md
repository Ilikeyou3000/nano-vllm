# 阶段 10 · CUDA Graph 优化

## 1. 结论

**为常见的 decode batch_size 提前「录制」一遍 forward 调用图，运行时直接 replay，省掉 Python / CUDA 调度开销，decode 吞吐能提升 30%~50%。**

## 2. 为什么这么做

- decode 阶段每步只 1 个 token，kernel 本身极快（μs 级），但 Python + PyTorch dispatch + CUDA launch 的开销也是 μs 级 —— **launch 开销与计算时间相当**。
- CUDA Graph 把一系列 kernel launch 录成一张图，replay 时 GPU 直接顺序执行，不再过 Python/Driver。
- prefill 阶段 token 数大、形状多变，不适合 Graph；只录 decode。

## 3. 三大支柱任务

### 3.1 准备静态 buffer
预分配最大 batch 的 `input_ids / positions / slot_mapping / context_lens / block_tables / outputs` 张量，所有 graph 共享同一组 buffer。

### 3.2 多 batch_size 录制
为 `[1, 2, 4, 8, 16, 24, 32, ..., max_bs]` 这一组离散值分别 capture 一张图，存入 `self.graphs[bs]` 字典；共享同一 `graph_pool` 节省显存。

### 3.3 运行时 dispatch
forward 时若是 decode 且 bs ≤ max_bs，把实际数据 copy_ 进静态 buffer，找到 `>=bs` 的最小已捕获图 → replay → 取出 outputs 的前 bs 行。

## 4. 验收标准

- [ ] enforce_eager=False 跑通 example.py，输出与 enforce_eager=True 一致
- [ ] decode 吞吐相对 eager 提升 ≥ 25%
- [ ] capture 阶段不 OOM（graph_pool 复用生效）

---

## 5. 练习题

### 练习 10.1（理解 CUDA Graph 限制）
回答以下问题（自查）：
- CUDA Graph capture 期间不能有 CPU↔GPU 同步（`.item()` / `.cpu()`）—— 为什么？
- Graph 录的是 kernel 序列还是张量值？replay 时输入怎么变？
- 为什么 prefill 不适合 Graph？

### 练习 10.2（静态 buffer 设计）
在 [`engine/model_runner.py`](nanovllm/engine/model_runner.py) `__init__` 末尾分配：
```python
max_bs = 512
max_blocks = (max_model_len + block_size - 1) // block_size
self.input_ids   = torch.zeros(max_bs, dtype=torch.int64,  device="cuda")
self.positions   = torch.zeros(max_bs, dtype=torch.int64,  device="cuda")
self.slot_mapping= torch.zeros(max_bs, dtype=torch.int32,  device="cuda")
self.context_lens= torch.zeros(max_bs, dtype=torch.int32,  device="cuda")
self.block_tables= torch.zeros(max_bs, max_blocks, dtype=torch.int32, device="cuda")
self.outputs     = torch.zeros(max_bs, hidden_size, dtype=torch.bfloat16, device="cuda")
```
这些 buffer 在 capture 时被「绑定」进图，replay 必须复用它们（指针不变）。

### 练习 10.3（捕获循环）
新增 `capture_cudagraph()` 方法：
```python
self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
self.graphs = {}
self.graph_pool = None

for bs in reversed(self.graph_bs):              # 从大到小：先大的能让池足够大
    graph = torch.cuda.CUDAGraph()
    set_context(is_prefill=False, slot_mapping=self.slot_mapping[:bs],
                context_lens=self.context_lens[:bs], block_tables=self.block_tables[:bs])
    self.model(self.input_ids[:bs], self.positions[:bs])  # warmup
    with torch.cuda.graph(graph, self.graph_pool):
        out = self.model(self.input_ids[:bs], self.positions[:bs])
    if self.graph_pool is None:
        self.graph_pool = graph.pool()
    self.graphs[bs] = (graph, out)              # 记下输出张量地址，replay 后从这里读
    reset_context()
```
- 关键：必须先做一次 warmup forward（让 lazy alloc 完成），再 capture，否则 graph 录到的 alloc 路径不能 replay。
- `graph_pool` 第一次 capture 后保存，之后所有 capture 都传入同一个 pool → 内存复用。

### 练习 10.4（运行时 dispatch）
修改 `run_model(input_ids, positions, is_prefill)`：
```python
if is_prefill or self.enforce_eager or input_ids.size(0) > self.graph_bs[-1]:
    return self.model.compute_logits(self.model(input_ids, positions))

bs = input_ids.size(0)
graph_bs = next(x for x in self.graph_bs if x >= bs)
graph, out_buf = self.graphs[graph_bs]

# 把真实数据 copy 进 静态 buffer（前 bs 行）
ctx = get_context()
self.input_ids[:bs].copy_(input_ids)
self.positions[:bs].copy_(positions)
self.slot_mapping[:graph_bs].zero_()
self.slot_mapping[:bs].copy_(ctx.slot_mapping)
self.context_lens[:graph_bs].zero_()
self.context_lens[:bs].copy_(ctx.context_lens)
self.block_tables[:graph_bs, :ctx.block_tables.size(1)].copy_(ctx.block_tables)

graph.replay()
return self.model.compute_logits(out_buf[:bs])
```
- padding 部分的 slot_mapping 设为 0 / context_lens 设为 0 不会读取真实 KV → 安全。
- 选 `>=bs` 的最小图：避免对每个 bs 都录图，离散化节省显存。

### 练习 10.5（attention kernel 兼容）
检查 [`layers/attention.py`](nanovllm/layers/attention.py) 中 decode 分支用的是 `flash_attn_with_kvcache`：
- 它支持 graph capture？答：是，因为它不做 CPU 同步、不做 dynamic shape。
- store_kvcache 的 Triton kernel 也能进 graph（但 prefill 路径才用，decode 用 with_kvcache 自带的写入）。

### 练习 10.6（端到端验证）
- 跑 example.py 在 enforce_eager=True 与 False 下，输出 token 应一致（贪心或 fix seed）。
- 用 `time.perf_counter` 测 decode-only 吞吐（构造 100 个长度=10 的 prompt 让其 decode 200 步），eager vs graph 对比。

### 练习 10.7（剖析 capture 显存）
- capture 前后用 `torch.cuda.memory_allocated()` 算差值。
- 不传 `graph_pool` 跑一次 → 显存暴涨；传 graph_pool 复用 → 显存基本不变。理解为什么。

---

## 6. 解答

> 参考实现：[`engine/model_runner.py`](nanovllm/engine/model_runner.py) 的 `capture_cudagraph` 与 `run_model`。

### 解答 10.1（限制原理）
- CUDA Graph 录的是 **kernel + 内存读写依赖**，不录 Python 控制流；任何 CPU↔GPU 同步会破坏 capture 模式。
- 张量地址被绑定：replay 时只能写入同一指针的 buffer；要换数据必须 `copy_` 到原 buffer 中。
- prefill：每次 token 数都不同，需要为每个长度录一张图 → 不现实；decode 每步只有 1 token，bs 才是变量。

### 解答 10.2（buffer）
- 用 max_bs=512 是经验值，覆盖大部分场景；超出走 eager 兜底。
- block_tables 第二维 = 最长序列对应的块数，确保最差情况能装下。

### 解答 10.3（capture）
- **从大到小 capture**：CUDA Graph 内存池一旦确定就不能扩，从大 bs 开始能让池足够大；之后小 bs 的图复用同一池。
- warmup 必须做：第一次跑 forward 会触发 cuBLAS workspace 等 lazy 分配，这些不能进图。
- `with torch.cuda.graph(graph, pool):` 是 capture 上下文，期间所有 CUDA 操作进图。

### 解答 10.4（dispatch）
- bs 选择：`next(x for x in self.graph_bs if x >= bs)` —— 因为 graph_bs 已排序，第一个 ≥bs 的就是最小可用图。
- padding 区写零的安全性：FlashAttention 的 with_kvcache 在 context_len=0 时不会读取 KV；slot_mapping=0 不会写真实位置（但需要保证 KV cache 第 0 个槽不被业务占用 —— nano-vllm 中 block 0 是哑块）。
- 输出 `out_buf[:bs]`：取静态 buffer 的前 bs 行作为真实结果。

### 解答 10.5（兼容性）
- FlashAttention with_kvcache 内部不做 CPU sync，可安全 capture。
- 如果用了带 if/else 分支的 attention kernel（按 cache_seqlen 跳转），仍然 OK，因为分支在 GPU 内部决策。

### 解答 10.6（验证）
- 输出一致性：贪心（temperature=0）下逐 token 完全相同；含温度时统计相似。
- 性能：decode-only 场景下 graph 通常带来 30%+ 的吞吐提升，越是大 bs 收益越大（kernel 时间增长但 launch 开销不变 → 占比减小）→ 但小 bs 收益更显著（launch 开销主导）。

### 解答 10.7（显存）
- 不复用池：每个 graph 都申请一份独立 workspace（attention/linear 中间张量）→ N 张图 = N 份显存。
- 复用 graph_pool：所有图共享同一份 workspace，因为同一时刻只有一张图在 replay → 安全节省 N-1 份。

### 常见坑
1. **忘记 warmup** → capture 时报「stream 在 capture 模式下不能 alloc」错误。
2. **dispatch 时 bs 已经超过 max(graph_bs)** → KeyError；务必加上 fallback 到 eager 的判断。
3. **block_tables 第二维变化** → 当某个 seq 突然分配了新 block，block_tables 维度变长但静态 buffer 是固定宽度 → 用 `.copy_(ctx.block_tables)` 时切片要对齐。
4. **enforce_eager=True 也跑 capture** → 浪费显存还慢；务必判断这个标志。
5. **Triton kernel 在 capture 下行为异常** → 部分早期 Triton 版本不兼容 graph capture，升级到 ≥2.1。

---

## 7. 自检提问

- [ ] 我能解释 CUDA Graph 为什么对 decode 有效、对 prefill 不适合
- [ ] 我能复述 capture 流程：warmup → capture → 记录 buffer 指针 → replay
- [ ] 我能解释 graph_pool 复用为什么能节省显存
- [ ] 我能算出 graph_bs 列表大约消耗多少额外显存（capture 时静态 buffer + workspace）
- [ ] 我能解释为什么 padding 区域填 0 是安全的