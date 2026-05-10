# 阶段 3 · Context 与 KV Cache 算子

## 1. 结论

**实现一个全局 `Context`，写一个 Triton kernel 把 K/V 写进分页 KV Cache，再用 FlashAttention 完成 prefill / decode 两条注意力路径。**

## 2. 为什么这么做

- 阶段 2 的算子都是**无状态**的；但 Attention 必须读写 KV Cache，是**整个引擎中唯一与 GPU 内存布局深度耦合的算子**。
- 使用「全局 Context」存放 prefill/decode 的元信息，可以让 Attention forward 不必修改其他模块的接口。
- Triton kernel 是 vLLM 风格 KV 写入的最简实现，不掌握就只能用慢速 PyTorch 索引。
- 这一阶段做完后，单层 Attention 就能跑出和 HuggingFace 一致的输出，是阶段 4 拼装 Qwen3 的基础。

## 3. 三大支柱任务

### 3.1 全局 Context
`Context` dataclass + `get_context / set_context / reset_context` 三函数。

### 3.2 KV 写入 kernel
`store_kvcache` —— 用 Triton 把每个 token 的 K/V 写到块内对应 slot。

### 3.3 Attention forward
- prefill：`flash_attn_varlen_func`（带可选 block_table 用于前缀 cache）
- decode：`flash_attn_with_kvcache`

## 4. 验收标准

- [ ] 单层 `Attention` 在「prefill 一段 + decode 若干步」流程后，输出与 HuggingFace `Qwen3Attention` 在相同输入下对齐（误差 < 5e-3）
- [ ] `slot == -1` 的 token 不会被写入 cache（CUDA Graph padding 兼容）
- [ ] 前缀 cache 路径下 `flash_attn_varlen_func(block_table=...)` 调用不报错

---

## 5. 练习题

### 练习 3.1（Context dataclass）
实现 [`utils/context.py`](nanovllm/utils/context.py)：
```python
@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
```
配套实现 `get_context()`、`set_context(...)`、`reset_context()`，**用模块级全局变量保存当前 context**。

> 思考：为什么不用线程局部变量？答：CUDA Graph 捕获时是单线程，单 ModelRunner 进程内全局变量足够，且简单。

### 练习 3.2（slot_mapping 概念）
回答：
- 设 block_size=4，序列 A 的 block_table=[2, 7]（已经填了 6 个 token），现在要 decode 第 7 个 token：它的 slot 应该是哪个数？
- prefill 时一段 [4, 8) 范围的 token，跨 block 时如何把整段 slot 一次性算出来？写出公式。

### 练习 3.3（Triton kernel）
实现 [`layers/attention.py`](nanovllm/layers/attention.py) 中的 `store_kvcache_kernel`：
- 每个程序 id 对应一个 token
- 拿到该 token 的 slot；若 `slot == -1` 直接 `return`（CUDA Graph padding）
- 计算偏移：`key_offsets = idx * key_stride + tl.arange(0, D)`
- 加载 key/value 后写到 `k_cache[slot * D + ...]`、`v_cache[slot * D + ...]`
- 用 `D: tl.constexpr` 让编译器特化

外层 launcher `store_kvcache(key, value, k_cache, v_cache, slot_mapping)`：
- 检查 stride 满足要求（`key.stride(-1)==1`、`key.stride(1)==head_dim`、`k_cache.stride(1)==D`）
- `D = num_heads * head_dim`，每个 token 的 K/V flatten 后是 D 个元素
- 调用 `store_kvcache_kernel[(N,)](...)`

### 练习 3.4（Attention forward）
实现 `Attention.forward(q, k, v)`：
1. 从 `get_context()` 拿元信息
2. 如果 `k_cache.numel() > 0`：调用 `store_kvcache(k, v, k_cache, v_cache, slot_mapping)`
3. 分支：
   - `is_prefill`：
     - 若 `block_tables is not None` → 前缀 cache 命中，`k = k_cache, v = v_cache`，把 block_table 传给 flash-attn
     - 否则 `k, v` 用入参
     - 调 `flash_attn_varlen_func(q, k, v, max_seqlen_q, cu_seqlens_q, max_seqlen_k, cu_seqlens_k, softmax_scale, causal=True, block_table=block_tables)`
   - decode：
     - `flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache, cache_seqlens=context_lens, block_table=block_tables, softmax_scale, causal=True)`
4. 返回 attention 输出（形状 `[total_tokens, num_heads, head_dim]`）

### 练习 3.5（KV Cache 内存布局）
解释下面这个 5D 张量每个维度的含义：
```python
self.kv_cache = torch.empty(2, num_layers, num_blocks, block_size, num_kv_heads, head_dim)
```
画图说明：当 slot=block_id*block_size+offset 时，从 5D 张量到「flatten 视角下的 D 维一维偏移」是怎么对应上的（这是 store_kvcache kernel 能用 1D 偏移直接写的关键）。

### 练习 3.6（端到端验证）
写一个测试：
1. 构造一个长度 64 的随机 input_ids；
2. 用 HuggingFace `Qwen3Model` 跑一次 prefill，记录最后一个 hidden_state；
3. 用你的 Attention（在阶段 4 拼好之前可以先用一个简化的 1-layer 模型）跑同样输入；
4. 对比两个 hidden_state 误差。

实在等不及阶段 4，可以**仅替换** `Qwen3Model` 内部的 `Qwen3Attention` 为你的实现，跑一层验证。

---

## 6. 解答

> 参考实现：[`utils/context.py`](nanovllm/utils/context.py)、[`layers/attention.py`](nanovllm/layers/attention.py)

### 解答 3.1（Context）
- 模块级全局 `_CONTEXT = Context()`，提供 setter / getter / reset。
- `set_context` 用 `*default=...` 形参一次性接所有字段，调用方按需填。
- 注意 `cu_seqlens_*` 用 `torch.int32`、`slot_mapping` 用 `int32`：FlashAttention 与 Triton 要求。

### 解答 3.2（slot_mapping）
- block_table=[2, 7]，block_size=4，已填 6 个 token：
  - block 2 用了 4 个 slot，block 7 用了 2 个 slot
  - 第 7 个 token 要写到 block 7 的第 3 个 slot（offset=2，因为前两个是 token #5,#6） → `slot = 7*4 + 2 = 30`
  - 一般化：`slot = block_table[-1] * block_size + last_block_num_tokens - 1`（这正是 prepare_decode 里的公式）
- prefill 跨块计算：对每个 i ∈ [start_block, end_block)，在该块内 slot 范围是
  - 起始：`block_table[i] * block_size + (i==start_block ? start%block_size : 0)`
  - 结束：`block_table[i] * block_size + (i==end_block-1 ? end - i*block_size : block_size)`
  - 对应 ModelRunner.prepare_prefill 的循环逻辑

### 解答 3.3（Triton kernel）
- 关键：把每个 token 的 (num_heads*head_dim) 一次性 load/store。Triton 编译器要求 `D` 是编译期常量，因此用 `tl.constexpr`。
- 必须断言 stride：
  - `key.stride(-1) == 1`：head_dim 维连续
  - `key.stride(1) == head_dim`：num_heads 维步长 = head_dim（普通行优先）
  - `k_cache.stride(1) == D`：blocks 维之后紧接着是 D 个连续元素 —— 这要求 KV Cache 是 `(2, L, B, S, H_kv, D_h)` 这种排布，最后两维拍平后是 D 个连续 float
- `slot == -1` 的 early return 是 CUDA Graph 的 padding 兼容（capture 时整个 batch 是 max_bs，多余位置 slot 填 -1）。

### 解答 3.4（Attention forward）
关键骨架：
```python
def forward(self, q, k, v):
    ctx = get_context()
    if self.k_cache.numel():
        store_kvcache(k, v, self.k_cache, self.v_cache, ctx.slot_mapping)
    if ctx.is_prefill:
        if ctx.block_tables is not None:    # 前缀命中：从 cache 读
            k, v = self.k_cache, self.v_cache
        return flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=ctx.cu_seqlens_q, max_seqlen_q=ctx.max_seqlen_q,
            cu_seqlens_k=ctx.cu_seqlens_k, max_seqlen_k=ctx.max_seqlen_k,
            softmax_scale=self.scale, causal=True, block_table=ctx.block_tables)
    # decode
    return flash_attn_with_kvcache(
        q.unsqueeze(1), self.k_cache, self.v_cache,
        cache_seqlens=ctx.context_lens, block_table=ctx.block_tables,
        softmax_scale=self.scale, causal=True)
```
- decode 时 q 只有 1 个 token，FlashAttention 要求 q 形状 `[B, 1, H, D]`，所以 `unsqueeze(1)`。

### 解答 3.5（KV Cache 布局）
- 维度 0：0=K, 1=V → 用同一张大张量节省 alloc / 让 Triton kernel 同时处理
- 维度 1：layer_id（每层独立 cache）
- 维度 2：block_id
- 维度 3：block 内 token 偏移（slot 内偏移）
- 维度 4：num_kv_heads（GQA 下 kv head 数 < q head 数）
- 维度 5：head_dim

slot 是「block_id × block_size + offset」一维编号。当 K/V 的最后两维 (H_kv, D_h) flat 成 `D` 之后，`slot * D + d` 就是它在「(num_blocks*block_size, D)」二维视角下的偏移 —— 这正是 Triton kernel 写入用的公式。

### 解答 3.6（端到端验证）
- 比对时注意：HuggingFace Qwen3Attention 没有 prefix cache 概念，所以 `block_tables` 应当传 None（走「k,v 用入参」分支）。
- 注意 RoPE 的 cos/sin 排布与 HF 不同时会导致单层 Attention 输出偏差，需要在阶段 2 已经对齐 RoPE。

### 常见坑
- flash-attn 版本不兼容：`flash_attn_varlen_func` 在不同版本里参数顺序或名字会变，对照 wheel 的 docstring 确认。
- KV Cache 不连续：如果你用切片得到的 view 而不是 contiguous，Triton kernel 的 stride 断言会失败 —— ModelRunner 里需要把 `module.k_cache = self.kv_cache[0, layer_id]` 这种切片视图保留下来，所以分配大张量时整体连续即可。

---

## 7. 自检提问

- [ ] 我能用一句话解释 `slot_mapping` 是什么
- [ ] 我能写出 prefill 跨块时 slot 范围的计算公式
- [ ] 我能解释为什么 prefill 路径要传 `block_tables`，而 decode 路径必传
- [ ] 我能说出 KV Cache 5D 张量为什么需要这个排布顺序
- [ ] 我能解释 `slot == -1` 跳过对 CUDA Graph 的意义