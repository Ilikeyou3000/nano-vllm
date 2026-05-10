# 阶段 2 · 基础算子层（单卡版本）

## 1. 结论

**实现 6 个原子算子（SiluAndMul / RMSNorm / RotaryEmbedding / Linear 簇 / VocabParallelEmbedding+ParallelLMHead / Sampler），每个都通过对照测试，先不引入 TP。**

## 2. 为什么这么做

- Qwen3 模型是这些算子的拼装结果，**算子错一个，模型一定错**。
- 单算子可以独立单测，定位 bug 比在大模型里 debug 容易 100 倍。
- 这一阶段主要训练你「PyTorch 里如何写一个高性能、可被 torch.compile 的算子」。
- 把 TP 留到阶段 9：先证明逻辑正确，再做并行化分布式。

## 3. 三大支柱任务

### 3.1 「计算型」算子（无可学权重）
SiluAndMul、RotaryEmbedding、Sampler —— 关注计算正确性与 `@torch.compile` 装饰。

### 3.2 「带权重」算子
RMSNorm、Linear 簇、VocabParallelEmbedding/ParallelLMHead —— 关注 Parameter 注册与 `weight_loader`。

### 3.3 单算子对照测试
每个算子写 5~20 行的 pytest 用例，与朴素 PyTorch 实现 / HuggingFace 官方实现对齐，误差 < 1e-3。

## 4. 验收标准

- [ ] 6 类算子全部通过对照测试
- [ ] 在 tp_size=1 的 ProcessGroup 下，所有 *Parallel* 类的输出 == 普通 Linear / Embedding 的输出
- [ ] 每个 Parameter 都挂上 `.weight_loader` 属性，便于阶段 4 的权重加载

---

## 5. 练习题

### 练习 2.1（SiluAndMul）
实现 [`layers/activation.py`](nanovllm/layers/activation.py)：
```
y = silu(x_left) * x_right    # 其中 x = concat([x_left, x_right], dim=-1)
```
要求：
- 用 `@torch.compile` 装饰 `forward`
- 输入 shape `[*, 2H]` → 输出 `[*, H]`
- 写测试：与 `F.silu(a) * b` 对齐

### 练习 2.2（RMSNorm）
实现 [`layers/layernorm.py`](nanovllm/layers/layernorm.py)：
- 只学一个 weight，shape `[hidden_size]`
- 提供两个 forward 入口：
  - `rms_forward(x)` —— 普通归一化
  - `add_rms_forward(x, residual)` —— **融合**：先 `x = x + residual`，再归一化，同时把 `residual` 更新为 `x` 的旧值（用于 Pre-LN 残差链）
- 用 `@torch.compile` 装饰
- 注意精度：`x = x.float()` 算方差，回到 `orig_dtype` 再乘 weight

### 练习 2.3（RotaryEmbedding）
实现 [`layers/rotary_embedding.py`](nanovllm/layers/rotary_embedding.py)：
- 在 `__init__` 中预计算 `cos_sin_cache`：shape `[max_pos, rotary_dim]`，`cat([cos, sin], dim=-1)` 后 `unsqueeze(1)`
- forward：
  ```
  cos_sin = self.cos_sin_cache[positions]
  cos, sin = chunk(cos_sin, 2, dim=-1)
  q = apply_rotary_emb(q, cos, sin)
  k = apply_rotary_emb(k, cos, sin)
  ```
- `apply_rotary_emb`：`x.chunk(2, -1)` → `(x1*cos - x2*sin, x2*cos + x1*sin)` → cat
- 用 `@lru_cache(1)` 包一个 `get_rope` 工厂函数，避免每层重新建 cache

### 练习 2.4（Linear 簇 · 单卡版）
实现 [`layers/linear.py`](nanovllm/layers/linear.py) 中 6 个类，**先全部按 tp_size=1 写**：
- `LinearBase`：基类，注册 weight 和 bias，给每个 Parameter 挂 `weight_loader`
- `ReplicatedLinear`：完整复制
- `ColumnParallelLinear`：tp_size=1 时等价 Replicated
- `MergedColumnParallelLinear`：output_size 是 list，按 list 偏移加载
- `QKVParallelLinear`：根据 `total_num_heads, total_num_kv_heads, head_size` 计算合并后的 output_size
- `RowParallelLinear`：tp_size=1 时等价 Replicated

每个 weight_loader 接收 `(param, loaded_weight)` 或 `(param, loaded_weight, shard_id)`。

### 练习 2.5（Embedding & LM Head）
实现 [`layers/embed_head.py`](nanovllm/layers/embed_head.py)：
- `VocabParallelEmbedding`：tp_size=1 时等价 `nn.Embedding`
- `ParallelLMHead`：继承 VocabParallelEmbedding，**注意 forward 里要从 hidden_states 中取每个序列的最后一个 token 的 hidden** 再做 lm_head 乘法
  ```
  if context.is_prefill:
      last_indices = context.cu_seqlens_q[1:] - 1
      x = x[last_indices]
  logits = F.linear(x, self.weight)
  ```
  这是 vLLM 风格：prefill 时整段 hidden_states 通过 transformer，但只对最后一个 token 算 logits（节省 vocab × seq_len 的乘法）。
- 由于这里依赖 `Context`，可以先写个 mock 的 context，等阶段 3 替换。

### 练习 2.6（Sampler）
实现 [`layers/sampler.py`](nanovllm/layers/sampler.py)：使用 Gumbel-Max 采样（**一行 argmax 完成**）。
```python
@torch.compile
def forward(self, logits, temperatures):
    logits = logits.float().div_(temperatures.unsqueeze(1))
    probs = torch.softmax(logits, dim=-1)
    sample = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
    return sample
```
解释：`probs / Exp(1)` 后 argmax 等价于按 probs 采样（Gumbel/Max 技巧），不需要 multinomial。

### 练习 2.7（对照测试）
写一个测试文件 `tests/test_layers.py`，每个算子至少一个用例：
- 输入随机 tensor，比较 `torch.testing.assert_close(my_out, ref_out, atol=1e-3, rtol=1e-3)`
- RMSNorm 与 `transformers.models.qwen3.modeling_qwen3.Qwen3RMSNorm` 对齐
- RotaryEmbedding 与 HuggingFace `apply_rotary_pos_emb` 对齐 —— 注意两者 cos/sin 排布约定可能不同

---

## 6. 解答

> 参考实现：[`layers/activation.py`](nanovllm/layers/activation.py)、[`layers/layernorm.py`](nanovllm/layers/layernorm.py)、[`layers/rotary_embedding.py`](nanovllm/layers/rotary_embedding.py)、[`layers/linear.py`](nanovllm/layers/linear.py)、[`layers/embed_head.py`](nanovllm/layers/embed_head.py)、[`layers/sampler.py`](nanovllm/layers/sampler.py)

### 解答 2.1（SiluAndMul）
- 用 `chunk(2, -1)` 切两半，分别作为 gate 和 value：`F.silu(gate) * value`。
- `@torch.compile` 后会自动融合成一个 kernel，比 eager 快 ~30%。

### 解答 2.2（RMSNorm）
- 易错点 1：`var = x.pow(2).mean(-1, keepdim=True)`，注意保留维度方便后续广播。
- 易错点 2：用 `mul_` 等 in-place 时要确保 x 已经是 float（避免 dtype 不一致）。
- `add_rms_forward` 的妙处：把残差加法融合进归一化的「读取阶段」，省一次显存往返 → 这就是为什么 Qwen3DecoderLayer 用它。

### 解答 2.3（RoPE）
- `cos_sin_cache` 在 init 时算好后注册成 buffer（`persistent=False` 不存 ckpt）。
- `apply_rotary_emb` 实现「半旋」式 RoPE，与 HuggingFace 的「交错」式不同；好在 Qwen3 已经用「半旋」格式存权重，因此与 HF 输出可以对齐。如果对不齐，先检查权重存储格式。
- `lru_cache(1)` 让所有 layer 共享同一个 RotaryEmbedding 实例（cache 只算一次）。

### 解答 2.4（Linear · 单卡）
关键骨架：
```python
class LinearBase(nn.Module):
    def __init__(self, in_size, out_size, bias=False, tp_dim=None):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()      # tp_size=1 时 rank=0
        self.tp_size = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(out_size, in_size))
        self.weight.weight_loader = self.weight_loader
        ...
```
- `weight_loader` 必须挂在 Parameter 上而非 Module，因为 loader 遍历的是 `state_dict` 名字 → 对应 Parameter。
- `QKVParallelLinear` 的 weight_loader 用 `loaded_shard_id ∈ {"q", "k", "v"}` 区分写入位置：
  - q：偏移 0，长度 `num_heads * head_size`
  - k：偏移 `num_heads * head_size`，长度 `num_kv_heads * head_size`
  - v：偏移 `(num_heads + num_kv_heads) * head_size`，长度 `num_kv_heads * head_size`
- `MergedColumnParallelLinear` 的 weight_loader 用 `loaded_shard_id ∈ {0, 1}` 区分 gate/up。

### 解答 2.5（Embedding & LM Head）
- `VocabParallelEmbedding` 在 tp_size=1 时只需 `F.embedding(x, self.weight)`，TP 部分留到阶段 9。
- `ParallelLMHead.forward` 的 last_indices 技巧：
  - 假设 batch 里有 3 个序列，长度分别是 [4, 6, 3]
  - cu_seqlens_q = [0, 4, 10, 13]
  - cu_seqlens_q[1:] - 1 = [3, 9, 12] —— 正是每个序列在 hidden_states 里最后一个 token 的位置
  - 这样 `x[last_indices]` 拿到的就是每个序列「最后一个 token」的 hidden，再做 lm_head 即可

### 解答 2.6（Sampler）
- 这套 Gumbel-Max trick 等价于 `torch.multinomial(probs)`，但在 GPU 上**比 multinomial 快很多**（避免了 CDF 排序）。
- `clamp_min_(1e-10)` 防止指数采样取到 0 后除零。

### 解答 2.7（对照测试）
- RMSNorm 测试要把 weight 设成 `ones` 然后输入 [1, 2, 3]，结果应该 ≈ 输入除以 RMS。
- RoPE 测试技巧：position=0 时 cos=1, sin=0，期望 RoPE(q) == q（恒等映射），可以用这个特例快速验证。

### 常见坑
- 忘记给 `dist.init_process_group("nccl", ..., world_size=1, rank=0)`：单卡测试时也要初始化 ProcessGroup，否则 `dist.get_rank()` / `get_world_size()` 报错。可以用 `dist.init_process_group("gloo", ...)` 在 CPU 调试时绕过 NCCL。
- `torch.compile` 在第一次调用会编译，单测要算第二次的耗时；对于纯小 tensor，`@torch.compile` 反而更慢，可以加 `mode="reduce-overhead"` 或测试时绕过。

---

## 7. 自检提问

- [ ] 我能讲清 RMSNorm 与 LayerNorm 的差异（前者不去均值、不加偏置）
- [ ] 我能解释为什么 RoPE 的 cos_sin 预计算在 `__init__` 而不是 forward
- [ ] 我能说出 QKV 合并投影相比拆开三个 Linear 的好处（一次 matmul，访存连续）
- [ ] 我能解释 ParallelLMHead 在 prefill 时只取最后一个 token 的原因
- [ ] 我能复述 Gumbel-Max 采样的一行公式