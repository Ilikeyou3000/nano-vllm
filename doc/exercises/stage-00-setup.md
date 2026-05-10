# 阶段 0 · 环境与前置知识

## 1. 结论

**先把官方 Nano-vLLM 跑起来，并用自己的话能讲清 5 个核心概念，才能进入阶段 1。**

## 2. 为什么这么做

复现一个推理引擎，**最大的风险不是写代码，而是方向走错**：
- 没有跑通 baseline，后续任何「我实现错了 vs 我实现慢了」都无法判断；
- 没有理解 PagedAttention / Continuous Batching 的本质，后面写 BlockManager 与 Scheduler 时会反复推倒重来。

所以本阶段 = **建立基线 + 建立心智模型**。

## 3. 三大支柱任务

### 3.1 装齐依赖并跑通官方示例
让 [`example.py`](example.py)、[`bench.py`](bench.py) 都能正常出结果。

### 3.2 用自己的话写下 5 个核心概念
PagedAttention、Prefix Cache、Continuous Batching、Tensor Parallelism、CUDA Graph。

### 3.3 整理 Qwen3-0.6B 的关键超参表
hidden_size、num_layers、num_heads、num_kv_heads、head_dim、vocab_size、max_position 等。

## 4. 验收标准

- [ ] `python example.py` 能输出两个 prompt 的完成文本
- [ ] `python bench.py` 能打印 throughput
- [ ] 在自己的笔记里能 ≤ 200 字解释 PagedAttention
- [ ] 能画出 Qwen3 一个 DecoderLayer 的数据流（embed→attn→mlp→norm）

---

## 5. 练习题

### 练习 0.1（环境）
在干净 conda 环境里安装依赖并跑通官方 example。请记录：你的 GPU 型号、显存、`nvidia-smi` 中 CUDA 版本、`pip list | grep -E "torch|triton|flash"` 的输出。

### 练习 0.2（baseline 数据）
跑 [`bench.py`](bench.py) 至少 3 次，记录：
- 每次输出的 `Total tokens / Time / Throughput`
- 三次的方差（吞吐稳定性如何？）

### 练习 0.3（PagedAttention 概念图）
画一张图（手绘也行），展示：
- 一个长度为 600 的序列，block_size=256，被切成几个块？
- 块表（block_table）长什么样？
- 当两个序列共享前 512 个 token（前缀复用）时，它们的 block_table 在内存里是「指向同一物理块」还是「各自复制一份」？

### 练习 0.4（Continuous Batching 概念）
回答：
1. 静态 batch 与 continuous batching 的本质差别是什么？
2. Chunked Prefill 解决了什么问题？什么场景才会触发？
3. 当显存不够时，vLLM 用什么策略？

### 练习 0.5（Qwen3 配置摸底）
用 `transformers.AutoConfig.from_pretrained("~/huggingface/Qwen3-0.6B")` 加载并打印：
`hidden_size, num_hidden_layers, num_attention_heads, num_key_value_heads, head_dim, vocab_size, max_position_embeddings, rope_theta, tie_word_embeddings`。

把这些数字写到自己的速查卡上 —— 阶段 4 验证模型时会反复用到。

---

## 6. 解答

### 解答 0.1
- 推荐 `python>=3.10,<3.13`，`torch>=2.4`，CUDA ≥ 12.1。
- `flash-attn` 可能编译慢，使用预编译 wheel：`pip install flash-attn --no-build-isolation`。
- 跑不动时常见原因：CUDA / torch / flash-attn 版本不匹配，先固定一个 torch 版本再装 flash-attn。

### 解答 0.2
- 第一次启动会包含 warmup + torch.compile，吞吐偏低；后续两次会接近稳态。
- 三次取最大值作为「峰值吞吐」、中位数作为「日常吞吐」即可。

### 解答 0.3 关键要点
- 长度 600，block_size=256 → ⌈600/256⌉ = **3** 个块（最后一块只用了 88 个 slot）。
- 块表是 `list[int]`，存的是物理块 id，例如 `[12, 47, 91]`。
- **同一物理块**：前缀复用通过哈希命中，把同一个 `block_id` 写进两个序列的 block_table，物理上不复制（这是 PagedAttention 节省显存的关键）。

### 解答 0.4 关键要点
1. 静态 batch：所有请求等齐后一起 prefill+decode 到结束；continuous batching：**每一步**都重新组 batch，已完成的退出，新的随时加入 → GPU 利用率显著提升。
2. Chunked Prefill：当 prefill token 总数超过 `max_num_batched_tokens` 时，把第一个序列的 prefill 切片，避免单步 OOM。只在「还没填满 batch」的时候触发。
3. 抢占（preemption）：选最新进入 running 的序列，回退到 waiting，把它已分配的 KV 块释放给当前需要扩展的序列。

### 解答 0.5 参考
Qwen3-0.6B 的典型值（不同版本可能微调）：
| 字段 | 值 |
|---|---|
| hidden_size | 1024 |
| num_hidden_layers | 28 |
| num_attention_heads | 16 |
| num_key_value_heads | 8 (GQA) |
| head_dim | 128 |
| vocab_size | 151936 |
| max_position_embeddings | 32768 |
| tie_word_embeddings | True |

> 阶段 4 写权重加载时，能立刻看出 lm_head 与 embed_tokens 共享 weight。

---

## 7. 自检提问（全部 ✅ 再进下一阶段）

- [ ] 我能在 5 分钟内重新跑通 example.py
- [ ] 我能解释「block_size=256」的含义和它影响什么
- [ ] 我能解释 prefill 与 decode 阶段计算特性的差异（计算密集 vs 访存密集）
- [ ] 我能说出 vLLM 抢占的触发条件
- [ ] 我能背出 Qwen3-0.6B 的 num_layers / num_heads / num_kv_heads / head_dim