# 阶段 11 · 性能压测与调优

## 1. 结论

**用 `bench.py` 与 vLLM 官方做横向对比，用 `torch.profiler` / Nsight 找瓶颈，按「显存利用率 → 调度参数 → 算子优化」的次序系统调参，把吞吐推到硬件极限。**

## 2. 为什么这么做

- 实现完整不等于「能用」。一个项目要进入生产候选，必须给出可重复的性能数字与调优手册。
- 复现 README 中 ≈1400 tok/s（RTX 4070, Qwen3-0.6B）这条基线，是这套教程的最终交付物。
- 调优能教会你「哪些参数真正影响吞吐」—— 这是工程师的核心能力。

## 3. 三大支柱任务

### 3.1 基准测试脚本
完成 [`bench.py`](bench.py)：构造 256 条随机长度 prompt（输入 100~1024，输出 100~1024），分别测 nano-vllm 与 vllm，输出 tokens/s。

### 3.2 Profiling 找瓶颈
用 `torch.profiler.profile(...)` 抓 1 步 prefill + 10 步 decode，导出 chrome trace；用 Nsight Systems 抓更精细的 GPU timeline。

### 3.3 系统调参
按以下旋钮逐个验证：
- `gpu_memory_utilization` (0.85 → 0.95)
- `kvcache_block_size` (256 → 128 / 512)
- `max_num_batched_tokens` (16384 → 8192 / 32768)
- `max_num_seqs` (512 → 256 / 1024)
- `enforce_eager` (False) / `tensor_parallel_size` (1 / 2)

## 4. 验收标准

- [ ] 在 RTX 4070 / Qwen3-0.6B 复现 ≥ 1300 tok/s（README 基线 1434.13）
- [ ] 输出一份 markdown 报告，含 5 项参数对吞吐的影响曲线 / 表格
- [ ] 能用 chrome trace 指出至少 1 个明显的 GPU 空闲间隙并解释原因

---

## 5. 练习题

### 练习 11.1（构造测试集）
完成 `bench.py` 的 prompt 构造：
```python
import random
from transformers import AutoTokenizer
random.seed(0)
num_seqs = 256
max_input_len, max_output_len = 1024, 1024
prompt_token_ids = [
    [random.randint(0, 10000) for _ in range(random.randint(100, max_input_len))]
    for _ in range(num_seqs)
]
sampling_params = [
    SamplingParams(temperature=0.6, ignore_eos=True,
                   max_tokens=random.randint(100, max_output_len))
    for _ in range(num_seqs)
]
```
- 用 `ignore_eos=True` 强制每条 seq 跑满 max_tokens，方便对比。
- fix 随机种子保证两次运行可比。

### 练习 11.2（计时与吞吐）
```python
t = time.time()
llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
elapsed = time.time() - t
total_tokens = sum(sp.max_tokens for sp in sampling_params)
print(f"Throughput: {total_tokens/elapsed:.2f} tok/s")
```
- 注意：吞吐口径是「output token 总数 / 总耗时」，不计 input tokens。
- 第一次跑会有 capture 开销，建议先 warmup 一次再正式测。

### 练习 11.3（对比 vLLM）
安装 `pip install vllm` 后，把 `from nanovllm import LLM` 换成 `from vllm import LLM`，其它代码不变 → 跑相同测试集 → 对比吞吐。
- 期望 nano-vllm ≈ vllm 0.95~1.05 倍（README 显示 nano 1434 vs vllm 1361，nano 反超 5%，因为缺少 vllm 的 overhead）。

### 练习 11.4（chrome trace）
```python
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             schedule=torch.profiler.schedule(wait=1, warmup=2, active=3),
             on_trace_ready=torch.profiler.tensorboard_trace_handler("./logs")) as prof:
    for step in range(6):
        llm.step()
        prof.step()
```
然后 `chrome://tracing` 打开 logs 下的 json，观察：
- 每个 decode step 的 GPU busy 时间
- step 之间的 CPU 工作（schedule + prepare + sample），是否在 GPU 空闲时完成
- attention kernel 占比、linear kernel 占比

### 练习 11.5（gpu_memory_utilization 调参）
| util | KV blocks | 吞吐 (tok/s) | OOM? |
|------|-----------|-------------|------|
| 0.80 | ?         | ?           | No   |
| 0.85 | ?         | ?           | No   |
| 0.90 | ?         | ?           | No   |
| 0.95 | ?         | ?           | ?    |

- util 太高会 OOM；太低吞吐降低（KV 块少 → 同时只能装少 seq → 调度受限）。

### 练习 11.6（block_size 调参）
| block_size | KV blocks | 吞吐 |
|-----------|-----------|------|
| 64        | ?         | ?    |
| 128       | ?         | ?    |
| 256（默认） | ?         | ?    |
| 512       | ?         | ?    |

- 太小：分配/释放频繁，元数据开销大，prefix cache 更细粒度但管理慢。
- 太大：碎片严重（一个 short seq 也占整块），prefix cache 命中率低。
- 对当前模型/数据集找到最佳值。

### 练习 11.7（max_num_batched_tokens）
- 这是 prefill 的 token 上限：太大 → 单 step prefill 太久（latency 高）；太小 → throughput 低（GPU 没喂饱）。
- 对 Qwen3-0.6B 在 RTX 4070 上跑：8192 / 16384 / 32768 / 65536 各测一遍，画曲线。
- 找拐点：吞吐提升与 step 时间增长的 trade-off。

### 练习 11.8（CUDA Graph 增益）
- enforce_eager=True 跑一次 → 记 throughput_eager
- enforce_eager=False 跑一次 → 记 throughput_graph
- 提升 = (graph - eager) / eager，期望 ≥ 30%（小 bs decode-heavy 场景）。

### 练习 11.9（TP 增益）
- tp=1 跑 → 记 t1
- tp=2 跑 → 记 t2
- 期望 t2/t1 在 1.5~1.8 倍（理论 2 倍，扣除 all_reduce/gather 开销）。
- 注意：单卡能放下时，TP 反而可能更慢（通信成本 > 计算分摊收益）；只在大模型显存吃紧时 TP 才划算。

### 练习 11.10（写一份调优报告）
按以下结构产出 `doc/perf-report.md`：
1. 硬件环境（GPU 型号、CUDA 版本、PyTorch 版本）
2. 测试集（prompt 数、长度分布、sampling 参数）
3. 基线（默认参数下的 nano-vllm vs vllm）
4. 调参曲线（5 张表格）
5. 瓶颈分析（chrome trace 截图 + 文字描述）
6. 最佳配置 & 建议

---

## 6. 解答

> 参考实现：[`bench.py`](bench.py)；参考报告结构：vLLM 官方 perf 文档。

### 解答 11.1~11.3（基准）
- 关键是控制变量：随机种子、ignore_eos、相同 prompt 长度分布。
- 第一次跑「冷启动」包含模型加载、capture，不算入吞吐；从第二次开始测。

### 解答 11.4（profiler）
- chrome trace 看到的典型现象：每个 decode step ≈ 5~10ms，其中 attention 占 30%、QKV linear 30%、MLP 30%、其他 10%。
- step 间隔的 CPU 工作（scheduler.schedule + prepare_decode + sample）应远小于 GPU 工作 → 否则 CPU 是瓶颈。
- 如果发现 GPU idle 长，原因可能：
  - sample 用了 .item()（CPU sync）
  - block 分配在 hot path（应预分配）
  - block_tables 张量频繁创建（应复用）

### 解答 11.5（util）
- 默认 0.9 是经验值；0.95 在 4070 12GB 上可能 OOM（需要给中间张量留 buffer）。
- 提升 util 会增加 KV blocks → 让 max_num_seqs 更大 batch → 吞吐提升（边际递减）。

### 解答 11.6（block_size）
- 默认 256 经过 vLLM 多次 tuning，对大多数场景是好的。
- 短文本场景（<256）block_size=128 更好（碎片少）。
- prefix cache 重的场景（系统提示词长）更小的 block 命中率高。

### 解答 11.7（max_num_batched_tokens）
- prefill 是计算密集型，token 数线性增加但 launch 开销不变 → 提高 batched_tokens 能摊薄开销。
- 但 step 时间会变长 → 影响其他正在 decode 的 seq（被一次大 prefill 阻塞）。
- 实际选择：默认 16384 是平衡点；如果是离线批处理，可以 32768 进一步提速。

### 解答 11.8~11.9（graph & TP）
- graph 增益主要在 decode（小 kernel 多）。
- TP 增益主要在大模型大 bs（all_reduce 占比小）。
- 二者都是「条件性优化」，不能一概而论。

### 解答 11.10（报告）
- 表格 + 曲线（matplotlib）+ chrome trace 截图。
- 给出一句话「最佳配置」：例如「RTX 4070 / Qwen3-0.6B / 默认数据集，推荐 util=0.9, block_size=256, batched_tokens=16384, enforce_eager=False」。

### 常见坑
1. **测吞吐忘了 warmup** → 第一次结果偏低（含 capture）。
2. **同一进程多次跑 LLM** → SharedMemory name 冲突；每次跑用新进程。
3. **profiler schedule 配置错误** → 抓不到稳定状态；`wait=1, warmup=2, active=3` 是经典配方。
4. **vLLM 与 nano-vllm 对比时 sampling_params 不同** → 必须完全相同。
5. **改完代码忘了 pip install -e .** → 跑的还是旧版本。

---

## 7. 自检提问

- [ ] 我能复述出 5 个对吞吐影响最大的参数及其最佳值
- [ ] 我能用 chrome trace 找出 GPU 空闲段并定位原因
- [ ] 我能解释为什么 prefix cache 在系统提示词长场景下显著加速
- [ ] 我能在硬件给定的情况下，不靠经验直接估算理论上限（FLOPs / bandwidth）
- [ ] 我能写一份让别人能复现的性能报告

---

## 🎓 全 11 阶段完成寄语

到这里，你已经从零拼装出一个 ≈1.2k 行的高性能 LLM 推理引擎，并把它的吞吐压到了 vLLM 同等水平。**复刻不是终点 —— 真正的收获是你对「PagedAttention / Continuous Batching / CUDA Graph / TP」每个概念都能讲清「为什么」与「怎么调」**。

下一步可探索：
1. 实现 Speculative Decoding（小模型起草 + 大模型验证）
2. 加 Sliding Window Attention（支持更长上下文）
3. 接 OpenAI 兼容的 HTTP API（FastAPI 包一层）
4. 移植到其他模型架构（Llama-3, Mistral, DeepSeek-V2）

每一步都是新的金字塔顶端，等你去攀登。