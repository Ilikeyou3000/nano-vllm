# 阶段 8 · LLMEngine 与对外 API

## 1. 结论

**把 Scheduler + ModelRunner + Tokenizer 拼起来，对外暴露 vLLM 风格的 `LLM.generate(...)`，让 [`example.py`](example.py) 能跑。**

## 2. 为什么这么做

- 前 7 个阶段的零件都已经具备，但用户还**无法用一行代码生成文本**。本阶段把它们粘合，建立用户视角的 API。
- 这一步会暴露之前所有遗漏的小 bug（顺序、生命周期、tqdm 进度），是上线前的最后一关。
- 拼装比想象简单 —— 但**关键决策点**（如何收集结果、如何展示吞吐、如何处理 EOS）值得仔细推敲。

## 3. 三大支柱任务

### 3.1 LLMEngine.__init__ 整合
解析 kwargs → Config → 启动 ModelRunner（含 worker 进程，留 TP 占位）→ Tokenizer → Scheduler。

### 3.2 add_request / step 单步循环
- `add_request(prompt, sp)` → encode → 创建 Sequence → `scheduler.add(seq)`
- `step()` → `scheduler.schedule()` → `model_runner.run(...)` → `scheduler.postprocess(...)` → 收集已完成

### 3.3 generate 批量生成入口
塞入所有 prompt → 循环 step 直到 is_finished → 用 tokenizer 反解码 → 返回 list[dict]。

## 4. 验收标准

- [ ] 你版本的 `example.py` 能输出与官方版本相同结构的结果
- [ ] tqdm 显示 prefill/decode 实时吞吐
- [ ] 多 prompt 并发场景下，生成顺序与输入顺序一致（按 seq_id 排序输出）

---

## 5. 练习题

### 练习 8.1（LLMEngine.__init__）
实现 [`engine/llm_engine.py`](nanovllm/engine/llm_engine.py) 的 `__init__(model, **kwargs)`：
1. 用 `dataclass.fields(Config)` 过滤 kwargs，只保留 Config 认识的参数
2. 创建 `Config(model, **filtered)`
3. **同步类变量**：`Sequence.block_size = config.kvcache_block_size`（让 Sequence 类的 `block(i)` 等方法用正确块大小）
4. （TP 占位）：当 `tensor_parallel_size > 1` 时用 mp.spawn 启 worker 进程；阶段 9 再做，这里先写 stub
5. 创建 `self.model_runner = ModelRunner(config, 0, [])`
6. `self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)`
7. `config.eos = tokenizer.eos_token_id`
8. `self.scheduler = Scheduler(config)`
9. `atexit.register(self.exit)` —— 进程退出时自动清理 worker

### 练习 8.2（add_request）
```python
def add_request(self, prompt, sampling_params):
    if isinstance(prompt, str):
        prompt = self.tokenizer.encode(prompt)
    seq = Sequence(prompt, sampling_params)
    self.scheduler.add(seq)
```
- 同时支持 str 和 list[int]（vLLM 兼容）

### 练习 8.3（step）
```python
def step(self):
    seqs, is_prefill = self.scheduler.schedule()
    num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
    token_ids = self.model_runner.call("run", seqs, is_prefill)
    self.scheduler.postprocess(seqs, token_ids, is_prefill)
    outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
    return outputs, num_tokens
```
- 用 `model_runner.call("run", ...)` 而非直接 `model_runner.run(...)`：是为了 TP 模式下走共享内存广播路径（阶段 9）。单卡 call 等价于直接调用。
- `num_tokens` 用正负号区分 prefill / decode：正数 = prefill 总 token，负数（其绝对值） = decode 序列数

### 练习 8.4（generate）
```python
def generate(self, prompts, sampling_params, use_tqdm=True):
    pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
    if not isinstance(sampling_params, list):
        sampling_params = [sampling_params] * len(prompts)
    for prompt, sp in zip(prompts, sampling_params):
        self.add_request(prompt, sp)
    
    outputs = {}
    prefill_throughput = decode_throughput = 0.
    while not self.is_finished():
        t = perf_counter()
        output, num_tokens = self.step()
        dt = perf_counter() - t
        if num_tokens > 0:
            prefill_throughput = num_tokens / dt
        else:
            decode_throughput = -num_tokens / dt
        pbar.set_postfix({
            "Prefill": f"{int(prefill_throughput)}tok/s",
            "Decode":  f"{int(decode_throughput)}tok/s",
        })
        for seq_id, token_ids in output:
            outputs[seq_id] = token_ids
            pbar.update(1)
    pbar.close()
    
    # 按输入顺序还原
    outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
    return [{"text": self.tokenizer.decode(t), "token_ids": t} for t in outputs]
```
- `seq_id` 全局自增，所以按 `sorted(outputs.keys())` 就是输入顺序。
- 返回 dict 而非 vLLM 的 RequestOutput 对象，让 API 更轻。

### 练习 8.5（LLM 子类与导出）
- [`llm.py`](nanovllm/llm.py)：`class LLM(LLMEngine): pass` —— 用户语义命名
- [`__init__.py`](nanovllm/__init__.py)：`from .llm import LLM`、`from .sampling_params import SamplingParams`

### 练习 8.6（端到端跑通）
复刻 [`example.py`](example.py)：
- 用 tokenizer.apply_chat_template 给 prompt 加上 chat 模板
- llm.generate 出结果
- 打印 prompt / completion

如果 OOM：减小 `gpu_memory_utilization`、`max_num_batched_tokens`、`max_num_seqs`。

---

## 6. 解答

> 参考实现：[`engine/llm_engine.py`](nanovllm/engine/llm_engine.py)、[`llm.py`](nanovllm/llm.py)、[`__init__.py`](nanovllm/__init__.py)

### 解答 8.1 关键点
- 用 `dataclass.fields(Config)` 拿出所有合法字段名，用 set 过滤 kwargs：避免用户传错参数被默默忽略。
- `Sequence.block_size = config.kvcache_block_size` 必须在 Scheduler / ModelRunner 之前调（创建 Sequence 时已经依赖正确的类变量）。
- atexit 注册：用户脚本退出时自动 join worker 进程，避免僵尸进程。

### 解答 8.3 关键点
- `model_runner.call("run", ...)` 在 TP 模式下：rank0 通过共享内存把 ("run", seqs, is_prefill) 广播给其他 worker，自己也调 self.run。这种「主从对称」让单进程版与多进程版用同一接口。
- `output` 用 `seq_id` 标识：因为 schedule 出来的 batch 顺序与添加顺序无关，必须用 id 关联。

### 解答 8.4 关键点
- prefill 吞吐用「这一步的总 token 数 / 耗时」；decode 吞吐用「这一步的序列数（每序列 1 token）/ 耗时」。
- tqdm.update 在序列结束时调一次，进度条表示「已完成请求数」。
- 排序：seq_id 全局自增，且 add_request 是顺序调的，所以 sorted 等价于按 prompt 输入顺序。

### 解答 8.5 关键点
- 用 `LLM(LLMEngine): pass` 是为了语义清晰且方便未来差异化（例如 LLM 加 chat 接口而 LLMEngine 保持纯 API）。

### 解答 8.6 端到端
- 期望：两个 prompt（"introduce yourself", "list all prime numbers within 100"）都生成连贯的回答。
- 检查点：
  - prefill 吞吐显示为正常值（千级 tok/s）
  - decode 吞吐稳定（百级 tok/s 到千级，看 GPU）
  - 两个 prompt 的输出顺序与输入一致

### 常见坑
1. **忘记 `Sequence.block_size = ...` 同步** → block(i) 切片错位 → KV Cache 与实际 token 不对齐 → 输出乱码。
2. **`model_runner.call("exit")` 未注册** → 用户 Ctrl-C 后 worker 进程不退出。
3. **decode 吞吐显示为 0**：通常是因为单步耗时被 prepare_prefill 的 first-time CPU→GPU 同步拖累；warmup 后会恢复正常。
4. **TP 模式下 add_request 在 worker 也跑** → 必须只在 rank0 添加，否则 seq_id 重复。LLMEngine 只在 rank0 实例化即可避免。

---

## 7. 自检提问

- [ ] 我能解释 `model_runner.call("run", ...)` 与直接 `model_runner.run(...)` 的差异
- [ ] 我能讲出 `num_tokens` 用正负号区分 prefill/decode 的妙处
- [ ] 我能说明 sorted(outputs.keys()) 为什么等价于按输入顺序
- [ ] 我能描述 atexit.register 的作用
- [ ] 我能在 5 分钟内复刻自己的 example.py 跑出输出