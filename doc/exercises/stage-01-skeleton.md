# 阶段 1 · 骨架与数据结构

## 1. 结论

**搭好你自己的 `mynanovllm/` 包目录，并实现 3 个不依赖 GPU 的纯 Python 类：`Config`、`SamplingParams`、`Sequence`。**

## 2. 为什么这么做

- 这些类是后续所有阶段的**通用语言**：BlockManager 操作 `Sequence`，Scheduler 决策依赖 `SamplingParams`，ModelRunner 读取 `Config`。
- 它们 100% 可以脱离 GPU 写完，**纯单测覆盖**，是建立信心的最佳起点。
- 写完之后，你会对「一个 LLM 推理请求在引擎眼里长什么样」有具体认知。

## 3. 三大支柱任务

### 3.1 项目骨架
建立 `mynanovllm/{engine,layers,models,utils}/__init__.py`，与原项目保持目录同构。

### 3.2 配置类
`Config`：从 HuggingFace 模型路径加载 hf_config，校验参数合法性。

### 3.3 序列对象
`Sequence`：维护 token 列表、block_table、缓存进度，提供按块切分的接口。

## 4. 验收标准

- [ ] 能够 `from mynanovllm import LLM, SamplingParams`（暂时让 LLM 是占位类）
- [ ] `Config(model="/path/to/Qwen3")` 能正确加载并填充 hf_config
- [ ] `Sequence([0..600], SamplingParams())` 创建后：
  - `len(seq) == 601`（含一个 last_token）—— 实际看你的 token 数
  - `seq.num_blocks` 与手算一致
  - `seq.block(i)` 返回正确切片
- [ ] 用 `pickle.dumps(seq)` 不抛异常

---

## 5. 练习题

### 练习 1.1（项目骨架）
建立目录树：
```
mynanovllm/
├── __init__.py        # 暴露 LLM, SamplingParams
├── config.py
├── sampling_params.py
├── llm.py             # 占位：class LLM: pass
├── engine/__init__.py
│   └── sequence.py
├── layers/__init__.py
├── models/__init__.py
└── utils/__init__.py
```

### 练习 1.2（SamplingParams）
实现一个 `@dataclass(slots=True)`：
- 字段：`temperature: float = 1.0`、`max_tokens: int = 64`、`ignore_eos: bool = False`
- 在 `__post_init__` 里断言 `temperature > 1e-10`（不允许贪心采样，避免精度爆炸）

### 练习 1.3（Config）
实现 `Config`：
- 字段：`model, max_num_batched_tokens=16384, max_num_seqs=512, max_model_len=4096, gpu_memory_utilization=0.9, tensor_parallel_size=1, enforce_eager=False, hf_config=None, eos=-1, kvcache_block_size=256, num_kvcache_blocks=-1`
- 在 `__post_init__` 里：
  1. 断言 `model` 是已存在的目录
  2. 断言 `kvcache_block_size % 256 == 0`
  3. 断言 `1 <= tensor_parallel_size <= 8`
  4. 用 `transformers.AutoConfig.from_pretrained` 填充 `hf_config`
  5. 用 `min(self.max_model_len, hf_config.max_position_embeddings)` 裁剪 max_model_len

### 练习 1.4（Sequence 基础）
实现 `SequenceStatus(Enum) = {WAITING, RUNNING, FINISHED}` 与 `Sequence`：
- 类变量 `block_size = 256`、`counter = itertools.count()` —— 用来给每条序列发唯一 id
- `__init__(token_ids, sampling_params)`：复制 token_ids、记录 `last_token`、`num_tokens`、`num_prompt_tokens`、`num_cached_tokens=0`、`num_scheduled_tokens=0`、`is_prefill=True`、`block_table=[]`
- `__len__`、`__getitem__`、`is_finished` property

### 练习 1.5（Sequence 块切分）
实现关键属性：
- `num_blocks`：⌈num_tokens / block_size⌉
- `last_block_num_tokens`：最后一块实际占用的 slot 数
- `block(i)`：返回 `token_ids[i*block_size : (i+1)*block_size]`
- `prompt_token_ids` / `completion_token_ids` / `num_completion_tokens`
- `append_token(token_id)`：把新 token 追加并维护 `last_token`、`num_tokens`

### 练习 1.6（pickle 优化 · 进阶）
TP 模式下 `Sequence` 需要在主进程→worker 之间通过共享内存 pickle 传递，**完整 token_ids 太占带宽**。请实现：
- `__getstate__`：当 `is_prefill=True` 时传完整 token_ids；否则只传 `last_token`
- `__setstate__`：根据收到的是 list 还是 int 还原对象

写一个验证脚本：构造一个 prefill 中的 seq 与 decode 中的 seq，分别 pickle.dumps 比较 size，证明 decode 状态的传输量很小。

---

## 6. 解答

> 参考实现：[`config.py`](nanovllm/config.py)、[`sampling_params.py`](nanovllm/sampling_params.py)、[`engine/sequence.py`](nanovllm/engine/sequence.py)

### 解答 1.1
注意 `__init__.py` 内容：
```python
# mynanovllm/__init__.py
from .llm import LLM
from .sampling_params import SamplingParams
```
后续阶段才把 LLM 改成真正继承 LLMEngine。

### 解答 1.2 关键点
- `slots=True` 节省每个对象的 `__dict__`（数百万 sequence 时省内存）。
- 不允许贪心是为了：本项目用 Gumbel-Max trick 采样，T=0 会让 `logits/T` 溢出。

### 解答 1.3 关键点
- 检查 `os.path.isdir(model)` —— 不允许 HuggingFace Hub 自动下载，保证可复现。
- `hf_config` 用 dataclass 默认 `None` + `__post_init__` 填充的做法，比写 `field(default_factory=...)` 更清楚。
- `num_kvcache_blocks=-1` 是占位，真实值在 ModelRunner.allocate_kv_cache 时算。

### 解答 1.4 + 1.5 关键点
- `counter = itertools.count()` 是类变量 → 全局 seq_id 自增；多进程下注意 worker 不应该再产生新 id（只主进程产生）。
- `num_blocks = (num_tokens + block_size - 1) // block_size`
- `last_block_num_tokens = num_tokens - (num_blocks - 1) * block_size`，例如 num_tokens=600, block_size=256 → num_blocks=3, last_block=600-512=88
- `block(i)`：`token_ids[i*block_size : (i+1)*block_size]`，对最后一块自动短切

### 解答 1.6 关键点
对照原实现：
```python
def __getstate__(self):
    last_state = self.last_token if not self.is_prefill else self.token_ids
    return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
            self.num_scheduled_tokens, self.block_table, last_state)

def __setstate__(self, state):
    (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
     self.num_scheduled_tokens, self.block_table, last_state) = state
    if isinstance(last_state, list):
        self.token_ids = last_state
        self.last_token = self.token_ids[-1]
    else:
        self.token_ids = []
        self.last_token = last_state
```

- decode 阶段 worker 只关心 last_token、block_table、cached/scheduled 计数，不需要完整 token 历史。
- 注意 status / temperature / max_tokens / ignore_eos / is_prefill / seq_id 这些 worker 用不到的字段没有进入 pickle —— 这正是优化点。

### 常见坑
- 把 `block_size=256` 写到 `Sequence` 类内做硬编码，但 LLMEngine 又会用 `Config.kvcache_block_size` 覆盖：原项目通过 `Sequence.block_size = config.kvcache_block_size` 修改类变量来同步，记得在 LLMEngine.__init__ 里这样做。
- `counter = count()` 在多进程 spawn 时会被复制，可能产生重复 id；在 TP 中我们只在 rank0 创建 Sequence，所以问题不大，但要记住这个限制。

---

## 7. 自检提问

- [ ] 我能解释为什么 `block_size=256` 是类变量而不是实例变量
- [ ] 我能写出 `num_blocks` 的公式
- [ ] 我能说明 prefill 阶段和 decode 阶段 pickle Sequence 的差别
- [ ] `Sequence` 中除了 token_ids，还有哪些字段会被 Scheduler 修改？（答：is_prefill / status / num_cached_tokens / num_scheduled_tokens / block_table）
- [ ] 我能在不查代码的情况下，说出 `last_block_num_tokens` 与 `block(i)` 的实现思路