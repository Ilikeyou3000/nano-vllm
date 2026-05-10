# 阶段 6 · Scheduler（持续批处理 + Chunked Prefill + 抢占）

## 1. 结论

**实现一个状态机：每一步把 waiting/running 队列转成「本轮可执行的 batch」，优先 prefill；空间不够时抢占；解码完后回收。**

## 2. 为什么这么做

- Scheduler 是引擎的「大脑」：决定**哪一步运行哪些序列**，直接决定吞吐与显存利用率。
- 它把「请求级 API」（add_request、is_finished）翻译成「张量级输入」给 ModelRunner，是抽象屏障所在。
- 它和 BlockManager 是好搭档：BlockManager 决定「能不能装」，Scheduler 决定「装谁、何时装」。

## 3. 三大支柱任务

### 3.1 队列与初始化
双队列 `waiting / running`、读取 Config 关键参数（`max_num_seqs / max_num_batched_tokens`）、持有 BlockManager。

### 3.2 schedule 主循环
两阶段优先级：先尝试调 prefill；没有 prefill 任务再做 decode；空间不够触发抢占。

### 3.3 postprocess
`hash_blocks` 建索引、`append_token` 写新 token、检查终止条件。

## 4. 验收标准

- [ ] 给定 100 条随机长度的 seq、有限 KV 块的 BlockManager，能 100 步内不死锁地推完
- [ ] 当显存不足时，**最新进入 running 的序列**优先被抢占
- [ ] EOS 或 max_tokens 触发后序列状态变为 FINISHED 且块被释放
- [ ] Chunked Prefill：第一个序列允许被切片，其他序列必须整段塞下

---

## 5. 练习题

### 练习 6.1（Scheduler 初始化）
实现 [`engine/scheduler.py`](nanovllm/engine/scheduler.py) 的 `__init__`：
- 读 Config：`max_num_seqs, max_num_batched_tokens, eos, kvcache_block_size`
- 初始化 `BlockManager(num_kvcache_blocks, block_size)`
- 创建两个 `deque`：`waiting`、`running`

### 练习 6.2（add / is_finished）
- `add(seq)`：append 到 waiting 队尾
- `is_finished()`：当且仅当 waiting 与 running 都空时返回 True

### 练习 6.3（Schedule 阶段一：Prefill）
实现 schedule() 的 prefill 部分：
- 循环：当 waiting 非空 **且** `len(scheduled) < max_num_seqs`：
  1. `seq = waiting[0]`（peek，不出队）
  2. `remaining = max_num_batched_tokens - num_batched_tokens`，若 remaining == 0 break
  3. 计算 num_tokens：
     - 若 seq.block_table 为空 → 首次调度，`num_cached_blocks = block_manager.can_allocate(seq)`，若返回 -1 表示 KV 不够，break
     - `num_tokens = seq.num_tokens - num_cached_blocks * block_size`
     - 否则（chunked prefill 续着上一次）：`num_tokens = seq.num_tokens - seq.num_cached_tokens`
  4. **chunked prefill 限制**：若 `remaining < num_tokens` **且** scheduled 非空 → break（只允许第一个 seq 被切片）
  5. 若是首次调度，调 `block_manager.allocate(seq, num_cached_blocks)`
  6. `seq.num_scheduled_tokens = min(num_tokens, remaining)`
  7. `num_batched_tokens += seq.num_scheduled_tokens`
  8. 若整段都装下了（`num_cached + num_scheduled == num_tokens`）：状态切 RUNNING、出队 → 入 running
  9. `scheduled.append(seq)`
- 若 scheduled 非空，返回 `(scheduled, True)` —— 这一步是 prefill batch

### 练习 6.4（Schedule 阶段二：Decode）
若没有可调 prefill：
- 循环：当 running 非空 **且** `len(scheduled) < max_num_seqs`：
  1. `seq = running.popleft()`
  2. while `not block_manager.can_append(seq)`：
     - 若 running 非空：抢占 `running.pop()`（最新加入的），调 `preempt`
     - 否则：抢占 seq 自己，break
  3. else（while 正常退出，说明 can_append 成立）：
     - `seq.num_scheduled_tokens = 1`
     - `seq.is_prefill = False`
     - `block_manager.may_append(seq)`
     - `scheduled.append(seq)`
- 复原顺序：`running.extendleft(reversed(scheduled))`（因为我们 popleft 之后又把它们 append 到 scheduled，要按原顺序回写）
- 返回 `(scheduled, False)`

### 练习 6.5（preempt）
```python
def preempt(self, seq):
    seq.status = SequenceStatus.WAITING
    seq.is_prefill = True
    self.block_manager.deallocate(seq)
    self.waiting.appendleft(seq)
```
- 释放块、状态回到 WAITING、放回 waiting **队首**（下一轮优先重启）
- 注意 is_prefill 设回 True：意味着下一轮会重新做 prefill（已生成的 token 还在 token_ids 里，只是 KV 丢了，需要重算）

### 练习 6.6（postprocess）
```python
def postprocess(self, seqs, token_ids, is_prefill):
    for seq, token_id in zip(seqs, token_ids):
        self.block_manager.hash_blocks(seq)
        seq.num_cached_tokens += seq.num_scheduled_tokens
        seq.num_scheduled_tokens = 0
        if is_prefill and seq.num_cached_tokens < seq.num_tokens:
            continue                                    # chunked prefill 还没结束
        seq.append_token(token_id)
        if (not seq.ignore_eos and token_id == self.eos) or \
           seq.num_completion_tokens == seq.max_tokens:
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)
            self.running.remove(seq)
```
- 关键：chunked prefill 在 prefill 中间步骤**不要 append_token**（采样出来的 token 在 chunk 没跑完时是无意义的）。

### 练习 6.7（端到端流程模拟）
不接 GPU，写一个 mock test：
- 假设 `num_kvcache_blocks=4, block_size=4, max_num_seqs=8, max_num_batched_tokens=32`
- 提交 3 条序列，长度分别 3 / 6 / 5
- 手动跑：
  - step 1：应该是 prefill，scheduled = [seq1, seq2, seq3]（如果都装得下）
  - step 2~：decode，每步生成 1 token
- 用 mock token_ids（一直返回非 eos）跑到 max_tokens 终止
- 验证：所有序列最终都 FINISHED，free_block_ids 数量 == 初始 num_blocks

### 练习 6.8（抢占场景验证 · 进阶）
缩小 num_kvcache_blocks（如 3），让 decode 阶段必然触发抢占：
- 提交 4 条长 prompt
- 跑若干步，观察日志：哪条序列被抢占？被抢占后下一轮如何重新 prefill？
- 验证最终结果：所有序列依然能完成，没有死锁

---

## 6. 解答

> 参考实现：[`engine/scheduler.py`](nanovllm/engine/scheduler.py)

### 解答 6.3 关键点（Prefill 阶段）
- **「peek 而非 popleft」**：因为可能 KV 不够（can_allocate 返回 -1），这条序列要保留在 waiting 头部等下一轮，不能误出队。
- **chunked prefill**：核心是「只允许第一个 seq 被切片，否则破坏 batch 紧凑度」。判断方式：当 `scheduled` 已经有别人时，剩余 budget 不够本 seq 整段就 break。
- **首次 vs 续上**：首次调度才需要 BlockManager 分配；续上的 chunked prefill 用上次留下的 block_table。

### 解答 6.4 关键点（Decode 阶段）
- 抢占顺序：`running.pop()` 取最新进入的（队尾），保留早期到达的请求继续推进 —— FIFO 公平性。
- 注意 while-else 的语义：while 条件**不再成立**（即 can_append 为真）才进入 else 块；如果是 break 出来则跳过 else（说明这个 seq 自己被抢占了）。
- 「复原顺序」：popleft + scheduled.append 已经把它们从队首取出，最后用 `extendleft(reversed(...))` 一次性塞回，保持原顺序。

### 解答 6.5 关键点（preempt）
- preempt 后块全部释放，但 **token_ids 保留** —— 下一次 prefill 时会重算 KV。这就是「重计算式抢占」（vLLM 的经典策略，比把 KV 写硬盘的 swap 简单且对短上下文够用）。
- 放回 waiting 队首：保证抢占的序列下一轮能立刻被尝试重启。

### 解答 6.6 关键点（postprocess）
- `hash_blocks` 必须在 `num_cached_tokens` **更新之前**调，因为它依赖当前 num_cached_tokens 来定位 start。
- chunked prefill 中间步骤跳过 append_token：因为模型这一步的 logits 是「还没结束的 prefill 中间状态」，对应的最后一个 token 是 prompt 自带的，不是要采样的输出。
- finished 时从 running 移除：注意只在 decode 路径（is_prefill=False）才会发生 finish，因为 EOS 是 decode 出来的；prefill 中虽然完整跑完，但还没生成新 token。

### 解答 6.7 模拟过程
设 block_size=4：
- seq1 (3 tokens) 需 1 块；seq2 (6 tokens) 需 2 块；seq3 (5 tokens) 需 2 块 → 总 5 块，但只有 4 块 → seq3 等待
- step1：scheduled=[seq1, seq2]，prefill 分别 3+6=9 tokens
- step2..：decode，3 个块（seq1 用 1，seq2 用 2）；当 seq2 长度达 9 时跨第 3 块，需要再分一块 → 此时 free=1，没问题
- 一旦 seq1 完成释放，seq3 才会被允许 prefill

### 解答 6.8 抢占验证
- num_blocks=3 时若同时跑 4 条长 prompt，必然在 decode 跨块时触发抢占。
- 观察现象：最早完成 prefill 的序列最先 decode，进度领先；空间不足时**最新加入的**被踢回 waiting；下一轮它重新 prefill 重算。
- 死锁防护：`while not can_append: ... else: ...` 的 break 分支会**抢占自己**，确保即使 running 只有它一个、KV 也不够，最坏情况下也能让出空间，不会卡住。

### 常见坑
1. **prefill 里把 popleft 写在判断之前** → 当 KV 不够时 seq 被错误移除。一定先 peek，能装下再 popleft。
2. **`max_num_batched_tokens` 边界没处理 0** → 当上一步刚把额度用完，本步进入 schedule 时 remaining=0 要立刻 break。
3. **抢占自身后 break 出 while** → 别忘了在 break 后**不要走 else 分支**（这条序列已经回到 waiting 不能再加入 scheduled）。
4. **postprocess 里 chunked prefill 误 append_token** → 模型在 prefill 中间步骤 logits 是噪音，append 会污染 token_ids。

---

## 7. 自检提问

- [ ] 我能讲清 Prefill 优先于 Decode 的原因（吞吐导向）
- [ ] 我能解释为什么 chunked prefill 只对第一个序列开放
- [ ] 我能用一句话说出 `running.pop()` 而非 `popleft()` 的目的（保留先到达请求）
- [ ] 我能描述抢占的完整后果（块释放 + 状态回退 + token_ids 保留 + is_prefill 重置）
- [ ] 我能解释 postprocess 里的 hash_blocks 与 num_cached_tokens 更新顺序为什么不能颠倒