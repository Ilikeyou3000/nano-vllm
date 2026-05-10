# 阶段 5 · BlockManager（分页 + 前缀缓存）

## 1. 结论

**实现一个能「按块分配 KV、按 hash 复用前缀、按引用计数释放」的 BlockManager —— 这是 vLLM 节省显存的核心。**

## 2. 为什么这么做

- 朴素 KV Cache 给每个序列预分配 max_len 大小，**显存浪费率 60% 以上**；分页方案只在需要时分配。
- Prefix Cache 让相同的 system prompt / few-shot 模板**只存一份**，在多请求场景下命中率经常 >50%。
- 这一阶段是整个 nano-vllm 中**最纯算法**的部分，没有 GPU、没有模型 —— 但坑也最深，需要极仔细地维护引用计数和哈希表的一致性。

## 3. 三大支柱任务

### 3.1 块对象与块池
`Block` 类（id/ref_count/hash/token_ids）+ 块池（free_block_ids、used_block_ids、hash_to_block_id）。

### 3.2 前缀复用机制
`compute_hash`（链式哈希）+ `can_allocate`（扫描可命中的前缀块数）+ `allocate`（命中复用、未命中分配新块）。

### 3.3 增量与回收
- `can_append / may_append`：decode 时按需追加新块
- `deallocate`：序列结束时归还块
- `hash_blocks`：每次写入完整块时立即建立 hash 索引

## 4. 验收标准

- [ ] 同一 prompt 提交两次：第二次 `can_allocate` 返回 `num_blocks - 1`（最后一个不完整块不哈希）
- [ ] 一个序列结束后再来一个相同前缀的，前面的物理块仍可命中（被某序列释放但 hash 仍在）
- [ ] 引用计数：当一个块被 N 个序列共享，前 N-1 次 deallocate 不会真正释放
- [ ] free + used 的总数始终等于初始化的 num_blocks（不会漏块）

---

## 5. 练习题

### 练习 5.1（Block 类）
实现 `Block`：
- 字段：`block_id, ref_count=0, hash=-1, token_ids=[]`
- 方法：
  - `update(hash, token_ids)`：写入 hash 与该块的 tokens
  - `reset()`：把 `ref_count=1, hash=-1, token_ids=[]`（_allocate 时调用）

### 练习 5.2（compute_hash）
实现 [`engine/block_manager.py`](nanovllm/engine/block_manager.py) 中的 `BlockManager.compute_hash`：
- 用 `xxhash.xxh64`（比 Python 内置 hash 稳定，且支持增量）
- 接受 `token_ids: list[int]` 与 `prefix: int = -1`（前一块的 hash）
- 当 `prefix != -1` 时，先 update 8 字节 little-endian 的 prefix
- 然后 update `np.array(token_ids).tobytes()`
- 返回 `h.intdigest()`

为什么链式：让「相同 token_ids 但前文不同」的两个块产生不同 hash（避免错误命中）。

### 练习 5.3（块池初始化）
实现 `BlockManager.__init__(num_blocks, block_size)`：
- `self.blocks = [Block(i) for i in range(num_blocks)]`
- `self.free_block_ids = deque(range(num_blocks))`
- `self.used_block_ids = set()`
- `self.hash_to_block_id = dict()`

### 练习 5.4（_allocate / _deallocate）
实现内部辅助：
```python
def _allocate_block(self):
    block_id = self.free_block_ids.popleft()
    block = self.blocks[block_id]
    assert block.ref_count == 0
    if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
        del self.hash_to_block_id[block.hash]
    block.reset()                          # ref_count=1, hash=-1, tokens=[]
    self.used_block_ids.add(block_id)
    return block_id

def _deallocate_block(self, block_id):
    assert self.blocks[block_id].ref_count == 0
    self.used_block_ids.remove(block_id)
    self.free_block_ids.append(block_id)
```
**重点理解**：`_allocate_block` 不一定意味着「该 free 块从未用过」—— 它可能曾经被用过、已经释放，但 hash 索引里还指着它。如果该 hash 仍指向自己，就清掉索引。

### 练习 5.5（can_allocate）
实现 `can_allocate(seq) -> int`：
- 扫描 seq 的前 `num_blocks - 1` 块（**最后一块不哈希**，因为它可能未填满）：
  - 链式 hash 计算每个块的 hash
  - 在 `hash_to_block_id` 里查询；命中且 token_ids 一致 → `num_cached_blocks += 1`
  - 一旦 mismatch → break
- 命中的块若已经 in `used_block_ids`，意味着「不需要新分配」，所以 `num_new_blocks -= 1`
- 最终返回值规则：
  - 若 `len(free_block_ids) < num_new_blocks` → 返回 `-1`（容量不够）
  - 否则返回 `num_cached_blocks`

### 练习 5.6（allocate）
实现 `allocate(seq, num_cached_blocks)`：
- 对前 `num_cached_blocks` 块：
  - 命中 hash → 拿到 block_id
  - 若该 block_id 在 used 中：`block.ref_count += 1`（共享）
  - 若仍在 free 中（曾被释放但 hash 没失效）：`block.ref_count = 1`，从 free 中删除并加入 used
  - 把 block_id append 到 `seq.block_table`
- 对剩余块（`num_cached_blocks` 到 `num_blocks - 1`）：调用 `_allocate_block()` 拿新块
- 设置 `seq.num_cached_tokens = num_cached_blocks * block_size`

### 练习 5.7（deallocate）
实现 `deallocate(seq)`：
- **逆序**遍历 `seq.block_table`，每个块 `ref_count -= 1`，若降到 0 则 `_deallocate_block`
- 清空 `seq.block_table`、`seq.num_cached_tokens = 0`

> 为什么逆序？因为先释放后面的块，free_block_ids 队尾是后释放的；下次 popleft 拿到的是最早释放的「老」块，更容易命中其上面的 hash 索引（LIFO 在这个场景反而不利）。这是工程上的小优化，可以先实现正序看是否影响。

### 练习 5.8（can_append / may_append）
实现 decode 阶段的增量分配：
```python
def can_append(self, seq):
    return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

def may_append(self, seq):
    if len(seq) % self.block_size == 1:
        seq.block_table.append(self._allocate_block())
```
- 触发条件：序列长度模 block_size 等于 1 → 刚刚跨过 block 边界，需要一个新块
- `can_append` 返回 bool：是否有 free 块可用（如果不需要新块则返回 True）

### 练习 5.9（hash_blocks）
实现 `hash_blocks(seq)`：在 prefill / decode 完成后被 Scheduler.postprocess 调用，给「这一步刚刚被填满的所有块」建立 hash 索引：
```python
def hash_blocks(self, seq):
    start = seq.num_cached_tokens // self.block_size
    end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
    if start == end: return
    h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
    for i in range(start, end):
        block = self.blocks[seq.block_table[i]]
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h)
        block.update(h, token_ids)
        self.hash_to_block_id[h] = block.block_id
```
- 注意 `start - 1` 取上一个块的 hash 作为 prefix（链式）
- 只有「填满的整块」才哈希；最后一个不完整块跳过

### 练习 5.10（端到端单测）
写测试：
```python
bm = BlockManager(num_blocks=8, block_size=4)
seq1 = Sequence([1,2,3,4, 5,6,7,8, 9])    # 9 tokens, 3 blocks
n1 = bm.can_allocate(seq1)            # 0（首次，无命中）
bm.allocate(seq1, n1)
bm.hash_blocks(seq1)                  # 模拟 schedule 后

seq2 = Sequence([1,2,3,4, 5,6,7,8, 99])   # 前 8 tokens 与 seq1 完全一致
n2 = bm.can_allocate(seq2)            # 应该是 2（前 2 块命中）
bm.allocate(seq2, n2)
assert seq1.block_table[0] == seq2.block_table[0]
assert seq1.block_table[1] == seq2.block_table[1]
assert seq1.block_table[2] != seq2.block_table[2]   # 最后一块各自分配
```

---

## 6. 解答

> 参考实现：[`engine/block_manager.py`](nanovllm/engine/block_manager.py)

### 解答 5.2（哈希链）
- xxhash 是非加密哈希但极快，且 update 顺序敏感：先 prefix 再 token_ids 与「先 token 再 prefix」会得到不同结果。这里规约「prefix 在前」即可。
- `intdigest()` 返回 64 位整数，作为 dict key 没问题（碰撞概率 ~2^-64）。

### 解答 5.5 关键点（can_allocate）
- 「最后一块不哈希」：因为 prefill 完成后，最后一块可能还有 N < block_size 个 token，且未来会继续追加 —— 这种部分块如果哈希了，下一次新增 token 后 hash 就失效，反而扰乱索引。
- `num_new_blocks` 的更新：命中且**该物理块已被某序列持有（在 used 中）**时，本序列不再需要新分配；命中但该物理块在 free 队列里时，仍占用一个 free 块（只是不需要重新 hash）。

### 解答 5.6 关键点（allocate）
- 共享场景示例：seq A 持有 block 12（ref=1），seq B 提交相同前缀后 allocate → block 12 ref_count=2，B 的 block_table 也指向 12。
- 「曾释放但仍命中」的场景：seq A 完成后 block 12 ref=0 进入 free。这时 seq B 提交，hash_to_block_id 仍指着 12 → B 把 block 12 从 free 拉回 used，ref_count=1。

### 解答 5.7 关键点（deallocate）
- 必须先 `block.ref_count -= 1` 再判断 == 0：避免引用计数为正时误回收。
- 注意 hash 索引此时 **不立即删除**：让其他可能命中此前缀的新序列还能复用该物理块（块没被复写就还有效）。
- 真正清除 hash 索引发生在 `_allocate_block`：当这个 free 块被实际复用 / 重置时。

### 解答 5.8 关键点（may_append）
- decode 时长度从 N → N+1：
  - 若 N % block_size != 0：原最后一块还有空间，无需新块
  - 若 N % block_size == 0：原最后一块刚填满，N+1 后 `(N+1) % block_size == 1` → 需新块
- `can_append` 返回 bool 是为了 Scheduler 提前判断是否要抢占。

### 解答 5.9 关键点（hash_blocks）
- 调用时机：Scheduler.postprocess 里**每一步**都调一次，把这一步新填满的块加到 hash 索引。这样下一次新序列提交时立刻能命中。
- 当 `start == 0` 时 prefix 用 -1（chain 起点）；否则用 `seq.block_table[start-1]` 这个块的 hash 作为 prefix。

### 解答 5.10 答案
- seq1 首次：n1=0，分配 3 个全新块 [b0, b1, b2]。
- hash_blocks(seq1) 后：b0 与 b1 有 hash 索引（最后一块只有 1 token，不哈希）。
- seq2 前 8 个 token 与 seq1 一致：can_allocate 返回 2，前两块命中复用，第三块新分配。
- 共享后 b0、b1 的 ref_count 都变为 2。

### 常见坑
1. **忘记在 `_allocate_block` 里清掉旧 hash 索引** → 同一 block_id 在 hash_to_block_id 里挂着两个值，前缀复用 bug。
2. **deallocate 时不按逆序** → 通常没错，但偶尔在 hash 链结构上会带来「先释放上层 hash、下层还引用着」的迷惑日志。
3. **`hash_blocks` 在 chunked prefill 时**：seq.num_cached_tokens 与 num_scheduled_tokens 都会随多次 schedule 增长，公式必须基于「当前已 scheduled 的范围」算块边界。
4. **共享 block 的 token_ids 比对**：`hash_to_block_id` 命中后还要 `block.token_ids == token_ids` 校验 —— 哈希碰撞虽然罕见，但必须防。

---

## 7. 自检提问

- [ ] 我能解释为什么哈希链要带 prefix（不带会怎样？）
- [ ] 我能讲清「最后一块不哈希」的两个原因（不完整 + 会继续追加）
- [ ] 我能描述「曾释放但仍命中」场景下 ref_count 的变化路径
- [ ] 我能在 30 秒内说出 `can_append` 的触发条件
- [ ] 我能讲解 hash_blocks 调用时机及 prefix 的取值规则