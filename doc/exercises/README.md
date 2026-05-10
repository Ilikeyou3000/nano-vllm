# Nano-vLLM 复现练习册 · 总览

> **一句话结论**：本目录把 [`学习计划.md`](doc/学习计划.md) 的 11 个阶段拆成 11 份独立练习，每份先给结论、再给任务、最后给解答，按顺序完成即可从零复现 Nano-vLLM。

## 使用方式（3 步）

1. **按阶段顺序打开对应文件**（必须顺序，前置阶段会成为后置阶段的依赖）。
2. **先只读「练习」部分**自行动手，卡壳超过 30 分钟再看「解答」。
3. **解答只看思路与验收**，代码自己敲一遍，避免复制。

## 阶段索引

| # | 主题 | 文件 | 难度 | 预计耗时 |
|---|---|---|---|---|
| 0 | 环境与前置知识 | [`stage-00-setup.md`](doc/exercises/stage-00-setup.md) | ★ | 0.5d |
| 1 | 骨架与数据结构 | [`stage-01-skeleton.md`](doc/exercises/stage-01-skeleton.md) | ★ | 0.5d |
| 2 | 基础算子层（单卡） | [`stage-02-layers.md`](doc/exercises/stage-02-layers.md) | ★★ | 1.5d |
| 3 | Context 与 KV Cache 算子 | [`stage-03-kvcache.md`](doc/exercises/stage-03-kvcache.md) | ★★★ | 1d |
| 4 | Qwen3 模型与权重加载 | [`stage-04-qwen3.md`](doc/exercises/stage-04-qwen3.md) | ★★ | 1d |
| 5 | BlockManager（Paged + Prefix Cache） | [`stage-05-blockmanager.md`](doc/exercises/stage-05-blockmanager.md) | ★★★★ | 1d |
| 6 | Scheduler（Continuous Batching） | [`stage-06-scheduler.md`](doc/exercises/stage-06-scheduler.md) | ★★★★ | 0.5d |
| 7 | ModelRunner（单卡 eager） | [`stage-07-modelrunner.md`](doc/exercises/stage-07-modelrunner.md) | ★★★ | 1d |
| 8 | LLMEngine 与对外 API | [`stage-08-llmengine.md`](doc/exercises/stage-08-llmengine.md) | ★★ | 0.5d |
| 9 | Tensor Parallelism | [`stage-09-tp.md`](doc/exercises/stage-09-tp.md) | ★★★★ | 1d |
| 10 | CUDA Graph 优化 | [`stage-10-cudagraph.md`](doc/exercises/stage-10-cudagraph.md) | ★★★ | 0.5d |
| 11 | 性能压测与调优 | [`stage-11-bench.md`](doc/exercises/stage-11-bench.md) | ★★ | 0.5d |

## 文档组织方式（金字塔原理）

每份练习文件结构一致，**自顶向下**展开：

```
1. 结论（一句话说清这个阶段要做什么）
2. 为什么要做（和全局的关系）
3. 三大支柱任务（本阶段必须完成的 3 个里程碑任务）
   3.1 任务 A
   3.2 任务 B
   3.3 任务 C
4. 验收标准（可量化）
5. 练习题（动手清单）
6. 解答（思路 + 关键实现要点 + 坑）
7. 自检提问（能答上来再进入下一阶段）
```

## 如何对照参考实现

每一题在「解答」里都会标注参考源码位置，例如 [`nanovllm/layers/layernorm.py`](nanovllm/layers/layernorm.py)。**建议动手完成练习之后再点开对照**，否则体验等同于抄写。

## 推荐节奏

- 工作日每天 1 个阶段，周末把阶段 5~7 这种硬骨头留作大块时间。
- 每完成一个阶段，在文末的「自检提问」里给自己打 ✅/❌，❌ 的条目回到该阶段补齐。

祝顺利完成从 0 到 1 的复现之旅。