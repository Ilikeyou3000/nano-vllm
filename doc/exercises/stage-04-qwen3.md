# 阶段 4 · Qwen3 模型与权重加载

## 1. 结论

**用阶段 2~3 的算子拼出完整 `Qwen3ForCausalLM`，并实现一个能从 safetensors 正确加载 Qwen3-0.6B 权重的 loader。**

## 2. 为什么这么做

- 把零件拼成整机，是检验前面阶段的最强测试 —— 模型层加载完真实权重，跑出和 HuggingFace 接近的 logits，证明所有底层算子都正确。
- 权重加载里有一个关键 trick：原始权重里 q/k/v 是分开的，我们用了 `QKVParallelLinear` 合并存放，必须通过 `packed_modules_mapping` 做映射。
- 这是阶段 7 ModelRunner 的前置 —— 没有可加载的模型就跑不动 prefill。

## 3. 三大支柱任务

### 3.1 Qwen3 网络拼装
`Qwen3Attention` + `Qwen3MLP` + `Qwen3DecoderLayer` + `Qwen3Model` + `Qwen3ForCausalLM`。

### 3.2 packed_modules_mapping
类属性，告诉 loader「q_proj 应该被合并到 qkv_proj 的 q 分片」等。

### 3.3 通用 safetensors loader
`load_model(model, path)`：扫描所有 safetensors，按映射调用对应 Parameter 的 `weight_loader`。

## 4. 验收标准

- [ ] 加载 Qwen3-0.6B 真实权重，单 batch + 短 prompt（如长度 8）prefill 一次，输出的 logits 与 HuggingFace `Qwen3ForCausalLM` 在相同输入下 top-1 token 一致
- [ ] 全流程不抛 missing_keys / unexpected_keys
- [ ] `tie_word_embeddings=True` 时，lm_head 与 embed_tokens 共享权重

---

## 5. 练习题

### 练习 4.1（Qwen3Attention）
实现 [`models/qwen3.py`](nanovllm/models/qwen3.py) 中的 `Qwen3Attention`：
- 用 `QKVParallelLinear` 合并 Q/K/V 投影
- 用 `RowParallelLinear` 做 O 投影
- 包含 RoPE（通过 `get_rope`）与 Q/K 各自的 RMSNorm（**仅当 qkv_bias=False 时启用**）
- forward：
  ```
  qkv = qkv_proj(hidden_states)
  q, k, v = split([q_size, kv_size, kv_size])
  q = q.view(-1, num_heads, head_dim); k = ...; v = ...
  if not qkv_bias:
      q = q_norm(q); k = k_norm(k)
  q, k = rotary_emb(positions, q, k)
  o = attn(q, k, v)
  return o_proj(o.flatten(1, -1))
  ```

### 练习 4.2（Qwen3MLP）
实现 `Qwen3MLP`：
- gate 与 up 用 `MergedColumnParallelLinear` 合并
- down 用 `RowParallelLinear`
- 激活：`SiluAndMul`
- forward：`down_proj(SiluAndMul(gate_up_proj(x)))`

### 练习 4.3（Qwen3DecoderLayer · 残差链）
实现 `Qwen3DecoderLayer`：
- 包含 `self_attn`、`mlp`、`input_layernorm`、`post_attention_layernorm`
- forward 签名：`(positions, hidden_states, residual)` → `(hidden_states, residual)`
- 残差融合规则（**关键，容易写错**）：
  ```
  if residual is None:
      hidden, residual = input_ln(hidden), hidden          # 第 0 层特殊
  else:
      hidden, residual = input_ln(hidden, residual)        # 残差融合
  hidden = self_attn(positions, hidden)
  hidden, residual = post_attention_ln(hidden, residual)
  hidden = mlp(hidden)
  return hidden, residual
  ```

### 练习 4.4（Qwen3Model & Qwen3ForCausalLM）
实现 `Qwen3Model`（embed → 28 层 layer → 最终 norm）和 `Qwen3ForCausalLM`（model + lm_head）：
- `tie_word_embeddings=True` 时：`self.lm_head.weight.data = self.model.embed_tokens.weight.data`
- 提供 `compute_logits(hidden_states)` 接口（调 lm_head）
- 类属性 `packed_modules_mapping`：
  ```python
  packed_modules_mapping = {
      "q_proj": ("qkv_proj", "q"),
      "k_proj": ("qkv_proj", "k"),
      "v_proj": ("qkv_proj", "v"),
      "gate_proj": ("gate_up_proj", 0),
      "up_proj":   ("gate_up_proj", 1),
  }
  ```

### 练习 4.5（safetensors loader）
实现 [`utils/loader.py`](nanovllm/utils/loader.py)：
```python
def load_model(model, path):
    mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(f"{path}/*.safetensors"):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # 1. 命中 packed: 替换 weight_name 中的 q_proj 为 qkv_proj，调用对应分片 weight_loader
                # 2. 未命中: 调用 default_weight_loader
                ...
```
- 用 `model.get_parameter(param_name)` 拿到 Parameter
- 调用 `param.weight_loader(param, tensor, shard_id)`（packed）或 `param.weight_loader(param, tensor)`（普通）

### 练习 4.6（与 HuggingFace 对照）
写一个测试：
```python
hf_model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).cuda().eval()
my_model = Qwen3ForCausalLM(hf_config).cuda()
load_model(my_model, path)

ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]).cuda()
with torch.inference_mode():
    hf_logits = hf_model(ids).logits[0, -1]    # 最后 token
    # 你需要为 my_model 构造一个简单 Context（is_prefill=True, cu_seqlens_q=[0,8]）
    my_logits = my_model.compute_logits(my_model(ids[0], positions=torch.arange(8).cuda()))[0]

torch.testing.assert_close(hf_logits.argmax(), my_logits.argmax())
```

---

## 6. 解答

> 参考实现：[`models/qwen3.py`](nanovllm/models/qwen3.py)、[`utils/loader.py`](nanovllm/utils/loader.py)

### 解答 4.1（Qwen3Attention）
- `q_size = num_heads * head_dim`、`kv_size = num_kv_heads * head_dim`，注意 GQA 时两者不同
- `q_norm / k_norm` 是 Qwen3 特有的 head-level RMSNorm（在 head_dim 维度归一化），**只在 qkv_bias=False 时启用**（这是 Qwen3 的设计 —— 不带 bias 但带 head norm）
- `o_proj` 输入维度是 `total_num_heads * head_dim`，输出是 hidden_size
- `attn(q, k, v)` 返回 `[total_tokens, num_heads, head_dim]`，`flatten(1, -1)` 拍成 `[total_tokens, num_heads*head_dim]` 喂给 o_proj

### 解答 4.2（Qwen3MLP）
- `MergedColumnParallelLinear(hidden, [intermediate]*2)` 把 gate 和 up 合并 → 一次 matmul 出 `[*, 2*intermediate]`
- `SiluAndMul` 把它切成两半再点乘 → `[*, intermediate]`
- 最后 `RowParallelLinear` 投回 hidden_size

### 解答 4.3（DecoderLayer）
**residual 这一段是 Pre-LN + 融合残差的精髓**：
- 进入 layer：拿到的 `hidden_states` 是「上一层的输出 + 残差」之**未归一化**值
- `input_layernorm(hidden, residual)`：内部做 `hidden = norm(hidden + residual)`，同时把更新后的 `hidden + residual` 写回 residual
- 这样下层得到的 residual 已经是「上一层 attn 之前的累加和」，省了一次显存往返

### 解答 4.4（顶层）
- `tie_word_embeddings`：直接共享 `weight.data` —— 注意是 `data` 而非整个 Parameter，避免 lm_head 注册了一个新 Parameter 但底层 storage 共享
- `packed_modules_mapping` 的 value 形式：
  - QKV 用字符串 shard_id（"q"/"k"/"v"）
  - gate_up 用整数（0=gate, 1=up）
  这两种 weight_loader 接受不同类型的 shard_id，loader 转发即可

### 解答 4.5（loader）
关键骨架：
```python
def load_model(model, path):
    mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in mapping:
                    if k in weight_name:
                        v, shard_id = mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        param.weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:    # for-else：没有 break 才进
                    param = model.get_parameter(weight_name)
                    loader = getattr(param, "weight_loader", default_weight_loader)
                    loader(param, f.get_tensor(weight_name))
```
- 注意 `safe_open(file, "pt", "cpu")` 把权重读到 CPU；`weight_loader` 内部 `param_data.copy_(loaded_weight)` 会自动搬到 GPU（如果 param 在 GPU）。
- `for-else` 是 Python 特色写法：`for` 循环 break 出来不会进 else，只有正常跑完才进。

### 解答 4.6（与 HF 对照）
- 误差来源：bfloat16 累加顺序不同 → 单 token 输出 logits 量级可能差 0.01，但 argmax 应当一致。
- 如果 argmax 不一致，按以下顺序排查：
  1. RoPE：`base/theta` 设错；cos/sin 排布与 HF 不同
  2. q_norm/k_norm：忘记加（Qwen3 才有，Qwen2 没有）
  3. attention_bias：Qwen3 默认 `attention_bias=True`，但实际 0.6B 是 `False`；以 hf_config 为准
  4. tie_word_embeddings：忘了共享 lm_head

### 常见坑
- `attention_bias` 在不同 Qwen3 版本不一样：直接 `getattr(config, 'attention_bias', True)` 读，别硬编码
- 权重文件可能被分片成多个 safetensors（large 模型），loader 必须遍历所有文件
- `model.get_parameter(name)` 要求 name 是 dot-path（例如 "model.layers.0.mlp.down_proj.weight"），与 state_dict key 一致

---

## 7. 自检提问

- [ ] 我能解释 packed_modules_mapping 为什么需要存在
- [ ] 我能讲清 DecoderLayer 里 residual 的融合写法（哪一行更新了 residual）
- [ ] 我能说出 q_norm / k_norm 在 Qwen3 中的作用
- [ ] 我能解释 `tie_word_embeddings` 共享权重时为什么用 `.data =` 而不是 `lm_head.weight = embed.weight`
- [ ] 我能描述 safetensors loader 的两条加载路径（packed vs default）