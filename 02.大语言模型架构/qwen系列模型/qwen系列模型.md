# QWEN 系列模型详解

> QWEN（通义千问）是阿里云通义实验室开源的大语言模型系列，涵盖基础模型、对话模型、多模态模型等多个版本。

---

## 1. QWEN 系列发展时间线

| 版本 | 发布时间 | 模型规模 | 核心特点 |
|------|---------|---------|---------|
| **QWEN** | 2023.08 | 7B / 14B / 72B | 初代模型，双语预训练 |
| **QWEN-1.5** | 2024.02 | 0.5B~110B | 全尺寸覆盖，代码/数学增强 |
| **QWEN2** | 2024.06 | 0.5B~72B + MoE | GQA 引入，推理优化 |
| **QWEN2.5** | 2024.09 | 0.5B~72B | 指令跟随、长上下文全面提升 |
| **QWEN-Coder** | 2024.04 | 1.8B~32B | 代码专用模型 |
| **QWEN-Math** | 2024.04 | 1.5B~72B | 数学推理专用模型 |
| **Qwen3** | 2025.05 | 0.6B~235B-A22B | 思考模式（Thinking/Non-Thinking），稠密+MoE双轨 |
| **Qwen3-Next** | 2025.09 | 80B-A3B | 超稀疏MoE + 混合注意力架构，极致推训效率 |
| **Qwen3.5** | 2026.02 | 0.8B~397B-A17B | 统一多模态基础，201语言支持，RL规模化泛化 |
| **Qwen3.6** | 2026.04 | 35B-A3B / 27B | Agentic Coding、思考保持(Thinking Preservation)、原生智能体能力 |
| **QWEN-VL** | 2023.08 | 7B / 9B | 视觉语言多模态 |
| **QWEN-Audio** | 2024.07 | - | 音频理解多模态 |

---

## 2. 核心架构概览

### 2.1 统一架构设计

QWEN 全系列采用 **Decoder-Only** 架构，核心设计哲学与 LLaMA 类似，但在**词表设计、长上下文、多语言**等方面做了大量优化。

```
输入 → Token Embedding → N × QWEN Block → RMSNorm → LM Head → 输出
                              ↓
                    ┌──────────────────┐
                    │  Self-Attention  │
                    │  (GQA/Full MHA)  │
                    └──────────────────┘
                              ↓
                    ┌──────────────────┐
                    │      FFN         │
                    │   (SwiGLU)       │
                    └──────────────────┘
```

### 2.2 各代架构对比

| 特性 | QWEN (初代) | QWEN-1.5 | QWEN2 | QWEN2.5 |
|------|------------|----------|-------|---------|
| **注意力** | Full MHA | Full MHA | **GQA** | **GQA** |
| **位置编码** | RoPE | RoPE | RoPE | RoPE |
| **激活函数** | SwiGLU | SwiGLU | SwiGLU | SwiGLU |
| **归一化** | RMSNorm | RMSNorm | RMSNorm | RMSNorm |
| **上下文长度** | 8K / 32K | 32K | 32K / 128K | 32K / 128K |
| **词表大小** | 151,936 | 151,936 | 151,936 | 151,936 |
| **滑动窗口** | ❌ | ❌ | ✅ (4K) | ✅ (4K) |
| **注意力实现** | Flash Attention | Flash Attention | Flash Attention 2 | Flash Attention 2 |

---

## 3. QWEN2 架构详解

QWEN2 是目前最广泛使用的基座版本，其架构设计代表了 QWEN 系列的核心思想。

### 3.1 模型参数配置

| 模型 | 层数 | 隐藏维度 | 头数 | KV头数 | FFN中间维度 | 总参数量 |
|------|------|---------|------|--------|------------|---------|
| QWEN2-0.5B | 24 | 896 | 14 | 2 | 2,368 | 0.49B |
| QWEN2-1.5B | 28 | 1,536 | 12 | 2 | 8,960 | 1.54B |
| QWEN2-7B | 28 | 3,584 | 28 | 4 | 18,944 | 7.62B |
| QWEN2-72B | 80 | 8,192 | 64 | 8 | 29,504 | 72.71B |
| QWEN2-57B-A14B (MoE) | 64 | 4,096 | 48 | 4 | - | 57.41B (激活14.2B) |

### 3.2 核心组件

#### (1) GQA (Grouped Query Attention)

QWEN2 引入了 **GQA** 替代全多头注意力，大幅降低 KV-Cache 内存占用：

```python
# QWEN2-7B 注意力配置
num_attention_heads = 28        # Query 头数
num_key_value_heads = 4         # KV 共享头数 (GQA)
group_size = 7                  # 每 7 个 Query 共享 1 组 KV

# KV-Cache 节省比例
full_mha_kv = 28                # 28 组 KV
gqa_kv = 4                      # 4 组 KV
节省比例 = (28 - 4) / 28 = 85.7%  # KV-Cache 减少 85.7%
```

#### (2) SwiGLU 激活函数

```
FFN(x) = (x · W_gate) ⊙ Swish(x · W_up) · W_down

其中：
- W_gate: [d, hidden_dim]     (门控投影)
- W_up:   [d, hidden_dim]     (上采样投影)
- W_down: [hidden_dim, d]     (下采样投影)
- hidden_dim 通常 = 2.67 × d  (非精确 4d，如 LLaMA2)
```

#### (3) RoPE + NTK-aware 扩展

```python
# QWEN2 使用 RoPE 位置编码，支持长上下文扩展
base = 1_000_000                # 比 LLaMA 更大的 base (LLaMA 是 10000)
                                # 更大的 base → 更好的长序列外推性

# 长上下文训练策略
原始训练长度：4K → 32K 逐步扩展
使用技术：NTK-aware interpolation + YaRN
```

**base 对比**：

| 模型 | RoPE base | 训练长度 | 效果 |
|------|----------|---------|------|
| LLaMA-2 | 10,000 | 4K | 外推性一般 |
| QWEN2 | **1,000,000** | 4K → 32K | 更好的长序列感知 |

#### (4) 滑动窗口注意力 (Sliding Window Attention)

```
QWEN2 部分层使用滑动窗口注意力：

局部注意力窗口大小 = 4096
全局注意力层 = 每 4 层中的 1 层

设计目的：
- 降低长序列的 Attention 计算复杂度 O(n²)
- 同时保留全局信息传递能力
```

### 3.3 与 LLaMA2 架构对比

| 对比项 | LLaMA2 | QWEN2 | 说明 |
|--------|--------|-------|------|
| **词表大小** | 32,000 | **151,936** | QWEN 多语言词表更大 |
| **位置编码 base** | 10,000 | **1,000,000** | QWEN 长序列外推更好 |
| **注意力** | GQA | **GQA** | 两者都采用 |
| **激活函数** | SwiGLU | **SwiGLU** | 相同 |
| **归一化** | RMSNorm | **RMSNorm** | 相同 |
| **上下文长度** | 4K | **32K/128K** | QWEN 原生更长 |
| **多语言** | 主要英文 | **中英双语优化** | QWEN 中文能力更强 |

---

## 4. Qwen3 / 3.5 / 3.6 新一代架构演进

### 4.1 Qwen3（2025.05）

Qwen3 是一次架构层面的重大升级，核心亮点包括：

#### (1) 思考模式（Thinking / Non-Thinking）

```
同一模型，两种推理模式：

Thinking Mode（思考模式）：
  - 内部产生思考链（Chain-of-Thought）
  - 适合数学推理、代码生成、复杂逻辑
  - 类似 DeepSeek-R1 的推理模式

Non-Thinking Mode（非思考模式）：
  - 直接输出，无中间推理
  - 适合简单问答、对话、翻译
  - 响应更快，Token 消耗更少
```

#### (2) 稠密 + MoE 双轨

| 模型 | 类型 | 参数量 | 激活参数 |
|------|------|--------|---------|
| Qwen3-0.6B | 稠密 | 0.6B | 0.6B |
| Qwen3-1.7B | 稠密 | 1.7B | 1.7B |
| Qwen3-4B | 稠密 | 4B | 4B |
| Qwen3-8B | 稠密 | 8B | 8B |
| Qwen3-14B | 稠密 | 14B | 14B |
| Qwen3-32B | 稠密 | 32B | 32B |
| Qwen3-30B-A3B | MoE | 30B | 3B |
| Qwen3-235B-A22B | MoE | 235B | 22B |

**MoE 架构特点**：

- 总参数大，但激活参数小 → 高吞吐、低成本
- 每 token 只激活部分专家，稀疏计算

---

### 4.2 Qwen3-Next（2025.09）

> 基于全新架构设计，追求极致训练与推理效率

**核心创新**：

1. **超稀疏 MoE**：80B 参数 / 仅激活 3B（极端稀疏化）
2. **混合注意力架构（Hybrid Attention）**：
   - 部分层使用 Full Attention（全局信息传递）
   - 部分层使用 Sliding Window Attention（局部高效计算）
   - 自适应融合两者的优势
3. **Gated Delta Networks**：在 FFN 中引入门控机制，减少冗余计算

**效率提升**：

```
传统 80B 稠密模型：推理显存 ~160GB
Qwen3-Next-80B-A3B：推理显存 ~15GB（仅 3B 激活）
                    效率提升 ~10 倍！
```

---

### 4.3 Qwen3.5（2026.02）

> 迈向原生多模态统一模型的关键一代

**核心突破**：

1. **统一多模态基础**：
   - 早期融合训练（Early Fusion）：文本和图像在初始阶段就混合训练
   - 数万亿多模态 Token 训练
   - 单个模型同时处理文本、图像、代码

2. **201 种语言覆盖**：
   - 从之前的 30+ 种语言扩展到 201 种语言和方言
   - 包括小语种、方言、地域变体

3. **RL 规模化泛化**：
   - 百万级 Agent 环境中的强化学习
   - 渐进式任务分布：从简单到复杂逐步增加难度
   - 鲁棒的真实世界适应能力

4. **高效混合架构**：
   - Gated Delta Networks + 稀疏 MoE
   - 高吞吐推理，低延迟和成本开销

**模型家族**：

| 模型 | 类型 | 参数量 | 激活参数 |
|------|------|--------|---------|
| Qwen3.5-0.8B | 稠密 | 0.8B | 0.8B |
| Qwen3.5-2B | 稠密 | 2B | 2B |
| Qwen3.5-4B | 稠密 | 4B | 4B |
| Qwen3.5-9B | 稠密 | 9B | 9B |
| Qwen3.5-27B | 稠密 | 27B | 27B |
| Qwen3.5-35B-A3B | MoE | 35B | 3B |
| Qwen3.5-122B-A10B | MoE | 122B | 10B |
| Qwen3.5-397B-A17B | MoE（旗舰） | 397B | 17B |

---

### 4.4 Qwen3.6（2026.04，最新）

> 基于社区反馈，聚焦稳定性和实用性

**核心升级**：

1. **Agentic Coding（智能体编码）**：
   - 处理前端工作流和仓库级代码推理
   - 更流畅、更精准的编码体验

2. **Thinking Preservation（思考保持）**：

   ```
   传统多轮对话：
     轮次1: 思考链... → 输出
     轮次2: 无思考链  → 需重新推理
   
   Qwen3.6 思考保持：
     轮次1: 思考链... → 输出
     轮次2: 复用思考链 → 快速迭代
   ```

   - 跨对话历史保留思考上下文
   - 减少迭代开发的推理开销

3. **原生智能体能力**：
   - 工具调用更精准（Function Calling）
   - 长上下文窗口支持

**当前开源模型**：

| 模型 | 类型 | 参数量 | 激活参数 | 亮点 |
|------|------|--------|---------|------|
| Qwen3.6-35B-A3B | MoE | 35B | 3B | Agentic Coding 旗舰 |
| Qwen3.6-27B | 稠密 | 27B | 27B | 27B 稠密模型旗舰编码能力 |

---

## 5. 词表与分词设计

### 5.1 词表特点

```
QWEN 词表大小: 151,936

构成分析：
├── 英文 token: ~30,000 (基于 BPE)
├── 中文 token: ~50,000+ (大量常用词/字)
├── 代码 token: ~20,000 (编程语言关键字)
├── 数字/符号: ~20,000
├── 特殊 token: ~100 (pad, eos, bos, 系统指令等)
└── 其他语言: ~30,000 (日文、韩文、阿拉伯文等)
```

### 5.2 与 LLaMA 词表对比

| 特性 | LLaMA2 | QWEN2 |
|------|--------|-------|
| 词表大小 | 32,000 | 151,936 |
| 中文覆盖 | ❌ 差（用多个byte表示一个汉字） | ✅ 好（直接有中文token） |
| 序列效率 | 中文文本需要 ~2-3 倍 token | 中文文本仅需 ~1.3 倍 token |

**示例**：编码"你好世界"

```
LLaMA2: [你, 好, 世, 界] → 实际需要 6-8 个 token（按字节切分）
QWEN2:  [你好, 世界] → 仅需 2-3 个 token（有中文词表）
```

### 5.3 特殊 Token 设计

| Token | 用途 |
|-------|------|
| `<\|endoftext\|>` | 文本结束标记 |
| `<\|im_start\|>` | 消息开始（对话格式） |
| `<\|im_end\|>` | 消息结束（对话格式） |
| `<\|system\|>` | 系统角色标识 |
| `<\|user\|>` | 用户角色标识 |
| `<\|assistant\|>` | 助手角色标识 |
| `<\|tool\|>` | 工具调用标识 |
| `<\|fim_prefix\|>` | 代码填充前缀 |
| `<\|fim_middle\|>` | 代码填充中间 |
| `<\|fim_suffix\|>` | 代码填充后缀 |

---

## 6. 长上下文扩展技术

QWEN 系列在长上下文方面做了大量工作，原生支持 32K/128K。

### 6.1 训练阶段

```
第一阶段：预训练（Pre-training）
  - 长度：4K
  - 数据：海量中英文语料

第二阶段：长上下文扩展
  - 长度：4K → 32K（或 128K）
  - 技术：
    1. NTK-aware scaled RoPE（调整 base 和 scale）
    2. 逐步增加训练长度（4K → 8K → 16K → 32K）
    3. 长文本数据过滤与构造
```

### 6.2 推理阶段

```python
# QWEN2 支持动态长度扩展
# 通过调整 rope_scaling 参数

config = {
    "rope_scaling": {
        "type": "yarn",           # 使用 YaRN 扩展
        "factor": 4.0,             # 4 倍扩展 (8K → 32K)
        "original_max_position_embeddings": 32768
    }
}
```

---

## 7. 模型结构代码

### 7.1 QWEN2-7B 结构

```python
Qwen2ForCausalLM(
  (model): Qwen2Model(
    (embed_tokens): Embedding(151936, 3584)     # 词嵌入层
    (layers): ModuleList(
      (0-27): 28 x Qwen2DecoderLayer(            # 28 层 Decoder
        (self_attn): Qwen2SdpaAttention(         # GQA 注意力
          (q_proj): Linear(3584, 3584, bias=True)
          (k_proj): Linear(3584, 512, bias=True)  # KV 头压缩: 3584/28*4 = 512
          (v_proj): Linear(3584, 512, bias=True)
          (o_proj): Linear(3584, 3584, bias=False)
          (rotary_emb): Qwen2RotaryEmbedding()    # RoPE
        )
        (mlp): Qwen2MLP(                          # SwiGLU FFN
          (gate_proj): Linear(3584, 18944, bias=False)
          (up_proj): Linear(3584, 18944, bias=False)
          (down_proj): Linear(18944, 3584, bias=False)
          (act_fn): SiLU()                         # Swish/SiLU 激活
        )
        (input_layernorm): Qwen2RMSNorm()
        (post_attention_layernorm): Qwen2RMSNorm()
      )
    )
    (norm): Qwen2RMSNorm()
  )
  (lm_head): Linear(3584, 151936, bias=False)     # 输出层
)
```

### 7.2 注意力计算伪代码

```python
def gqa_attention(q, k, v, kv_cache, mask):
    """
    q: [batch, seq_len, num_heads * head_dim]
    k: [batch, seq_len, num_kv_heads * head_dim]
    v: [batch, seq_len, num_kv_heads * head_dim]
    """
    # 1. 投影
    q = q_proj(q)    # [batch, seq, 3584] = [batch, seq, 28 * 128]
    k = k_proj(k)    # [batch, seq, 512] = [batch, seq, 4 * 128]
    v = v_proj(v)    # [batch, seq, 512] = [batch, seq, 4 * 128]

    # 2. 应用 RoPE
    q, k = apply_rotary_emb(q, k, cos, sin)

    # 3. KV-Cache 更新
    if kv_cache is not None:
        k = concat([kv_cache.k, k], dim=1)   # [batch, cache_len + seq, 512]
        v = concat([kv_cache.v, v], dim=1)

    # 4. GQA: 扩展 KV 以匹配 Query 头数
    # [batch, seq, 4 * 128] → [batch, seq, 28 * 128]
    k = repeat_kv(k, n_rep=7)   # 28 / 4 = 7
    v = repeat_kv(v, n_rep=7)

    # 5. 计算 Attention Score
    scores = matmul(q, k.transpose(-2, -1)) / sqrt(head_dim)
    scores = scores + mask
    attn = softmax(scores, dim=-1)

    # 6. 加权求和
    out = matmul(attn, v)       # [batch, seq, 28 * 128]
    out = o_proj(out)           # [batch, seq, 3584]

    return out, KVCache(k, v)
```

---

## 8. QWEN 多模态模型

### 8.1 QWEN-VL（视觉语言）

```
架构：
  视觉编码器（ViT） + QWEN 语言模型

处理流程：
  图像 → ViT → 视觉特征 → 投影层 → 与文本 token 拼接 → QWEN → 输出

特点：
  - 支持图像理解、图文对话
  - 图像分辨率：448×448
  - 支持多图输入
```

### 8.2 QWEN-Audio（音频理解）

```
架构：
  音频编码器 + QWEN 语言模型

特点：
  - 支持语音、音乐、环境音理解
  - 音频转文本（ASR）
  - 音频内容描述
```

---

## 9. 对话格式 (ChatML)

QWEN 使用 ChatML 格式组织对话：

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
你好！<|im_end|>
<|im_start|>assistant
你好！很高兴见到你，有什么我可以帮你的吗？<|im_end|>
<|im_start|>user
帮我写一首诗。<|im_end|>
<|im_start|>assistant
```

**与 Base 模型的区别**：

- Base 模型：纯文本续写
- Chat 模型：在上述格式上 SFT + RLHF

---

## 10. 模型选型建议

| 场景 | 推荐模型 | 理由 |
|------|---------|------|
| 端侧部署 | QWEN2.5-0.5B/1.5B/3B | 小参数，速度快 |
| 个人开发 | QWEN2.5-7B | 性价比最优 |
| 生产服务 | QWEN2.5-14B/32B | 效果与性能平衡 |
| 极致效果 | QWEN2.5-72B | 最强能力 |
| 代码任务 | QWEN-Coder-32B | 代码专用训练 |
| 数学推理 | QWEN-Math-72B | 数学专用训练 |
| 视觉任务 | QWEN2-VL-72B | 多模态理解 |
| 超长文档 | QWEN2.5-7B-128K | 128K 上下文 |

---

## 11. 总结

### QWEN 系列核心优势

1. **多语言能力强**：15万+词表，中文效率远超 LLaMA
2. **长上下文原生支持**：32K/128K，无需额外微调
3. **全尺寸覆盖**：0.5B~72B，覆盖从端侧到服务器的全场景
4. **推理效率高**：GQA + FlashAttention 2，KV-Cache 大幅降低
5. **生态完善**：Base / Chat / Coder / Math / VL / Audio 全覆盖

### 与 LLaMA 的关系

```
QWEN ≈ LLaMA 架构 + 更大词表 + 更大 RoPE base + 更长上下文 + 中文优化

继承优点：
  ✅ Decoder-Only
  ✅ RMSNorm
  ✅ SwiGLU
  ✅ GQA

关键差异：
  📌 词表: 32K → 152K（多语言）
  📌 RoPE base: 10K → 1M（长序列）
  📌 上下文: 4K → 32K/128K
  📌 中文: 字节级 → 词级（效率提升 2-3 倍）
```
