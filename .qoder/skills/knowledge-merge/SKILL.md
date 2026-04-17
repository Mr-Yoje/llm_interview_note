---
name: knowledge-merge
description: 自动读取当前会话的聊天内容，智能识别主题并合并到对应的本地知识文件中。当用户要求归档问答、保存对话、整理知识、合并到笔记，或提到"记下来"、"保存"、"归档"等意图时使用。
---

# 知识问答智能归档

自动读取当前会话的聊天内容，智能识别主题，并合并到对应本地知识文件中。

## 工作流程

1. **读取当前会话**
   - 自动获取当前会话的完整聊天记录
   - 提取用户提问与模型回答内容
   - 识别对话中的核心知识点

2. **分析内容主题**
   - 提取问答中的核心关键词
   - 匹配下方主题映射表
   - 确定目标知识文件

3. **定位目标文件**
   - 根据主题映射找到对应 .md 文件
   - 确认文件存在（如不存在则提示用户）

4. **智能合并内容**
   - 分析目标文件的现有结构和知识点分布
   - 根据内容主题匹配最合适的插入位置
   - 将问答内容插入到相关知识点后面
   - 保持原有文件结构和层级关系

5. **格式化输出**
   - 使用统一问答归档格式

## 主题映射表

| 主题关键词 | 目标文件路径 |
|-----------|-------------|
| 分词、tokenization、tokenizer、BPE、WordPiece | `01.大语言模型基础/1.分词/1.分词.md` |
| 词向量、Word2Vec、embedding、向量表示 | `01.大语言模型基础/5.词向量/5.词向量.md` |
| 激活函数、ReLU、GELU、SwiGLU | `01.大语言模型基础/1.激活函数/1.激活函数.md` |
| jieba分词 | `01.大语言模型基础/2.jieba分词用法及原理/2.jieba分词用法及原理.md` |
| Attention、注意力机制、自注意力、MHA、MQA、GQA | `02.大语言模型架构/1.attention/1.attention.md` |
| LayerNorm、RMSNorm、归一化 | `02.大语言模型架构/2.layer_normalization/2.layer_normalization.md` |
| 位置编码、Positional Encoding、RoPE、ALiBi | `02.大语言模型架构/3.位置编码/3.位置编码.md` |
| Transformer、编码器、解码器、Encoder、Decoder | `02.大语言模型架构/Transformer架构细节/Transformer架构细节.md` |
| BERT、RoBERTa、ALBERT、DeBERTa | `02.大语言模型架构/bert细节/bert细节.md` |
| LLaMA、LLaMA2、LLaMA3、羊驼 | `02.大语言模型架构/llama系列模型/llama系列模型.md` |
| MoE、Mixture of Experts、专家混合、Switch Transformer | `02.大语言模型架构/1.MoE论文/1.MoE论文.md` |
| Top-k、Top-p、Temperature、解码策略、采样 | `02.大语言模型架构/解码策略（Top-k & Top-p & Temperature）/解码策略（Top-k & Top-p & Temperature）.md` |
| 数据并行、DP、DDP、DistributedDataParallel | `04.分布式训练/2.数据并行/2.数据并行.md` |
| 流水线并行、Pipeline Parallelism、GPipe | `04.分布式训练/3.流水线并行/3.流水线并行.md` |
| 张量并行、Tensor Parallelism、Megatron | `04.分布式训练/4.张量并行/4.张量并行.md` |
| 序列并行、Sequence Parallelism | `04.分布式训练/5.序列并行/5.序列并行.md` |
| DeepSpeed、ZeRO、Offload | `04.分布式训练/deepspeed介绍/deepspeed介绍.md` |
| MoE并行、专家并行 | `04.分布式训练/8.moe并行/8.moe并行.md` |
| 微调、Fine-tuning、SFT、有监督微调 | `05.有监督微调/1.微调/1.微调.md` |
| Prompt、提示工程、Prompting、上下文学习 | `05.有监督微调/2.prompting/2.prompting.md` |
| Adapter、Adapter-Tuning、适配器 | `05.有监督微调/3.adapter-tuning/3.adapter-tuning.md` |
| LoRA、低秩适应、参数高效微调、PEFT | `05.有监督微调/4.lora/4.lora.md` |
| 推理、Inference、vLLM、TGI、FasterTransformer | `06.推理/1.推理/1.推理.md` |
| RLHF、PPO、DPO、人类反馈强化学习 | `07.强化学习/1.rlhf相关/1.rlhf相关.md` |
| RAG、检索增强生成、向量检索、Embedding检索 | `08.检索增强rag/检索增强llm/检索增强llm.md` |
| Agent、智能体、ReAct、工具调用 | `08.检索增强rag/大模型agent技术/大模型agent技术.md` |
| 幻觉、Hallucination、事实性 | `09.大语言模型评估/1.大模型幻觉/1.大模型幻觉.md` |
| 评测、Benchmark、评估指标 | `09.大语言模型评估/1.评测/1.评测.md` |
| LangChain、LLM应用框架 | `10.大语言模型应用/1.langchain/1.langchain.md` |
| CoT、思维链、Chain-of-Thought | `10.大语言模型应用/1.思维链（cot）/1.思维链（cot）.md` |

## 问答归档格式

```markdown

---

## [问题摘要/标题]

**Q**: [用户原问题]

**A**: [模型回答内容]

*归档时间: [YYYY-MM-DD]*
```

## 使用示例

**用户输入**: "把刚才的问答保存到笔记"

**执行步骤**:
1. 读取当前会话聊天记录，提取关于LoRA原理的问答内容
2. 识别主题: LoRA、低秩适应
3. 匹配文件: `05.有监督微调/4.lora/4.lora.md`
4. 追加内容（使用上述格式）

## 智能插入规则

1. **分析文件结构**
   - 读取目标文件的目录结构和标题层级
   - 识别现有的知识点分块

2. **匹配插入位置**
   - 根据问答主题与文件中的章节标题匹配度
   - 找到最相关的知识点位置
   - 插入到该知识点的内容之后

3. **示例场景**
   - 若问答是关于 "LoRA 原理"，而文件中有 `## LoRA 原理` 章节
   - 则将问答插入到该章节内容的后面
   - 而非文件末尾

## 注意事项

- **自动读取**: 无需用户提供内容，自动从当前会话提取问答
- **智能插入**: 根据知识点匹配插入到相关位置，而非简单追加到文件末尾
- 若主题匹配多个文件，选择最相关的一个
- 若无法确定主题，询问用户目标文件
- 保持原有文件格式，仅追加不修改
- 添加分隔线 `---` 区分不同问答
