# GRPO

Group Relative Policy Optimization（群体相对策略优化）

- Paper: [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- Paper: [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)

## 1. 简介

GRPO（Group Relative Policy Optimization，群体相对策略优化）是 DeepSeek 团队提出的一种强化学习算法，首次在 DeepSeekMath 论文中提出，后在 DeepSeek-R1 中大规模应用。其核心创新是**去除了 PPO 中的 Critic（价值）模型**，通过在同一问题上生成多个回复并做组内相对比较来估计优势函数，显著降低了训练的内存和计算开销。

> **GRPO 的本质思路**：通过在同一个问题上生成多条回答，把它们彼此之间做"相对比较"，来代替传统 PPO 中的"价值模型"。

### 1.1 PPO 的痛点

PPO 在 LLM 对齐中面临以下问题：

1. **需要 Critic 模型**：PPO 需要单独的价值模型来估计每个响应的值，使内存和计算要求加倍
2. **Critic 难训练**：价值模型训练复杂、容易出错，尤其对主观或细微评价的任务
3. **计算成本高**：4 个模型（Actor + Critic + Reward + Reference）同时运行，资源消耗大
4. **绝对奖励评估**：难以适应各种任务，推广性差

### 1.2 GRPO 如何解决

- **无 Critic 优化**：通过比较组内响应消除对价值模型的需求，显著减少计算开销
- **相对评估**：使用群体动力学评估某个响应相对于同组其他响应的表现
- **高效训练**：关注基于群体的优势，简化奖励估计过程

## 2. GRPO 目标函数

### 2.1 完整目标函数

$$J_{GRPO}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(\cdot|q)} \left[ \frac{1}{G} \sum_{i=1}^G \min\left(\frac{\pi_\theta(o_i|q)}{\pi_{old}(o_i|q)} A_i, \text{clip}\left(\frac{\pi_\theta(o_i|q)}{\pi_{old}(o_i|q)}, 1-\varepsilon, 1+\varepsilon\right) A_i\right) - \beta \mathbb{D}_{KL}(\pi_\theta \| \pi_{ref}) \right]$$

### 2.2 逐步拆解

目标函数由三部分组成：

#### (1) 期望值计算

$$\mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(\cdot|q)}$$

- 从训练数据分布 $P(Q)$ 中采样查询 $q$
- 对每个查询，从旧策略 $\pi_{\theta_{old}}$ 采样 $G$ 个候选响应 $\{o_1, o_2, ..., o_G\}$

#### (2) 裁剪目标（Clipped Objective）

$$\frac{1}{G} \sum_{i=1}^G \min\left(\frac{\pi_\theta(o_i|q)}{\pi_{old}(o_i|q)} A_i, \text{clip}\left(\frac{\pi_\theta(o_i|q)}{\pi_{old}(o_i|q)}, 1-\varepsilon, 1+\varepsilon\right) A_i\right)$$

- **策略比值**：$r_i = \frac{\pi_\theta(o_i|q)}{\pi_{old}(o_i|q)}$，衡量新旧策略的变化幅度
- **裁剪机制**：源自 PPO，限制策略比值在 $[1-\varepsilon, 1+\varepsilon]$ 之间，防止策略更新过大导致不稳定
- **min 操作**：取未裁剪项和裁剪项的较小值，保守更新

#### (3) KL 散度正则项

$$\beta \mathbb{D}_{KL}(\pi_\theta \| \pi_{ref})$$

- 确保新策略不会偏离参考策略太远
- $\beta$ 控制 KL 正则项的影响力度

### 2.3 核心创新：组内相对优势（Group Relative Advantage）

这是 GRPO 与 PPO 的**最大区别**：

$$\tilde{A}_i = \frac{r_i - \text{mean}(r_1, ..., r_G)}{\text{std}(r_1, ..., r_G)}$$

其中：
- $r_i$：对第 $i$ 个响应计算的奖励
- $\text{mean}(r_1, ..., r_G)$：该组响应的平均奖励
- $\text{std}(r_1, ..., r_G)$：该组奖励的标准差

**直觉理解**：

- 回答优于小组平均水平的 → 获得正优势 → 增大该动作概率
- 回答低于小组平均水平的 → 获得负优势 → 减小该动作概率
- 除以标准差 → 归一化，减少奖励尺度的影响，提高训练稳定性
- **鼓励组内竞争，推动模型产生更好的响应**

## 3. GRPO 训练流程

### 3.1 完整步骤

```
Step 1: 选择查询
  从训练数据集 P(Q) 中选择一个查询 q
  例如: "8 + 5 的总和是多少？"

Step 2: 生成一组响应
  用当前策略 π_θ 生成 G 个候选响应 {o_1, o_2, ..., o_G}
  例如 G=4:
    o_1: "答案是13。"
    o_2: "十三。"
    o_3: "是12。"
    o_4: "总数是13。"

Step 3: 计算每个响应的奖励
  用奖励函数（规则或模型）为每个响应打分
  例如:
    r_1 = 1.0  （正确且格式良好）
    r_2 = 0.9  （正确但不太正式）
    r_3 = 0.0  （错误答案）
    r_4 = 1.0  （正确且格式良好）

Step 4: 计算组内相对优势
  mean = (1.0 + 0.9 + 0.0 + 1.0) / 4 = 0.725
  std  = 0.411

  Ã_1 = (1.0 - 0.725) / 0.411 = +0.669  ← 高于平均，正优势
  Ã_2 = (0.9 - 0.725) / 0.411 = +0.426  ← 高于平均，正优势
  Ã_3 = (0.0 - 0.725) / 0.411 = -1.764  ← 低于平均，负优势
  Ã_4 = (1.0 - 0.725) / 0.411 = +0.669  ← 高于平均，正优势

Step 5: 使用裁剪更新策略
  对每个响应计算策略比值 r_i = π_θ(o_i|q) / π_old(o_i|q)
  计算 min(r_i × Ã_i, clip(r_i, 1-ε, 1+ε) × Ã_i)
  如果策略比值超出 [1-ε, 1+ε] 范围，则被裁剪

Step 6: 使用 KL 散度惩罚偏差
  添加 β × KL[π_θ || π_ref] 项
  防止更新后的策略偏离参考策略太远

Step 7: 梯度下降更新参数
  最小化 L_GRPO = -J_GRPO（取负号用梯度下降）
```

### 3.2 流程图

```
查询 q ──→ 策略 π_θ ──→ 生成 G 个响应 {o_1,...,o_G}
                              │
                              ↓
                    奖励函数打分 {r_1,...,r_G}
                              │
                              ↓
              组内归一化 Ã_i = (r_i - mean) / std
                              │
                              ↓
          计算策略比值 r_i = π_θ(o_i|q) / π_old(o_i|q)
                              │
                              ↓
         裁剪目标 min(r_i·Ã_i, clip(r_i,1-ε,1+ε)·Ã_i)
                              │
                              ↓
           减去 KL 惩罚 β × KL[π_θ || π_ref]
                              │
                              ↓
                    梯度下降更新 π_θ
```

## 4. GRPO 与 PPO 的详细对比

### 4.1 模型需求对比

| 组件 | PPO | GRPO |
|------|-----|------|
| 策略模型（Actor） | ✅ 需要训练 | ✅ 需要训练 |
| 价值模型（Critic） | ✅ 需要训练 | ❌ **不需要** |
| 奖励模型（Reward） | ✅ 需要推理 | ✅ 需要推理（或规则奖励） |
| 参考模型（Reference） | ✅ 冻结 | ✅ 冻结 |
| **模型总数** | **4 个** | **2-3 个** |
| **显存需求** | 高 | **显著降低** |

### 4.2 优势计算对比

| | PPO | GRPO |
|--|-----|------|
| 优势来源 | Critic 模型估计 V(s)，GAE 计算 | 组内奖励归一化 |
| 优势公式 | $A_t = \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}$ | $\tilde{A}_i = (r_i - \text{mean}) / \text{std}$ |
| 依赖额外模型 | 是（Critic） | 否 |
| 计算复杂度 | 高（需训练 Critic） | 低（只需统计归一化） |

### 4.3 整体对比

| 维度 | PPO | GRPO |
|------|-----|------|
| 核心思想 | 绝对优势估计 + clip 约束 | 组内相对优势 + clip 约束 |
| 奖励信号 | 绝对奖励分数 | 组内相对排名 |
| 采样方式 | 每轮 1 个响应 per prompt | 每轮 G 个响应 per prompt |
| 价值模型 | 必须 | 不需要 |
| 实现复杂度 | 高 | 低 |
| 显存消耗 | 高（4模型） | 低（2-3模型） |
| 适用场景 | 通用对齐 | 推理任务（规则奖励） |
| 代表应用 | InstructGPT, ChatGPT | DeepSeek-R1, DeepSeekMath |

## 5. GRPO 的奖励函数设计

GRPO 支持多种奖励函数：

### 5.1 规则奖励（DeepSeek-R1 使用）

- **准确性奖励**：基于响应的正确性（如数学答案是否正确）
- **格式奖励**：确保响应符合结构指南（如 `<think>...</think>` 中包含推理过程）
- **语言一致性奖励**：惩罚语言混合或不连贯的格式

### 5.2 模型奖励

与传统 RLHF 一样使用奖励模型打分，但不再需要 Critic 估计基线。

### 5.3 自定义奖励函数（TRL 示例）

```python
from trl import GRPOTrainer

# 规则奖励：奖励较长的完成
def reward_func(completions, **kwargs):
    return [float(len(completion)) for completion in completions]

# 格式奖励：检查 <think>...</think> 格式
import re
def format_reward_func(completions, **kwargs):
    pattern = r"^<think>.*?</think>\n<answer>.*?</answer>$"
    matches = [re.match(pattern, c[0]["content"]) for c in completions]
    return [1.0 if match else 0.0 for match in matches]
```

## 6. GRPO 的 KL 散度处理

GRPO 中 KL 散度的计算方式与 PPO 不同：

### PPO 的 KL 散度

在 loss 中作为独立惩罚项加入：

$$L_{PPO} = L_{actor} + c_1 L_{critic} + c_2 L_{KL}$$

### GRPO 的 KL 散度

直接在目标函数中作为正则项：

$$J_{GRPO} = \text{clip\_objective} - \beta \mathbb{D}_{KL}(\pi_\theta \| \pi_{ref})$$

KL 散度的具体计算（逐 token 近似）：

$$\mathbb{D}_{KL}(\pi_\theta \| \pi_{ref}) = \frac{\pi_{ref}(o_i|q)}{\pi_\theta(o_i|q)} - \log \frac{\pi_{ref}(o_i|q)}{\pi_\theta(o_i|q)} - 1$$

这种近似确保 KL 项可微，且计算高效（只需前向传播，不需要采样）。

## 7. GRPO 在 DeepSeek-R1 中的应用

DeepSeek-R1 的训练流程：

```
DeepSeek-V3-Base（预训练基座）
        │
        ↓ Stage 1: 冷启动 SFT
   少量长 CoT 数据微调
        │
        ↓ Stage 2: GRPO 强化学习
   规则奖励（准确性 + 格式 + 语言一致性）
   大规模 GRPO 训练，涌现推理能力
        │
        ↓ Stage 3: 拒绝采样 + SFT
   用 GRPO 训练后的模型生成数据
   结合其他领域数据做 SFT
        │
        ↓ DeepSeek-R1
```

**关键发现**：DeepSeek-R1 证明了仅用规则奖励 + GRPO，不依赖任何 SFT 数据，也能让模型自发涌现出 Chain-of-Thought 推理能力（包括自我反思、自我纠正等行为）。

## 8. GRPO 的优缺点

### 8.1 优点

1. **无需 Critic 模型**：显著降低显存和计算开销
2. **实现简单**：减少了模型管理和超参数调优的复杂性
3. **规则奖励友好**：特别适合有明确判定标准的任务（数学、代码）
4. **组内归一化**：自动适应不同难度的问题，减少奖励尺度的影响
5. **涌现能力强**：DeepSeek-R1 证明 GRPO 能激发推理能力

### 8.2 缺点

1. **采样成本**：每个 query 需生成 G 个响应（通常 G=4~16），增加推理成本
2. **组内方差**：如果 G 太小，优势估计方差大；G 太大又增加成本
3. **规则奖励局限**：对主观性强的任务（如创意写作），规则奖励难以设计
4. **离线数据不如 PPO**：GRPO 本质上还是 on-policy 的，不如 DPO 能利用纯离线数据
5. **相对评估局限**：当组内所有响应都很差时，相对比较可能导致"矮子里拔将军"

## 9. PPO → DPO → GRPO 演进路线

```
PPO (2020)                    DPO (2023)                    GRPO (2024)
┌─────────────────┐          ┌─────────────────┐          ┌─────────────────┐
│ 4 个模型         │          │ 2 个模型         │          │ 2-3 个模型       │
│ Actor + Critic  │          │ Policy + Ref    │          │ Policy + Ref    │
│ + Reward + Ref  │          │ (无 RM, 无 RL)  │          │ (+ Reward/规则) │
│                 │          │                 │          │                 │
│ 绝对优势        │    →     │ 隐式奖励        │    →     │ 组内相对优势     │
│ GAE + Critic    │          │ log概率差       │          │ (r-mean)/std    │
│                 │          │                 │          │                 │
│ 通用对齐        │          │ 通用对齐        │          │ 推理任务专精     │
│ 计算成本最高    │          │ 计算成本最低    │          │ 计算成本中等     │
│ 训练最不稳定    │          │ 训练最稳定      │          │ 训练较稳定       │
└─────────────────┘          └─────────────────┘          └─────────────────┘

核心演进逻辑:
  PPO → DPO:  去掉 RM + 去掉 RL（隐式奖励替代显式奖励）
  PPO → GRPO: 去掉 Critic（组内相对比较替代价值估计）
  DPO ↔ GRPO: DPO 不需要采样但需偏好对；GRPO 需采样但只需规则奖励
```

## 10. 常见面试问题

### Q1: GRPO 和 PPO 的最大区别是什么？

**最大区别是去掉了 Critic 模型**。PPO 用 Critic 估计价值函数 V(s) 来计算优势 A = R - V(s)，GRPO 通过在同一问题上生成 G 个响应并做组内归一化来估计优势 Ã = (r - mean) / std。这使得 GRPO 的显存开销减半，实现更简单。

### Q2: GRPO 的组内相对优势为什么要除以标准差？

除以标准差是一种归一化操作，有两个好处：
1. **消除奖励尺度影响**：不同任务/问题的奖励绝对值可能差异很大，归一化后优势值在同一量级
2. **稳定训练**：避免奖励值过大导致梯度爆炸或过小导致学习缓慢

### Q3: GRPO 适用于什么场景？

最适合**有明确判定标准的推理任务**（数学、代码、逻辑推理），因为这类任务容易设计规则奖励。对于主观性强的任务（创意写作、对话风格），规则奖励难以设计，可能仍需 PPO + RM。

### Q4: GRPO 中 G（组大小）怎么选择？

G 越大，优势估计越稳定（类似大数定律），但推理成本也越高。通常 G=4~16。DeepSeek-R1 实践中使用了较大的 G 值来确保优势估计的可靠性。

### Q5: GRPO 和 DPO 该选哪个？

- **有规则奖励**（数学、代码）→ GRPO（可直接用规则奖励，无需标注偏好对）
- **有偏好数据**（对话、安全）→ DPO（直接用离线偏好数据，无需在线采样）
- **两者可组合**：先用 GRPO 训推理能力，再用 DPO 做对齐微调

## 参考资料

- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- [TRL GRPO Trainer 文档](https://huggingface.co/docs/trl/main/en/grpo_trainer)
