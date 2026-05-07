# DAPO

Decoupled Clip and Dynamic Sampling Policy Optimization（解耦裁剪与动态采样策略优化）

- Paper: [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/pdf/2503.14476)
- Source: 字节跳动 & 清华

## 1. 简介

DAPO（Decoupled Clip and Dynamic Sampling Policy Optimization）是字节跳动联合清华提出的强化学习算法，针对 GRPO 在**长思维链（Long CoT）推理**场景中暴露的四个核心问题（熵崩溃、无效梯度、样本级损失信号稀释、长度膨胀）进行改进，通过四大核心技术显著提升了推理任务的训练效果。

> **DAPO 的本质思路**：GRPO 在长 CoT 推理中有四类问题 → DAPO 针对性地逐个解决。

### 1.1 GRPO 在长 CoT 场景的问题

| 问题 | 表现 | 影响 |
|------|------|------|
| 熵崩溃 | 低概率 token 被 clip 上界卡死，策略越来越确定 | 失去探索能力，多样性差 |
| 无效梯度 | 太简单/太难的 prompt 让组内优势全为 0 | 训练效率下降 |
| 信号稀释 | 样本级 loss，长序列中好/坏 token 获得相同梯度 | 推理质量无法精确优化 |
| 长度膨胀 | 模型倾向生成更长回复来"碰运气" | 冗余推理，效率低下 |

## 2. DAPO 的四大核心技术

### 2.1 Clip-Higher（解耦裁剪上界）

**问题根源**：GRPO/PPO 的 clip 范围 $[1-\varepsilon, 1+\varepsilon]$ 对上下界一视同仁，但实际影响不对称：

```
低概率 token（如 p=0.01）：
  clip 上界: 0.01 × 1.2 = 0.012  ← 提升空间只有 0.002，太死了！
  
高概率 token（如 p=0.9）：
  clip 上界: 0.9 × 1.2 = 1.08   ← 本身已经饱和

→ 低概率 token 无法被充分探索 → 策略快速收敛到确定性行为 → 熵崩溃
```

**DAPO 方案**：上界更松，下界保持甚至更严

$$
\text{clip}(r_t, 1-\varepsilon_{low}, 1+\varepsilon_{high})
$$

其中 $\varepsilon_{high} > \varepsilon_{low}$，例如 $[0.8, 2.0]$。

- $\varepsilon_{high}$ 较大 → 上界放宽，让低概率 token 有更大提升空间
- $\varepsilon_{low}$ 不变或更严 → 防止策略退化

**效果**：生成熵保持更高，样本多样性更好，防止熵崩溃。

### 2.2 Dynamic Sampling（动态采样）

**问题根源**：太简单/太难的 prompt 让优势为 0，无梯度贡献

```
太简单: 8 个回答全部正确 → 所有 r_i 相同 → A_i 全为 0 → 无梯度
太难:   8 个回答全部错误 → 同上
→ 有效 batch 中有梯度的样本越来越少
```

**DAPO 方案**：过采样 + 过滤

```
1. 对 prompt 过采样（如采 2× 所需数量）
2. 过滤掉"全对"或"全错"的 prompt
3. 只保留"既有对又有错"的 prompt 组成训练 batch
```

**效果**：
- 每个 batch 都有有效梯度
- 收敛更快（虽然采样成本增加，但训练步数减少）

### 2.3 Token-Level Policy Gradient Loss（Token 级损失）

**问题根源**：GRPO 在**样本级别**计算 loss，即一个完整回复对应一个标量奖励

```
GRPO 的 loss:
  L = Σ_i A_i × (1/|y_i|) × Σ_t log π_θ(y_{i,t} | x, y_{i,<t})

问题: 一个长 CoT 回复中，有些 token 是好的推理步骤，有些是坏的
      但一个标量奖励 r_i 无法区分 → 好/坏 token 获得相同的梯度信号

例: "让我们逐步分析...[正确推理]...[错误跳步]...[凑数废话]...答案是42"
    reward = 1（最终答案正确）
    → 所有 token（包括错误跳步和凑数废话）都被强化 ❌
```

**DAPO 方案**：Token 级策略梯度

```
将奖励信号分配到 token 级别:
  L = Σ_i Σ_t A_i × log π_θ(y_{i,t} | x, y_{i,<t})

不再对序列长度取平均（去除 1/|y_i|）
→ 长/短回复获得同等的梯度权重
→ 避免"长回复被稀释"的问题
```

**GRPO vs DAPO 的 loss 对比**：

```
GRPO (样本级):
  L = (1/G) × Σ_i A_i × (1/|y_i|) × Σ_t log π(y_{i,t})
                   ↑ 对长度取平均，长序列被稀释

DAPO (Token 级):
  L = (1/Σ|y_i|) × Σ_i Σ_t A_i × log π(y_{i,t})
      ↑ 所有 token 平等，不受序列长度影响
```

### 2.4 Over-long Reward Shaping（过长奖励整形）

**问题根源**：模型倾向于生成更长的回复来"碰运气"

```
无惩罚时:
  更长的回答 → 更多机会猜对 → reward 可能更高
  → 模型学会"写得多"而非"写得好"
```

**DAPO 方案**：对超过长度阈值的回复施加惩罚

$$
r_i = \begin{cases} r(x, y_i) & \text{if } |y_i| \leq L_{max} \\ r_{penalty} & \text{if } |y_i| > L_{max} \end{cases}
$$

其中 $r_{penalty}$ 通常设为 0 或 -1。

**效果**：
- 抑制长度膨胀
- 鼓励简洁有效的推理

## 3. DAPO 目标函数

$$
J_{DAPO}(\theta) = \mathbb{E}_{(x, \{y_i\})} \left[ \frac{1}{\sum_{i=1}^G |y_i|} \sum_{i=1}^G \sum_{t=1}^{|y_i|} \min\left(\frac{\pi_\theta(y_{i,t}|\cdot)}{\pi_{old}(y_{i,t}|\cdot)} \hat{A}_i, \text{clip}\left(\frac{\pi_\theta(y_{i,t}|\cdot)}{\pi_{old}(y_{i,t}|\cdot)}, 1-\varepsilon_{low}, 1+\varepsilon_{high}\right) \hat{A}_i\right) \right]
$$

与 GRPO 目标函数的关键差异：

| 差异点 | GRPO | DAPO |
|--------|------|------|
| 归一化方式 | $1/G$ 按样本数 | $1/\sum|y_i|$ 按 token 总数 |
| Clip 范围 | 对称 $[1-\varepsilon, 1+\varepsilon]$ | 非对称 $[1-\varepsilon_{low}, 1+\varepsilon_{high}]$ |
| KL 约束 | 有 $\beta \cdot KL$ | **无**（推理任务不需要） |
| 采样策略 | 固定采样 | 动态采样（过滤无效 prompt） |
| 长度惩罚 | 无 | 过长奖励整形 |

## 4. DAPO 训练流程

```
Step 1: 选择查询并过采样
  从训练数据集中选择 prompt，过采样 2× 所需数量

Step 2: 生成一组响应
  用当前策略 π_θ 为每个 prompt 生成 G 个候选响应

Step 3: 计算奖励 + 过长惩罚
  用规则/模型对每个响应打分
  超过长度阈值的响应 → r = penalty

Step 4: 动态过滤
  过滤掉"全对"或"全错"的 prompt 组
  只保留"既有对又有错"的 prompt

Step 5: 计算组内相对优势
  Ã_i = (r_i - mean) / std    ← 与 GRPO 相同

Step 6: Token 级 clip 目标计算
  对每个 token 计算策略比值 r_t
  使用非对称 clip: [1-ε_low, 1+ε_high]
  计算 min(r_t × Ã_i, clip(r_t, 1-ε_low, 1+ε_high) × Ã_i)

Step 7: Token 级梯度更新
  对所有 token 等权求和（不按序列长度取平均）
  梯度下降更新 π_θ
```

## 5. DAPO 与 GRPO 的详细对比

### 5.1 算法对比

| 维度 | GRPO | DAPO |
|------|------|------|
| Clip 策略 | 对称 $[1-\varepsilon, 1+\varepsilon]$ | 非对称，上界更松 |
| 采样策略 | 固定采样 | 动态采样（过滤无效 prompt） |
| Loss 粒度 | 样本级（整个回复一个 loss） | Token 级（每个 token 独立梯度） |
| 长度控制 | 无 | 过长奖励整形 |
| 熵保持 | 容易崩溃 | Clip-Higher 保持多样性 |
| 梯度效率 | 可能全为 0 | 动态采样保证有效梯度 |
| KL 约束 | 有 KL 正则 | 去掉 KL（推理任务不需要） |
| 适用场景 | 通用 | 长思维链推理 |

### 5.2 优势计算对比

| | GRPO | DAPO |
|--|------|------|
| 优势公式 | $\tilde{A}_i = (r_i - \text{mean}) / \text{std}$ | 同 GRPO |
| 但 | 无长度惩罚 | 过长回复 → r = penalty |
| Loss 归一化 | $1/G$ 按样本 | $1/\sum|y_i|$ 按 token |

## 6. DAPO 的实验效果

在 AIME 2024 数学推理基准上：
- GRPO: ~30% 准确率
- DAPO: ~50% 准确率

四项技术的消融实验：
- Clip-Higher 贡献最大（防熵崩溃）
- Token-Level Loss 对长序列效果显著
- 动态采样提升训练效率
- 四个技术叠加效果 > 单独使用

## 7. GRPO 家族演进路线

```
PPO (2020)         GRPO (2024)          DAPO (2025)
┌──────────────┐   ┌──────────────┐    ┌──────────────┐
│ 4 个模型      │   │ 2-3 个模型    │    │ 2-3 个模型    │
│ 绝对优势      │ → │ 组内相对优势  │ → │ 组内相对优势  │
│ 对称 clip     │   │ 对称 clip     │    │ 非对称 clip   │
│ 样本级 loss   │   │ 样本级 loss   │    │ Token级 loss  │
│ 有 KL 约束    │   │ 有 KL 约束    │    │ 无 KL 约束    │
│ 固定采样      │   │ 固定采样      │    │ 动态采样      │
│ 通用对齐      │   │ 推理任务      │    │ 长CoT推理     │
└──────────────┘   └──────────────┘    └──────────────┘

核心演进逻辑:
  PPO → GRPO:  去掉 Critic（组内相对比较替代价值估计）
  GRPO → DAPO: 解决长CoT场景的四个问题
    1. 熵崩溃  → Clip-Higher
    2. 无效梯度 → Dynamic Sampling
    3. 信号稀释 → Token-Level Loss
    4. 长度膨胀 → Over-long Reward Shaping
```

### GRPO 家族其他变体

| 算法 | 核心改进 | 优势计算方式 |
|------|---------|------------|
| DAPO | 四大技术 | 比同一批平均 |
| VAPO | 加 Value 预测器 | $A_i = r_i - \alpha\bar{r} - (1-\alpha)V_\phi(x)$ |
| SRPO | 与过去自己比 | $A_i = r(y_i) - r(y_i^{ref})$ |
| GFPO | 排序/成对偏好 | $\log\sigma(\log\pi(y_i) - \log\pi(y_j))$ |

> 四种算法的唯一区别："A 是怎么算出来的？"采样、reward、backward 完全一致。

## 8. 常见面试问题

### Q1: DAPO 和 GRPO 的最大区别是什么？

**最大区别是 Token-Level Loss + Clip-Higher**。GRPO 在样本级别计算 loss，长序列中好/坏 token 获得相同梯度信号；DAPO 在 token 级别计算，且用非对称 clip 允许低概率 token 有更大提升空间。此外 DAPO 还增加了动态采样和过长惩罚。

### Q2: 为什么 DAPO 去掉了 KL 约束？

在推理任务中，策略偏离参考模型是**期望的行为**——模型需要学会新的推理模式，不应该被强制保持与 SFT 模型的相似性。KL 约束更适合通用对齐任务（防止模型说脏话等），而非推理能力提升。

### Q3: Clip-Higher 为什么能防止熵崩溃？

对称 clip 对低概率 token 的提升空间限制太死（如 p=0.01 最多提升到 0.012），导致策略无法探索新 token，逐渐收敛到确定性行为。Clip-Higher 放宽上界（如最多提升到 0.02），给低概率 token 更多探索空间，保持生成多样性。

### Q4: DAPO 适用于什么场景？

最适合**长思维链推理任务**（数学、代码、多步推理），因为四个改进都是针对长 CoT 的痛点设计的。对于通用对齐任务，GRPO + KL 可能更合适。

## 参考资料

- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/pdf/2503.14476)
- [DeepSeekMath (GRPO 原始论文)](https://arxiv.org/abs/2402.03300)
- [GRPO 家族算法演进脉络（DAPO/VAPO/SRPO/GFPO）](https://www.cnblogs.com/sddai/p/19569451)
