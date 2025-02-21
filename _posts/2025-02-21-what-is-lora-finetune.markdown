---
layout: post
title: "What is lora finetune"
date: 2025-02-21
---
LoRA是大模型微调方法的一种，它的特点是只在模型的 部分权重（如 QKV 矩阵） 上 添加可训练参数
通过 低秩矩阵（A×B） 来优化参数更新
优点：
极大降低显存消耗（7B 只需 10GB）
适用于多任务 LoRA 适配器切换
训练速度快

例如在 Transformer 里，自注意力（Self-Attention）计算：
Y=XW，
其中 X 是input, W是原始模型的权重矩阵（全连接层）.
传统的Fine-tuning就是直接对 W 进行梯度更新，导致需要存储整个 W 的更新版本，显存占用极大。

LoRA 关键思想：
不直接更新 W，而是 用两个小矩阵 $A$ 和 $B$ 近似建模 W 的变化：
$W' = W + \Delta W$
$\Delta W = AB$

其中：
$A \in \mathbb{R}^{d \times r}$
$B \in \mathbb{R}^{r \times d}$
$r \ll d$（低秩），一般 r=4, 8, 16，远小于 d。
所以只需要训练A 和 B，大幅减少训练参数量，$AB$近似$\Delta W$, 使得最终$W'$仍然能适应新任务。
训练时，只更新A和B， W保持冻结。
推理时，计算$W+AB$, 但A，B远小于W，开销极小。

代码简单演示一下如何在transformer的q_proj里加入LoRA
在 Transformer 里，q_proj 是 nn.Linear 层

```python
import torch
import torch.nn as nn
import math

class LoRAQProj(nn.Module):
    def __init__(self, hidden_size, r=16, lora_alpha=16):
        super().__init__()
        self.hidden_size = hidden_size
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r  # LoRA 影响力
        
        # 原始 Q 投影层（冻结）
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # LoRA 适配器：A 和 B
        self.lora_A = nn.Linear(hidden_size, r, bias=False)  # 低秩 A
        self.lora_B = nn.Linear(r, hidden_size, bias=False)  # 低秩 B
        
        # 初始化 LoRA 参数
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)  # B 矩阵初始化为 0

    def forward(self, x):
        """
        计算 Self-Attention 里的 Query 矩阵：
            Q = X * (W_q + AB)
        """
        base_output = self.q_proj(x)  # 原始投影
        lora_output = self.lora_B(self.lora_A(x)) * self.scaling  # LoRA 适配器
        return base_output + lora_output  # 总输出

# 测试模型
hidden_size = 512
batch_size = 4
seq_len = 10

x = torch.randn(batch_size, seq_len, hidden_size)  # 输入数据
model = LoRAQProj(hidden_size)
output = model(x)

print("LoRA Q-Projection Output Shape:", output.shape)  # (4, 10, 512)
```

### 训练LoRA适配器
训练时，冻结self.q_proj, 只训练lora_A 和 lora_B

```python
# 训练 LoRA
optimizer = torch.optim.AdamW(
    [p for n, p in model.named_parameters() if "lora" in n], lr=1e-4
)

for epoch in range(10):
    for batch in dataloader:  # 假设 dataloader 提供训练数据
        optimizer.zero_grad()
        output = model(batch["input_ids"])
        loss = loss_function(output, batch["labels"])  # 计算损失
        loss.backward()
        optimizer.step()

```

### 推理时合并LoRA
LoRA 训练完成后，我们需要合并 A, B 到 q_proj
计算 $W_{q}' = W_{q} + AB$,
这样，可以移除A，B，只保留$W_{q}'$, 加速推理

```python
def merge_lora(model):
    """
    合并 LoRA 适配器到原始权重：
    W_q' = W_q + AB
    """
    with torch.no_grad():
        model.q_proj.weight += (model.lora_B.weight @ model.lora_A.weight) * model.scaling

    # 移除 LoRA 适配器
    del model.lora_A
    del model.lora_B

    return model

# 进行推理时合并 LoRA
merged_model = merge_lora(model)
```

不过实际中，不需要我们自己去写这些代码，可以用unsloth, LLaMA-Factory 等框架来实现。
