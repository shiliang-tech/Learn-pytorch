# 第 20 课：BatchNorm 与 Dropout 的 train/eval 差异

> 这一页只讲本节知识点和常用写法。练习题在 `lessons` 目录，答案在 `answers` 目录。

## 核心概念

- BatchNorm 在训练时使用当前 batch 统计，并更新 running_mean/running_var。
- eval 模式下 BatchNorm 使用累计统计，Dropout 则关闭随机丢弃。
- 这就是为什么验证和推理阶段一定要调用 model.eval()。

## 关键写法详解

### 1. nn.BatchNorm1d(num_features)

```python
nn.BatchNorm1d(num_features)
[batch, features]
```

说明：

`nn.BatchNorm1d(num_features)` 常用于 MLP 的 `[batch, features]`。
- 写这类代码时，第一步先确认张量形状。PyTorch 的很多报错不是公式错了，而是某一维没有对齐。
- 数据管道的重点是第 0 维样本数一致。进入训练循环后，每个 batch 都应该能直接喂给模型。

### 2. nn.BatchNorm2d(num_channels)

```python
nn.BatchNorm2d(num_channels)
[N, C, H, W]
```

说明：

`nn.BatchNorm2d(num_channels)` 常用于 CNN 的 `[N, C, H, W]`。
- 写这类代码时，第一步先确认张量形状。PyTorch 的很多报错不是公式错了，而是某一维没有对齐。
- 数据管道的重点是第 0 维样本数一致。进入训练循环后，每个 batch 都应该能直接喂给模型。

### 3. 概念和经验

说明：

BatchNorm 训练时用当前 batch 统计，同时更新 running_mean/running_var。
- 数据管道的重点是第 0 维样本数一致。进入训练循环后，每个 batch 都应该能直接喂给模型。

### 4. 概念和经验

说明：

BatchNorm eval 时使用 running_mean/running_var，不再用当前 batch 统计。
- 数据管道的重点是第 0 维样本数一致。进入训练循环后，每个 batch 都应该能直接喂给模型。

### 5. 概念和经验

说明：

Dropout train 时随机，eval 时关闭。

### 6. model.train()

```python
model.train()
model.eval()
```

说明：

因此训练前 `model.train()`，验证/推理前 `model.eval()` 是必须习惯。

### 7. 概念和经验

说明：

小 batch 下 BatchNorm 统计可能不稳定，可考虑 LayerNorm 或 GroupNorm。
- 数据管道的重点是第 0 维样本数一致。进入训练循环后，每个 batch 都应该能直接喂给模型。

### 8. 概念和经验

说明：

BatchNorm 一般放在 Linear/Conv 后、激活函数前或后，具体结构可按常见架构习惯。
- 涉及模型层时，把每一层都看成一次 shape 变换；不确定时在 forward 中临时打印中间结果 shape。
- 数据管道的重点是第 0 维样本数一致。进入训练循环后，每个 batch 都应该能直接喂给模型。

### 9. 概念和经验

说明：

比较两个输出是否一样时要先固定模式，否则 Dropout 会让结果随机。

### 10. 概念和经验

说明：

不要在验证阶段忘记 eval，否则指标会飘。


## 本节任务

- 构建 Linear-BatchNorm-ReLU-Dropout-Linear 网络。
- 喂入同一个 batch，比较 train 和 eval 下输出差异。
- 打印 BatchNorm 的 running_mean。
