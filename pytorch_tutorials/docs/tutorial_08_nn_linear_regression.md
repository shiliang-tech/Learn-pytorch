# 第 08 课：nn.Module 版线性回归

> 这一页只讲本节知识点和常用写法。练习题在 `lessons` 目录，答案在 `answers` 目录。

## 核心概念

- nn.Module 是 PyTorch 组织模型参数和前向计算的标准方式。
- nn.Linear(in_features, out_features) 实现 y=xW^T+b。
- optimizer 负责根据梯度更新参数，常见有 SGD、Adam。

## 关键写法详解

### 1. from torch import nn

```python
from torch import nn
nn.Module
nn.Linear
```

说明：

`from torch import nn` 后，可以用 `nn.Module`、`nn.Linear`、各种 loss 和层。
- 训练时关注顺序：先前向得到输出，再算 loss，然后清空旧梯度、反向传播、更新参数。顺序乱了通常不会得到正确训练。
- 涉及模型层时，把每一层都看成一次 shape 变换；不确定时在 forward 中临时打印中间结果 shape。

### 2. nn.Linear(in_features, out_features)

```python
nn.Linear(in_features, out_features)
[batch, in_features]
[batch, out_features]
```

说明：

`nn.Linear(in_features, out_features)` 输入 shape 是 `[batch, in_features]`，输出是 `[batch, out_features]`。
- 写这类代码时，第一步先确认张量形状。PyTorch 的很多报错不是公式错了，而是某一维没有对齐。
- 涉及模型层时，把每一层都看成一次 shape 变换；不确定时在 forward 中临时打印中间结果 shape。
- 数据管道的重点是第 0 维样本数一致。进入训练循环后，每个 batch 都应该能直接喂给模型。

### 3. model.parameters()

```python
model.parameters()
```

说明：

`model.parameters()` 会返回模型里需要优化的参数，直接交给 optimizer。
- 训练时关注顺序：先前向得到输出，再算 loss，然后清空旧梯度、反向传播、更新参数。顺序乱了通常不会得到正确训练。

### 4. nn.MSELoss()

```python
nn.MSELoss()
```

说明：

`nn.MSELoss()` 适合回归，预测和标签 shape 通常要一致。
- 写这类代码时，第一步先确认张量形状。PyTorch 的很多报错不是公式错了，而是某一维没有对齐。
- 训练时关注顺序：先前向得到输出，再算 loss，然后清空旧梯度、反向传播、更新参数。顺序乱了通常不会得到正确训练。

### 5. torch.optim.SGD(model.parameters(), lr=...)

```python
torch.optim.SGD(model.parameters(), lr=...)
```

说明：

`torch.optim.SGD(model.parameters(), lr=...)` 是最基础优化器；Adam 通常更省调参。

### 6. opt.zero_grad(); loss.backward(); opt.step()

```python
opt.zero_grad(); loss.backward(); opt.step()
```

说明：

标准顺序是 `opt.zero_grad(); loss.backward(); opt.step()`。
- 训练时关注顺序：先前向得到输出，再算 loss，然后清空旧梯度、反向传播、更新参数。顺序乱了通常不会得到正确训练。
- 和梯度有关的写法要小心计算图是否被断开。用于打印日志时可以 `.item()`，但参与训练的张量不要过早转成 Python 数字。

### 7. model.weight

```python
model.weight
model.bias
```

说明：

`model.weight` 和 `model.bias` 可以查看 Linear 的参数。
- 涉及模型层时，把每一层都看成一次 shape 变换；不确定时在 forward 中临时打印中间结果 shape。

### 8. nn.Sequential(...)

```python
nn.Sequential(...)
nn.Module
```

说明：

用 `nn.Sequential(...)` 可以快速堆简单模型；复杂模型建议自定义 `nn.Module`。

### 9. print(model)

```python
print(model)
```

说明：

训练前可用 `print(model)` 查看模型结构。

### 10. 概念和经验

说明：

回归输出层一般不加 sigmoid/softmax，直接输出连续值。
- 分类任务要分清 logits 和概率。大多数 PyTorch loss 直接接收 logits，只有推理或展示结果时才需要 sigmoid/softmax。


## 本节任务

- 用 nn.Linear(1, 1) 拟合 y=-3x+0.5。
- 使用 MSELoss 和 SGD。
- 训练结束后打印 weight、bias 和 loss。
