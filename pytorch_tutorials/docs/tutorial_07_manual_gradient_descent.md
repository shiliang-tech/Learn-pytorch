# 第 07 课：手写梯度下降：拟合 y=2x+1

> 这一页只讲本节知识点和常用写法。练习题在 `lessons` 目录，答案在 `answers` 目录。

## 核心概念

- 训练的基本循环是：前向计算、计算损失、反向传播、更新参数、清空梯度。
- 梯度下降用参数减去 learning_rate * gradient，让 loss 往下降方向移动。
- 先手写一次更新过程，可以更清楚 nn.Module 和 optimizer 后面在替你做什么。

## 关键写法详解

### 1. pred -> loss -> zero_grad -> backward -> step

```python
pred -> loss -> zero_grad -> backward -> step
step
```

说明：

最小训练循环是 `pred -> loss -> zero_grad -> backward -> step`，手写版则把 `step` 换成手动减梯度。
- 训练时关注顺序：先前向得到输出，再算 loss，然后清空旧梯度、反向传播、更新参数。顺序乱了通常不会得到正确训练。
- 和梯度有关的写法要小心计算图是否被断开。用于打印日志时可以 `.item()`，但参与训练的张量不要过早转成 Python 数字。

### 2. requires_grad=True

```python
requires_grad=True
w = torch.randn(1, 1, requires_grad=True)
```

说明：

手动参数需要 `requires_grad=True`，例如 `w = torch.randn(1, 1, requires_grad=True)`。
- 和梯度有关的写法要小心计算图是否被断开。用于打印日志时可以 `.item()`，但参与训练的张量不要过早转成 Python 数字。

### 3. pred = x @ w + b

```python
pred = x @ w + b
x
[batch, features]
```

说明：

线性回归预测常写成 `pred = x @ w + b`，其中 `x` 是 `[batch, features]`。
- 写这类代码时，第一步先确认张量形状。PyTorch 的很多报错不是公式错了，而是某一维没有对齐。
- 数据管道的重点是第 0 维样本数一致。进入训练循环后，每个 batch 都应该能直接喂给模型。

### 4. ((pred - y) ** 2).mean()

```python
((pred - y) ** 2).mean()
```

说明：

MSE 可手写为 `((pred - y) ** 2).mean()`。

### 5. with torch.no_grad(): w -= lr * w.grad

```python
with torch.no_grad(): w -= lr * w.grad
```

说明：

参数更新要放进 `with torch.no_grad(): w -= lr * w.grad`，否则更新本身也会被记录进计算图。
- 和梯度有关的写法要小心计算图是否被断开。用于打印日志时可以 `.item()`，但参与训练的张量不要过早转成 Python 数字。

### 6. w.grad.zero_()

```python
w.grad.zero_()
b.grad.zero_()
```

说明：

更新后要 `w.grad.zero_()` 和 `b.grad.zero_()`，否则下一轮梯度会叠加。
- 和梯度有关的写法要小心计算图是否被断开。用于打印日志时可以 `.item()`，但参与训练的张量不要过早转成 Python 数字。

### 7. lr

```python
lr
```

说明：

`lr` 太大 loss 可能变成 NaN 或震荡；太小则收敛很慢。
- 训练时关注顺序：先前向得到输出，再算 loss，然后清空旧梯度、反向传播、更新参数。顺序乱了通常不会得到正确训练。

### 8. 概念和经验

说明：

训练时可以每隔若干 epoch 打印 loss，观察是否下降。
- 训练时关注顺序：先前向得到输出，再算 loss，然后清空旧梯度、反向传播、更新参数。顺序乱了通常不会得到正确训练。

### 9. 概念和经验

说明：

真实项目更常用 optimizer，但手写一次能帮你理解 optimizer 到底替你做了什么。
- 训练时关注顺序：先前向得到输出，再算 loss，然后清空旧梯度、反向传播、更新参数。顺序乱了通常不会得到正确训练。


## 本节任务

- 生成 x 与 y=2x+1 的训练数据。
- 用两个 requires_grad 参数 w、b 手动训练。
- 每轮用 no_grad 更新参数，并把梯度清零。
