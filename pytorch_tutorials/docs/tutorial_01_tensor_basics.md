# 第 01 课：Tensor 入门：创建、运算、矩阵乘法

> 这一页只讲本节知识点和常用写法。练习题在 `lessons` 目录，答案在 `answers` 目录。

## 核心概念

- Tensor 是 PyTorch 的核心数据结构，可以把它理解成支持 GPU 和自动求导的多维数组。
- 标量是 0 维，向量是 1 维，矩阵是 2 维，更高维常用来表达 batch、通道、图像高宽、时间步等。
- 深度学习里的大多数计算最终都会落到张量运算：加减乘除、矩阵乘法、聚合统计。

## 关键写法详解

### 1. import torch

```python
import torch
```

说明：

`import torch` 是所有 PyTorch 脚本的起点，通常先放在文件顶部。

### 2. 常用写法

```python
torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
```

说明：

`torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)` 从 Python 列表创建张量，并指定浮点类型。
- 写这类代码时，第一步先确认张量形状。PyTorch 的很多报错不是公式错了，而是某一维没有对齐。
- 同时注意 dtype：模型输入通常是浮点张量，分类标签通常是 `torch.long`，二分类 BCE 标签通常是浮点 0/1。

### 3. torch.zeros(shape)

```python
torch.zeros(shape)
torch.ones(shape)
torch.full(shape, value)
```

说明：

`torch.zeros(shape)`、`torch.ones(shape)`、`torch.full(shape, value)` 分别创建全 0、全 1、指定值张量。
- 写这类代码时，第一步先确认张量形状。PyTorch 的很多报错不是公式错了，而是某一维没有对齐。

### 4. torch.arange(start, end, step)

```python
torch.arange(start, end, step)
torch.linspace(start, end, steps)
```

说明：

`torch.arange(start, end, step)` 创建等差整数序列；`torch.linspace(start, end, steps)` 创建等距浮点序列。

### 5. torch.randn(shape)

```python
torch.randn(shape)
torch.rand(shape)
```

说明：

`torch.randn(shape)` 创建标准正态随机张量；`torch.rand(shape)` 创建 0 到 1 的均匀随机张量。
- 写这类代码时，第一步先确认张量形状。PyTorch 的很多报错不是公式错了，而是某一维没有对齐。

### 6. x + y

```python
x + y
x - y
x * y
x / y
```

说明：

`x + y`、`x - y`、`x * y`、`x / y` 是逐元素运算，要求形状相同或可以广播。

### 7. x @ w

```python
x @ w
torch.matmul(x, w)
x.shape[1] == w.shape[0]
```

说明：

`x @ w` 或 `torch.matmul(x, w)` 是矩阵乘法；二维时要求 `x.shape[1] == w.shape[0]`。
- 写这类代码时，第一步先确认张量形状。PyTorch 的很多报错不是公式错了，而是某一维没有对齐。

### 8. x.T

```python
x.T
transpose(dim0, dim1)
permute(...)
```

说明：

`x.T` 可以转置二维矩阵；高维张量更常用 `transpose(dim0, dim1)` 或 `permute(...)`。

### 9. x.shape

```python
x.shape
x.ndim
x.numel()
```

说明：

`x.shape`、`x.ndim`、`x.numel()` 分别查看形状、维度数、元素总数。
- 写这类代码时，第一步先确认张量形状。PyTorch 的很多报错不是公式错了，而是某一维没有对齐。

### 10. x.mean()

```python
x.mean()
x.sum()
x.max()
x.min()
x.std()
```

说明：

`x.mean()`、`x.sum()`、`x.max()`、`x.min()`、`x.std()` 是最常见统计操作。

### 11. x.item()

```python
x.item()
```

说明：

`x.item()` 把只含一个元素的张量转成 Python 数字，常用于打印 loss。
- 训练时关注顺序：先前向得到输出，再算 loss，然后清空旧梯度、反向传播、更新参数。顺序乱了通常不会得到正确训练。

### 12. print(x, x.shape, x.dtype, x.device)

```python
print(x, x.shape, x.dtype, x.device)
```

说明：

`print(x, x.shape, x.dtype, x.device)` 是排查 PyTorch 代码最朴素也最有效的方法。
- 写这类代码时，第一步先确认张量形状。PyTorch 的很多报错不是公式错了，而是某一维没有对齐。
- 同时注意 dtype：模型输入通常是浮点张量，分类标签通常是 `torch.long`，二分类 BCE 标签通常是浮点 0/1。
- 如果代码要兼容 CPU/GPU，模型、输入、标签和新建临时张量都要放在同一个 device。


## 本节任务

- 创建一个 2x3 的浮点张量 x。
- 创建一个 3x2 的浮点张量 w，并计算 x @ w。
- 打印结果的 shape、均值、最大值。
