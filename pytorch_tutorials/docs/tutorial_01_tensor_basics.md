# 第 01 课：Tensor 入门：创建、运算、矩阵乘法

> 这份文档是本节的详细教程和速查表。先读它，再去写 `lesson` 文件；卡住时回来查，不要急着打开答案。

## 学习路径

| 步骤 | 你要做什么 | 目的 |
| --- | --- | --- |
| 1 | 读完本页的“核心概念”和“关键写法” | 先理解整体思路，知道 API 在解决什么问题 |
| 2 | 打开 `../lessons/lesson_01_tensor_basics.py` 补全 TODO | 把概念变成能运行的代码 |
| 3 | 运行练习文件，观察输出 | 用 shape、loss、accuracy 判断代码是否合理 |
| 4 | 最后对照 `../answers/answer_01_tensor_basics.py` | 查漏补缺，而不是直接抄答案 |

## 核心概念

- Tensor 是 PyTorch 的核心数据结构，可以把它理解成支持 GPU 和自动求导的多维数组。
- 标量是 0 维，向量是 1 维，矩阵是 2 维，更高维常用来表达 batch、通道、图像高宽、时间步等。
- 深度学习里的大多数计算最终都会落到张量运算：加减乘除、矩阵乘法、聚合统计。

## 理解思路

写 PyTorch 代码时，先不要把注意力放在“背 API”上，而是先问四个问题：

1. 数据是什么形状？例如 `[batch, features]`、`[batch, time, features]` 或 `[N, C, H, W]`。
2. 标签是什么类型？回归通常是浮点，分类通常是整数类别编号或 0/1 浮点标签。
3. 模型输入和输出应该是什么 shape？loss 函数会严格要求它们匹配。
4. 哪些张量需要梯度，哪些只是数据或指标？不要让日志、评估和保存逻辑干扰计算图。

## 关键写法速查

| 写法 | 什么时候用 |
| --- | --- |
| `import torch` | `import torch` 是所有 PyTorch 脚本的起点，通常先放在文件顶部。 |
| `torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)` | `torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)` 从 Python 列表创建张量，并指定浮点类型。 |
| `torch.zeros(shape)`<br>`torch.ones(shape)`<br>`torch.full(shape, value)` | `torch.zeros(shape)`、`torch.ones(shape)`、`torch.full(shape, value)` 分别创建全 0、全 1、指定值张量。 |
| `torch.arange(start, end, step)`<br>`torch.linspace(start, end, steps)` | `torch.arange(start, end, step)` 创建等差整数序列；`torch.linspace(start, end, steps)` 创建等距浮点序列。 |
| `torch.randn(shape)`<br>`torch.rand(shape)` | `torch.randn(shape)` 创建标准正态随机张量；`torch.rand(shape)` 创建 0 到 1 的均匀随机张量。 |
| `x + y`<br>`x - y`<br>`x * y`<br>... | `x + y`、`x - y`、`x * y`、`x / y` 是逐元素运算，要求形状相同或可以广播。 |
| `x @ w`<br>`torch.matmul(x, w)`<br>`x.shape[1] == w.shape[0]` | `x @ w` 或 `torch.matmul(x, w)` 是矩阵乘法；二维时要求 `x.shape[1] == w.shape[0]`。 |
| `x.T`<br>`transpose(dim0, dim1)`<br>`permute(...)` | `x.T` 可以转置二维矩阵；高维张量更常用 `transpose(dim0, dim1)` 或 `permute(...)`。 |
| `x.shape`<br>`x.ndim`<br>`x.numel()` | `x.shape`、`x.ndim`、`x.numel()` 分别查看形状、维度数、元素总数。 |
| `x.mean()`<br>`x.sum()`<br>`x.max()`<br>... | `x.mean()`、`x.sum()`、`x.max()`、`x.min()`、`x.std()` 是最常见统计操作。 |
| `x.item()` | `x.item()` 把只含一个元素的张量转成 Python 数字，常用于打印 loss。 |
| `print(x, x.shape, x.dtype, x.device)` | `print(x, x.shape, x.dtype, x.device)` 是排查 PyTorch 代码最朴素也最有效的方法。 |

## 练习前代码骨架

下面不是答案，只是提醒你代码应该从哪里开始。真正实现请写在 lesson 文件里。

```python
def main():
    # TODO: 创建 x 和 w，然后完成矩阵乘法。
    pass


if __name__ == "__main__":
    main()
```

## 写题步骤

1. 先把本节会用到的 import、张量 shape、loss 或模型层写出来。
2. 每完成一步就打印一次关键变量，特别是 `shape`、`dtype`、`device`。
3. 如果是训练任务，先让代码跑通，再观察 loss 是否下降、accuracy 是否合理。
4. 不确定 API 怎么用时，优先回到本页的关键写法，不要急着看答案。

## 常见排错表

| 现象 | 优先检查 |
| --- | --- |
| shape 报错 | 打印参与运算的所有张量 shape，按维度逐个对齐 |
| dtype 报错 | 分类标签通常要 `torch.long`，回归值和模型输入通常要浮点 |
| device 报错 | 模型、输入、标签必须在同一个 device |
| loss 不下降 | 检查是否执行了 `zero_grad()`、`backward()`、`step()`，学习率是否过大或过小 |
| 指标奇怪 | 确认标签含义、输出 shape、阈值或 `argmax(dim=...)` 是否写对 |

## 本节任务

- 创建一个 2x3 的浮点张量 x。
- 创建一个 3x2 的浮点张量 w，并计算 x @ w。
- 打印结果的 shape、均值、最大值。

## 完成标准

- 练习文件能直接运行，没有异常。
- 你能说清楚每个关键张量的 shape。
- 你能解释为什么使用这些 API，以及为什么要打印这些检查项。
- 看答案后，能关掉答案再独立复写一遍。

## 文件位置

- 练习文件：`../lessons/lesson_01_tensor_basics.py`
- 答案文件：`../answers/answer_01_tensor_basics.py`

建议先独立完成练习文件，再打开答案对照。能不看答案复写一遍，才算真的吃下来了。
