# 第 05 课：聚合统计、范数与标准化

> 这份文档是本节的详细教程和速查表。先读它，再去写 `lesson` 文件；卡住时回来查，不要急着打开答案。

## 学习路径

| 步骤 | 你要做什么 | 目的 |
| --- | --- | --- |
| 1 | 读完本页的“核心概念”和“关键写法” | 先理解整体思路，知道 API 在解决什么问题 |
| 2 | 打开 `../lessons/lesson_05_reductions_norms.py` 补全 TODO | 把概念变成能运行的代码 |
| 3 | 运行练习文件，观察输出 | 用 shape、loss、accuracy 判断代码是否合理 |
| 4 | 最后对照 `../answers/answer_05_reductions_norms.py` | 查漏补缺，而不是直接抄答案 |

## 核心概念

- mean、sum、max、argmax 这类 reduction 会沿某些维度汇总信息。
- 范数常用来度量向量大小，训练中也会用于正则化、梯度裁剪、相似度计算。
- 标准化让数据均值接近 0、标准差接近 1，通常能让优化更稳定。

## 理解思路

写 PyTorch 代码时，先不要把注意力放在“背 API”上，而是先问四个问题：

1. 数据是什么形状？例如 `[batch, features]`、`[batch, time, features]` 或 `[N, C, H, W]`。
2. 标签是什么类型？回归通常是浮点，分类通常是整数类别编号或 0/1 浮点标签。
3. 模型输入和输出应该是什么 shape？loss 函数会严格要求它们匹配。
4. 哪些张量需要梯度，哪些只是数据或指标？不要让日志、评估和保存逻辑干扰计算图。

## 关键写法速查

| 写法 | 什么时候用 |
| --- | --- |
| `x.mean()`<br>`x.mean(dim=0)` | `x.mean()` 对所有元素求均值；`x.mean(dim=0)` 沿第 0 维聚合，常得到每列均值。 |
| `x.sum(dim=1)` | `x.sum(dim=1)` 对每一行求和，常用于统计每个样本的特征总量。 |
| `x.max(dim=1)`<br>`(values, indices)`<br>`x.argmax(dim=1)` | `x.max(dim=1)` 返回 `(values, indices)`，多分类里更常用 `x.argmax(dim=1)`。 |
| `keepdim=True` | `keepdim=True` 保留被聚合维度，方便后续和原张量广播运算。 |
| `x.std(dim=0)`<br>`(x - mean) / (std + 1e-8)` | `x.std(dim=0)` 计算标准差；标准化常写成 `(x - mean) / (std + 1e-8)`。 |
| `torch.linalg.vector_norm(x, dim=1)` | `torch.linalg.vector_norm(x, dim=1)` 计算每行向量范数。 |
| `torch.clamp(x, min=0, max=1)` | `torch.clamp(x, min=0, max=1)` 可以把数值限制在区间内。 |
| `torch.nan_to_num(x)` | `torch.nan_to_num(x)` 可以把 NaN/Inf 替换成有限数，在排查坏数据时有用。 |
| `dim` | `dim` 是 PyTorch 最重要的参数之一；不知道该写几，就先把 shape 按维度编号写在纸上。 |
| `.item()` | 统计值参与训练时仍然是张量；只打印时再 `.item()`，不要过早把它变成 Python 数字。 |

## 练习前代码骨架

下面不是答案，只是提醒你代码应该从哪里开始。真正实现请写在 lesson 文件里。

```python
def main():
    # TODO: 完成按列标准化。
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

- 创建一个 5x3 的随机数据矩阵。
- 计算每一列的均值和标准差。
- 把每一列标准化，并验证标准化后的均值接近 0。

## 完成标准

- 练习文件能直接运行，没有异常。
- 你能说清楚每个关键张量的 shape。
- 你能解释为什么使用这些 API，以及为什么要打印这些检查项。
- 看答案后，能关掉答案再独立复写一遍。

## 文件位置

- 练习文件：`../lessons/lesson_05_reductions_norms.py`
- 答案文件：`../answers/answer_05_reductions_norms.py`

建议先独立完成练习文件，再打开答案对照。能不看答案复写一遍，才算真的吃下来了。
