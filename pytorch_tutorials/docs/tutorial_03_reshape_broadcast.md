# 第 03 课：形状变换与广播机制

> 这份文档是本节的详细教程和速查表。先读它，再去写 `lesson` 文件；卡住时回来查，不要急着打开答案。

## 学习路径

| 步骤 | 你要做什么 | 目的 |
| --- | --- | --- |
| 1 | 读完本页的“核心概念”和“关键写法” | 先理解整体思路，知道 API 在解决什么问题 |
| 2 | 打开 `../lessons/lesson_03_reshape_broadcast.py` 补全 TODO | 把概念变成能运行的代码 |
| 3 | 运行练习文件，观察输出 | 用 shape、loss、accuracy 判断代码是否合理 |
| 4 | 最后对照 `../answers/answer_03_reshape_broadcast.py` | 查漏补缺，而不是直接抄答案 |

## 核心概念

- view/reshape 可以改变张量形状，但元素总数必须一致。
- unsqueeze/squeeze 用于增加或删除长度为 1 的维度。
- 广播让不同形状的张量参与运算，例如 batch 中每一行都加同一个偏置向量。

## 理解思路

写 PyTorch 代码时，先不要把注意力放在“背 API”上，而是先问四个问题：

1. 数据是什么形状？例如 `[batch, features]`、`[batch, time, features]` 或 `[N, C, H, W]`。
2. 标签是什么类型？回归通常是浮点，分类通常是整数类别编号或 0/1 浮点标签。
3. 模型输入和输出应该是什么 shape？loss 函数会严格要求它们匹配。
4. 哪些张量需要梯度，哪些只是数据或指标？不要让日志、评估和保存逻辑干扰计算图。

## 关键写法速查

| 写法 | 什么时候用 |
| --- | --- |
| `x.reshape(new_shape)`<br>`(3, 4)` | `x.reshape(new_shape)` 改变形状，元素总数必须不变，例如 12 个元素可以变成 `(3, 4)`。 |
| `x.view(...)`<br>`reshape` | `x.view(...)` 也能改形状，但要求内存连续；不确定时优先用 `reshape`。 |
| `x.flatten()`<br>`nn.Flatten()` | `x.flatten()` 展平成一维；`nn.Flatten()` 常放在 CNN 进入 Linear 前。 |
| `x.unsqueeze(dim)`<br>`[3] -> [1, 3]`<br>`[3, 1]` | `x.unsqueeze(dim)` 增加长度为 1 的维度，例如 `[3] -> [1, 3]` 或 `[3, 1]`。 |
| `x.squeeze(dim)` | `x.squeeze(dim)` 删除指定的长度为 1 的维度；不指定 dim 会删掉所有长度为 1 的维度，使用时要小心 batch 维。 |
| `x.permute(0, 2, 3, 1)` | `x.permute(0, 2, 3, 1)` 按任意顺序重排维度；图像数据经常在 NHWC 和 NCHW 间转换。 |
| `x.transpose(dim0, dim1)`<br>`x.T` | `x.transpose(dim0, dim1)` 交换两个维度；二维矩阵也可以用 `x.T`。 |
| 概念 / 经验 | 广播规则从最后一维开始对齐；两个维度相等，或其中一个为 1，才可以广播。 |
| `bias`<br>`[features]`<br>`[batch, features]` | `bias` 形状是 `[features]` 时，可以自动加到 `[batch, features]` 的每一行。 |
| `keepdim=True`<br>`(x - mean) / std` | `keepdim=True` 可以保留聚合后的维度，常用于 `(x - mean) / std` 这种标准化写法。 |
| 概念 / 经验 | 如果广播报错，先打印参与运算的两个 shape，从最后一维往前对齐检查。 |

## 练习前代码骨架

下面不是答案，只是提醒你代码应该从哪里开始。真正实现请写在 lesson 文件里。

```python
def main():
    # TODO: 练习 reshape、广播和 unsqueeze。
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

- 把 torch.arange(12) 变成 3x4 矩阵。
- 创建长度为 4 的 bias，并加到矩阵每一行。
- 用 unsqueeze 把结果变成 1x3x4。

## 完成标准

- 练习文件能直接运行，没有异常。
- 你能说清楚每个关键张量的 shape。
- 你能解释为什么使用这些 API，以及为什么要打印这些检查项。
- 看答案后，能关掉答案再独立复写一遍。

## 文件位置

- 练习文件：`../lessons/lesson_03_reshape_broadcast.py`
- 答案文件：`../answers/answer_03_reshape_broadcast.py`

建议先独立完成练习文件，再打开答案对照。能不看答案复写一遍，才算真的吃下来了。
