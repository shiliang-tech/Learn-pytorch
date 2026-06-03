# 第 17 课：自定义 Dataset

> 这份文档是本节的详细教程和速查表。先读它，再去写 `lesson` 文件；卡住时回来查，不要急着打开答案。

## 学习路径

| 步骤 | 你要做什么 | 目的 |
| --- | --- | --- |
| 1 | 读完本页的“核心概念”和“关键写法” | 先理解整体思路，知道 API 在解决什么问题 |
| 2 | 打开 `../lessons/lesson_17_custom_dataset.py` 补全 TODO | 把概念变成能运行的代码 |
| 3 | 运行练习文件，观察输出 | 用 shape、loss、accuracy 判断代码是否合理 |
| 4 | 最后对照 `../answers/answer_17_custom_dataset.py` | 查漏补缺，而不是直接抄答案 |

## 核心概念

- 自定义 Dataset 至少实现 __len__ 和 __getitem__。
- __getitem__ 返回一个样本，可以是张量、标签、字典等结构。
- 把数据生成或预处理封装到 Dataset 后，训练循环会更稳定清晰。

## 理解思路

写 PyTorch 代码时，先不要把注意力放在“背 API”上，而是先问四个问题：

1. 数据是什么形状？例如 `[batch, features]`、`[batch, time, features]` 或 `[N, C, H, W]`。
2. 标签是什么类型？回归通常是浮点，分类通常是整数类别编号或 0/1 浮点标签。
3. 模型输入和输出应该是什么 shape？loss 函数会严格要求它们匹配。
4. 哪些张量需要梯度，哪些只是数据或指标？不要让日志、评估和保存逻辑干扰计算图。

本节涉及模型时，优先把模型看成一个 shape 转换器：输入张量进来，经过若干层，输出 logits、预测值或重建结果。

本节涉及训练时，请把训练循环固定成一个习惯：前向计算 -> 计算损失 -> 清空梯度 -> 反向传播 -> 更新参数 -> 记录指标。

## 关键写法速查

| 写法 | 什么时候用 |
| --- | --- |
| `torch.utils.data.Dataset` | 自定义 Dataset 继承 `torch.utils.data.Dataset`。 |
| `__len__(self)` | `__len__(self)` 返回样本数，DataLoader 依赖它判断一轮有多少数据。 |
| `__getitem__(self, idx)`<br>`(x, y)` | `__getitem__(self, idx)` 返回第 idx 个样本，可以是 `(x, y)`、字典或更多字段。 |
| `__init__`<br>`__getitem__` | 数据可以在 `__init__` 中提前准备，也可以在 `__getitem__` 中按需读取。 |
| 概念 / 经验 | 返回的数值最好转换成张量，避免训练循环里到处做类型转换。 |
| `DataLoader`<br>`collate_fn` | `DataLoader` 会把多个样本自动 collate 成 batch；形状不一致时需要自定义 `collate_fn`。 |
| 概念 / 经验 | 回归标签通常是 float，分类标签通常是 long。 |
| `__getitem__` | 数据增强常放在 Dataset 的 `__getitem__` 中。 |
| 概念 / 经验 | Dataset 不负责训练，只负责稳定地提供样本。 |
| `loader = DataLoader(dataset, batch_size=4)` | 先用小数据和 `loader = DataLoader(dataset, batch_size=4)` 打印一个 batch，确认 shape。 |

## 练习前代码骨架

下面不是答案，只是提醒你代码应该从哪里开始。真正实现请写在 lesson 文件里。

```python
class SineDataset:
    # TODO: 继承 Dataset，实现 __len__ 和 __getitem__。
    pass


def main():
    # TODO: 使用 DataLoader 训练回归模型。
    pass


if __name__ == "__main__":
    main()
```

训练类题目可以把下面这段顺序刻进手里：

```python
pred_or_logits = model(x)
loss = loss_fn(pred_or_logits, y)
optimizer.zero_grad()
loss.backward()
optimizer.step()
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
| 模型输出不符合预期 | 在 forward 中逐层打印 shape，确认 Linear 或 Conv 的输入维度 |
| 训练越来越差 | 先降低学习率，再检查标签、loss 函数和模型输出是否匹配 |

## 本节任务

- 写一个 SineDataset，输入 x，标签 y=sin(x)。
- 用 DataLoader 批量读取。
- 训练一个小 MLP 做回归。

## 完成标准

- 练习文件能直接运行，没有异常。
- 你能说清楚每个关键张量的 shape。
- 你能解释为什么使用这个 loss、这个 dtype、这个输出维度。
- 训练任务的 loss 或指标有合理变化，而不是完全随机。
- 看答案后，能关掉答案再独立复写一遍。

## 文件位置

- 练习文件：`../lessons/lesson_17_custom_dataset.py`
- 答案文件：`../answers/answer_17_custom_dataset.py`

建议先独立完成练习文件，再打开答案对照。能不看答案复写一遍，才算真的吃下来了。
