# 第 09 课：TensorDataset 与 DataLoader

> 这份文档是本节的详细教程和速查表。先读它，再去写 `lesson` 文件；卡住时回来查，不要急着打开答案。

## 学习路径

| 步骤 | 你要做什么 | 目的 |
| --- | --- | --- |
| 1 | 读完本页的“核心概念”和“关键写法” | 先理解整体思路，知道 API 在解决什么问题 |
| 2 | 打开 `../lessons/lesson_09_dataloader_basics.py` 补全 TODO | 把概念变成能运行的代码 |
| 3 | 运行练习文件，观察输出 | 用 shape、loss、accuracy 判断代码是否合理 |
| 4 | 最后对照 `../answers/answer_09_dataloader_basics.py` | 查漏补缺，而不是直接抄答案 |

## 核心概念

- Dataset 定义如何取一个样本，DataLoader 负责批量、打乱、多进程读取。
- batch 训练比一次喂全部数据更常见，也更接近真实项目。
- shuffle=True 常用于训练集，验证集和测试集通常不需要 shuffle。

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
| `TensorDataset(x, y)` | `TensorDataset(x, y)` 把多个张量按第 0 维对齐，组成样本集。 |
| `DataLoader(dataset, batch_size=32, shuffle=True)` | `DataLoader(dataset, batch_size=32, shuffle=True)` 自动切 batch 并打乱训练数据。 |
| `for xb, yb in loader:` | `for xb, yb in loader:` 是 mini-batch 训练的基本写法。 |
| `len(dataset)` | Dataset 里的第 0 维通常是样本数，`len(dataset)` 返回样本数量。 |
| `shuffle=True` | `shuffle=True` 常用于训练集；验证/测试集一般设为 False。 |
| `zero_grad -> backward -> step` | 每个 batch 内仍要做完整训练三连：`zero_grad -> backward -> step`。 |
| `xb`<br>`yb` | 如果使用 GPU，要在循环里把每个 `xb`、`yb` 移到 device。 |
| `drop_last=True` | `drop_last=True` 可以丢弃最后一个不满 batch 的小批次，BatchNorm 或固定形状场景可能有用。 |
| `if __name__ == '__main__':` | Windows 上自定义多进程 DataLoader 时要注意 `if __name__ == '__main__':` 保护。 |
| `num_workers=0` | 刚入门时 `num_workers=0` 最稳，等流程跑通再考虑加速数据读取。 |

## 练习前代码骨架

下面不是答案，只是提醒你代码应该从哪里开始。真正实现请写在 lesson 文件里。

```python
def main():
    # TODO: 使用 TensorDataset 和 DataLoader 训练模型。
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

- 用 TensorDataset 包装 x 和 y。
- 用 DataLoader 每次取 16 条样本。
- 写一个 mini-batch 训练循环拟合线性回归。

## 完成标准

- 练习文件能直接运行，没有异常。
- 你能说清楚每个关键张量的 shape。
- 你能解释为什么使用这个 loss、这个 dtype、这个输出维度。
- 训练任务的 loss 或指标有合理变化，而不是完全随机。
- 看答案后，能关掉答案再独立复写一遍。

## 文件位置

- 练习文件：`../lessons/lesson_09_dataloader_basics.py`
- 答案文件：`../answers/answer_09_dataloader_basics.py`

建议先独立完成练习文件，再打开答案对照。能不看答案复写一遍，才算真的吃下来了。
