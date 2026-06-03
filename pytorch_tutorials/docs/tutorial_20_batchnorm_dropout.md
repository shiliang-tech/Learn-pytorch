# 第 20 课：BatchNorm 与 Dropout 的 train/eval 差异

> 这份文档是本节的详细教程和速查表。先读它，再去写 `lesson` 文件；卡住时回来查，不要急着打开答案。

## 学习路径

| 步骤 | 你要做什么 | 目的 |
| --- | --- | --- |
| 1 | 读完本页的“核心概念”和“关键写法” | 先理解整体思路，知道 API 在解决什么问题 |
| 2 | 打开 `../lessons/lesson_20_batchnorm_dropout.py` 补全 TODO | 把概念变成能运行的代码 |
| 3 | 运行练习文件，观察输出 | 用 shape、loss、accuracy 判断代码是否合理 |
| 4 | 最后对照 `../answers/answer_20_batchnorm_dropout.py` | 查漏补缺，而不是直接抄答案 |

## 核心概念

- BatchNorm 在训练时使用当前 batch 统计，并更新 running_mean/running_var。
- eval 模式下 BatchNorm 使用累计统计，Dropout 则关闭随机丢弃。
- 这就是为什么验证和推理阶段一定要调用 model.eval()。

## 理解思路

写 PyTorch 代码时，先不要把注意力放在“背 API”上，而是先问四个问题：

1. 数据是什么形状？例如 `[batch, features]`、`[batch, time, features]` 或 `[N, C, H, W]`。
2. 标签是什么类型？回归通常是浮点，分类通常是整数类别编号或 0/1 浮点标签。
3. 模型输入和输出应该是什么 shape？loss 函数会严格要求它们匹配。
4. 哪些张量需要梯度，哪些只是数据或指标？不要让日志、评估和保存逻辑干扰计算图。

本节涉及模型时，优先把模型看成一个 shape 转换器：输入张量进来，经过若干层，输出 logits、预测值或重建结果。

## 关键写法速查

| 写法 | 什么时候用 |
| --- | --- |
| `nn.BatchNorm1d(num_features)`<br>`[batch, features]` | `nn.BatchNorm1d(num_features)` 常用于 MLP 的 `[batch, features]`。 |
| `nn.BatchNorm2d(num_channels)`<br>`[N, C, H, W]` | `nn.BatchNorm2d(num_channels)` 常用于 CNN 的 `[N, C, H, W]`。 |
| 概念 / 经验 | BatchNorm 训练时用当前 batch 统计，同时更新 running_mean/running_var。 |
| 概念 / 经验 | BatchNorm eval 时使用 running_mean/running_var，不再用当前 batch 统计。 |
| 概念 / 经验 | Dropout train 时随机，eval 时关闭。 |
| `model.train()`<br>`model.eval()` | 因此训练前 `model.train()`，验证/推理前 `model.eval()` 是必须习惯。 |
| 概念 / 经验 | 小 batch 下 BatchNorm 统计可能不稳定，可考虑 LayerNorm 或 GroupNorm。 |
| 概念 / 经验 | BatchNorm 一般放在 Linear/Conv 后、激活函数前或后，具体结构可按常见架构习惯。 |
| 概念 / 经验 | 比较两个输出是否一样时要先固定模式，否则 Dropout 会让结果随机。 |
| 概念 / 经验 | 不要在验证阶段忘记 eval，否则指标会飘。 |

## 练习前代码骨架

下面不是答案，只是提醒你代码应该从哪里开始。真正实现请写在 lesson 文件里。

```python
def main():
    # TODO: 演示 BatchNorm/Dropout 在 train 和 eval 下的差异。
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
| 模型输出不符合预期 | 在 forward 中逐层打印 shape，确认 Linear 或 Conv 的输入维度 |

## 本节任务

- 构建 Linear-BatchNorm-ReLU-Dropout-Linear 网络。
- 喂入同一个 batch，比较 train 和 eval 下输出差异。
- 打印 BatchNorm 的 running_mean。

## 完成标准

- 练习文件能直接运行，没有异常。
- 你能说清楚每个关键张量的 shape。
- 你能解释为什么使用这些 API，以及为什么要打印这些检查项。
- 看答案后，能关掉答案再独立复写一遍。

## 文件位置

- 练习文件：`../lessons/lesson_20_batchnorm_dropout.py`
- 答案文件：`../answers/answer_20_batchnorm_dropout.py`

建议先独立完成练习文件，再打开答案对照。能不看答案复写一遍，才算真的吃下来了。
