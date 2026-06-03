# 第 32 课：端到端小项目：从数据到保存模型

> 这份文档是本节的详细教程和速查表。先读它，再去写 `lesson` 文件；卡住时回来查，不要急着打开答案。

## 学习路径

| 步骤 | 你要做什么 | 目的 |
| --- | --- | --- |
| 1 | 读完本页的“核心概念”和“关键写法” | 先理解整体思路，知道 API 在解决什么问题 |
| 2 | 打开 `../lessons/lesson_32_end_to_end_project.py` 补全 TODO | 把概念变成能运行的代码 |
| 3 | 运行练习文件，观察输出 | 用 shape、loss、accuracy 判断代码是否合理 |
| 4 | 最后对照 `../answers/answer_32_end_to_end_project.py` | 查漏补缺，而不是直接抄答案 |

## 核心概念

- 完整项目通常包含：数据准备、模型、训练循环、验证指标、保存最好模型。
- 把 train_one_epoch 和 evaluate 拆成函数，会让代码更容易维护。
- 最终你应该能独立搭起一个小型监督学习项目骨架。

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
| 概念 / 经验 | 端到端脚本通常包含：配置、数据、模型、loss、optimizer、train、evaluate、save。 |
| `train_one_epoch` | `train_one_epoch` 负责 train 模式、前向、loss、反向、更新，并返回训练日志。 |
| `evaluate` | `evaluate` 负责 eval 模式、no_grad、计算 loss 和指标，不更新参数。 |
| 概念 / 经验 | 主循环里每个 epoch 调一次训练和验证。 |
| `best_val_acc`<br>`best_val_loss` | 用 `best_val_acc` 或 `best_val_loss` 判断是否保存模型。 |
| `torch.save(model.state_dict(), path)` | `torch.save(model.state_dict(), path)` 保存验证集最好的参数。 |
| 概念 / 经验 | 实验要固定随机种子，便于比较不同改动。 |
| 概念 / 经验 | 每次改模型/学习率/batch size 后，都用同一验证集比较。 |
| 概念 / 经验 | 日志至少打印 epoch、train_loss、val_loss、val_acc。 |
| 概念 / 经验 | 项目变大后，可以把 dataset、model、train utils 拆成不同文件，但先把单文件流程写熟。 |

## 练习前代码骨架

下面不是答案，只是提醒你代码应该从哪里开始。真正实现请写在 lesson 文件里。

```python
def train_one_epoch(model, x, y, optimizer, loss_fn):
    # TODO: 完成一个训练 epoch。
    pass


def evaluate(model, x, y, loss_fn):
    # TODO: 返回 loss 和 accuracy。
    pass


def main():
    # TODO: 串起数据、模型、训练、验证、保存。
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

- 创建一个非线性二分类数据集。
- 写 train_one_epoch 与 evaluate。
- 保存验证准确率最高的模型参数。

## 完成标准

- 练习文件能直接运行，没有异常。
- 你能说清楚每个关键张量的 shape。
- 你能解释为什么使用这个 loss、这个 dtype、这个输出维度。
- 训练任务的 loss 或指标有合理变化，而不是完全随机。
- 看答案后，能关掉答案再独立复写一遍。

## 文件位置

- 练习文件：`../lessons/lesson_32_end_to_end_project.py`
- 答案文件：`../answers/answer_32_end_to_end_project.py`

建议先独立完成练习文件，再打开答案对照。能不看答案复写一遍，才算真的吃下来了。
