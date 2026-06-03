# 第 15 课：保存与加载模型参数

> 这份文档是本节的详细教程和速查表。先读它，再去写 `lesson` 文件；卡住时回来查，不要急着打开答案。

## 学习路径

| 步骤 | 你要做什么 | 目的 |
| --- | --- | --- |
| 1 | 读完本页的“核心概念”和“关键写法” | 先理解整体思路，知道 API 在解决什么问题 |
| 2 | 打开 `../lessons/lesson_15_save_load.py` 补全 TODO | 把概念变成能运行的代码 |
| 3 | 运行练习文件，观察输出 | 用 shape、loss、accuracy 判断代码是否合理 |
| 4 | 最后对照 `../answers/answer_15_save_load.py` | 查漏补缺，而不是直接抄答案 |

## 核心概念

- 推荐保存 model.state_dict()，它只包含参数和 buffer，结构清晰、可迁移。
- 加载时需要先创建相同结构的模型，再 load_state_dict。
- 推理前记得 model.eval()，避免 Dropout 和 BatchNorm 处在训练行为。

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
| `torch.save(model.state_dict(), path)` | `torch.save(model.state_dict(), path)` 是推荐保存方式，只保存参数和 buffer。 |
| `model = MyModel(); model.load_state_dict(torch.load(path))` | 加载时先创建同结构模型：`model = MyModel(); model.load_state_dict(torch.load(path))`。 |
| `map_location='cpu'` | `map_location='cpu'` 可以把 GPU 保存的参数加载到 CPU。 |
| `model.eval()` | 推理前调用 `model.eval()`，保证 Dropout/BatchNorm 行为正确。 |
| 概念 / 经验 | 如果还要恢复训练，应同时保存 optimizer state、epoch、best metric。 |
| `{'model': model.state_dict(), 'optimizer': opt.state_dict(), 'epoch': epoch}` | checkpoint 常见结构：`{'model': model.state_dict(), 'optimizer': opt.state_dict(), 'epoch': epoch}`。 |
| `strict=False` | `strict=False` 可以在模型结构略有变化时加载部分参数，但要仔细检查 missing/unexpected keys。 |
| `model` | 只保存整个 `model` 对象不够稳，代码结构变化后更容易加载失败。 |
| `pathlib.Path` | 保存路径用 `pathlib.Path` 更跨平台，也方便创建目录。 |
| 概念 / 经验 | 加载后可用同一输入比较两个模型输出，确认参数恢复正确。 |

## 练习前代码骨架

下面不是答案，只是提醒你代码应该从哪里开始。真正实现请写在 lesson 文件里。

```python
def main():
    # TODO: 训练、保存、加载，并比较输出。
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

- 训练一个小模型。
- 把 state_dict 保存到 pytorch_tutorials/tmp_linear.pt。
- 创建新模型加载参数，验证两者输出一致。

## 完成标准

- 练习文件能直接运行，没有异常。
- 你能说清楚每个关键张量的 shape。
- 你能解释为什么使用这个 loss、这个 dtype、这个输出维度。
- 训练任务的 loss 或指标有合理变化，而不是完全随机。
- 看答案后，能关掉答案再独立复写一遍。

## 文件位置

- 练习文件：`../lessons/lesson_15_save_load.py`
- 答案文件：`../answers/answer_15_save_load.py`

建议先独立完成练习文件，再打开答案对照。能不看答案复写一遍，才算真的吃下来了。
