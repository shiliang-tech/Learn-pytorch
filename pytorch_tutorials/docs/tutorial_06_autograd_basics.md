# 第 06 课：自动求导 autograd

> 这份文档是本节的详细教程和速查表。先读它，再去写 `lesson` 文件；卡住时回来查，不要急着打开答案。

## 学习路径

| 步骤 | 你要做什么 | 目的 |
| --- | --- | --- |
| 1 | 读完本页的“核心概念”和“关键写法” | 先理解整体思路，知道 API 在解决什么问题 |
| 2 | 打开 `../lessons/lesson_06_autograd_basics.py` 补全 TODO | 把概念变成能运行的代码 |
| 3 | 运行练习文件，观察输出 | 用 shape、loss、accuracy 判断代码是否合理 |
| 4 | 最后对照 `../answers/answer_06_autograd_basics.py` | 查漏补缺，而不是直接抄答案 |

## 核心概念

- requires_grad=True 会让 PyTorch 记录张量参与的计算图。
- loss.backward() 会从标量 loss 反向传播，计算叶子张量的 grad。
- 每次反向传播前通常要清空旧梯度，因为 PyTorch 默认会累加梯度。

## 理解思路

写 PyTorch 代码时，先不要把注意力放在“背 API”上，而是先问四个问题：

1. 数据是什么形状？例如 `[batch, features]`、`[batch, time, features]` 或 `[N, C, H, W]`。
2. 标签是什么类型？回归通常是浮点，分类通常是整数类别编号或 0/1 浮点标签。
3. 模型输入和输出应该是什么 shape？loss 函数会严格要求它们匹配。
4. 哪些张量需要梯度，哪些只是数据或指标？不要让日志、评估和保存逻辑干扰计算图。

本节涉及训练时，请把训练循环固定成一个习惯：前向计算 -> 计算损失 -> 清空梯度 -> 反向传播 -> 更新参数 -> 记录指标。

## 关键写法速查

| 写法 | 什么时候用 |
| --- | --- |
| `x = torch.tensor(1.0, requires_grad=True)` | `x = torch.tensor(1.0, requires_grad=True)` 会让 PyTorch 记录 x 参与的计算。 |
| `requires_grad=True` | 只有浮点或复数张量能 `requires_grad=True`，整数标签不能求梯度。 |
| `loss.backward()`<br>`gradient` | `loss.backward()` 要求 loss 通常是标量；非标量需要传入 `gradient` 参数。 |
| `.grad`<br>`w.grad` | 叶子张量的梯度保存在 `.grad`，例如 `w.grad`。 |
| 概念 / 经验 | PyTorch 默认梯度累加，所以训练循环里必须清空旧梯度。 |
| `with torch.no_grad():` | `with torch.no_grad():` 里面的计算不会被 autograd 记录，常用于手动更新参数和推理。 |
| `x.detach()` | `x.detach()` 得到一个不再连接当前计算图的新张量，常用于停止梯度传播。 |
| `loss.item()`<br>`.item()` | `loss.item()` 只用于日志，不要用 `.item()` 后的 Python 数字继续参与反向传播。 |
| `element 0 of tensors does not require grad` | 如果看到 `element 0 of tensors does not require grad`，说明 loss 和可训练参数之间的计算图断了。 |
| `_` | 如果遇到原地操作报错，检查是否对需要梯度的中间张量用了带 `_` 的方法或切片赋值。 |

## 练习前代码骨架

下面不是答案，只是提醒你代码应该从哪里开始。真正实现请写在 lesson 文件里。

```python
def main():
    # TODO: 创建 w，计算 loss，然后 backward。
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
| 训练越来越差 | 先降低学习率，再检查标签、loss 函数和模型输出是否匹配 |

## 本节任务

- 创建一个可求导的标量 w。
- 令 loss=(w-3)^2，调用 backward。
- 打印 w.grad，并解释为什么梯度等于 2*(w-3)。

## 完成标准

- 练习文件能直接运行，没有异常。
- 你能说清楚每个关键张量的 shape。
- 你能解释为什么使用这个 loss、这个 dtype、这个输出维度。
- 训练任务的 loss 或指标有合理变化，而不是完全随机。
- 看答案后，能关掉答案再独立复写一遍。

## 文件位置

- 练习文件：`../lessons/lesson_06_autograd_basics.py`
- 答案文件：`../answers/answer_06_autograd_basics.py`

建议先独立完成练习文件，再打开答案对照。能不看答案复写一遍，才算真的吃下来了。
