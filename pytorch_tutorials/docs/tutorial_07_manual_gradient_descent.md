# 第 07 课：手写梯度下降：拟合 y=2x+1

> 这份文档是本节的详细教程和速查表。先读它，再去写 `lesson` 文件；卡住时回来查，不要急着打开答案。

## 学习路径

| 步骤 | 你要做什么 | 目的 |
| --- | --- | --- |
| 1 | 读完本页的“核心概念”和“关键写法” | 先理解整体思路，知道 API 在解决什么问题 |
| 2 | 打开 `../lessons/lesson_07_manual_gradient_descent.py` 补全 TODO | 把概念变成能运行的代码 |
| 3 | 运行练习文件，观察输出 | 用 shape、loss、accuracy 判断代码是否合理 |
| 4 | 最后对照 `../answers/answer_07_manual_gradient_descent.py` | 查漏补缺，而不是直接抄答案 |

## 核心概念

- 训练的基本循环是：前向计算、计算损失、反向传播、更新参数、清空梯度。
- 梯度下降用参数减去 learning_rate * gradient，让 loss 往下降方向移动。
- 先手写一次更新过程，可以更清楚 nn.Module 和 optimizer 后面在替你做什么。

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
| `pred -> loss -> zero_grad -> backward -> step`<br>`step` | 最小训练循环是 `pred -> loss -> zero_grad -> backward -> step`，手写版则把 `step` 换成手动减梯度。 |
| `requires_grad=True`<br>`w = torch.randn(1, 1, requires_grad=True)` | 手动参数需要 `requires_grad=True`，例如 `w = torch.randn(1, 1, requires_grad=True)`。 |
| `pred = x @ w + b`<br>`x`<br>`[batch, features]` | 线性回归预测常写成 `pred = x @ w + b`，其中 `x` 是 `[batch, features]`。 |
| `((pred - y) ** 2).mean()` | MSE 可手写为 `((pred - y) ** 2).mean()`。 |
| `with torch.no_grad(): w -= lr * w.grad` | 参数更新要放进 `with torch.no_grad(): w -= lr * w.grad`，否则更新本身也会被记录进计算图。 |
| `w.grad.zero_()`<br>`b.grad.zero_()` | 更新后要 `w.grad.zero_()` 和 `b.grad.zero_()`，否则下一轮梯度会叠加。 |
| `lr` | `lr` 太大 loss 可能变成 NaN 或震荡；太小则收敛很慢。 |
| 概念 / 经验 | 训练时可以每隔若干 epoch 打印 loss，观察是否下降。 |
| 概念 / 经验 | 真实项目更常用 optimizer，但手写一次能帮你理解 optimizer 到底替你做了什么。 |

## 练习前代码骨架

下面不是答案，只是提醒你代码应该从哪里开始。真正实现请写在 lesson 文件里。

```python
def main():
    # TODO: 手写一个 100 轮的梯度下降线性回归。
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

- 生成 x 与 y=2x+1 的训练数据。
- 用两个 requires_grad 参数 w、b 手动训练。
- 每轮用 no_grad 更新参数，并把梯度清零。

## 完成标准

- 练习文件能直接运行，没有异常。
- 你能说清楚每个关键张量的 shape。
- 你能解释为什么使用这个 loss、这个 dtype、这个输出维度。
- 训练任务的 loss 或指标有合理变化，而不是完全随机。
- 看答案后，能关掉答案再独立复写一遍。

## 文件位置

- 练习文件：`../lessons/lesson_07_manual_gradient_descent.py`
- 答案文件：`../answers/answer_07_manual_gradient_descent.py`

建议先独立完成练习文件，再打开答案对照。能不看答案复写一遍，才算真的吃下来了。
