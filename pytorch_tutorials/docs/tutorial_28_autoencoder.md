# 第 28 课：AutoEncoder：压缩与重建

> 这份文档是本节的详细教程和速查表。先读它，再去写 `lesson` 文件；卡住时回来查，不要急着打开答案。

## 学习路径

| 步骤 | 你要做什么 | 目的 |
| --- | --- | --- |
| 1 | 读完本页的“核心概念”和“关键写法” | 先理解整体思路，知道 API 在解决什么问题 |
| 2 | 打开 `../lessons/lesson_28_autoencoder.py` 补全 TODO | 把概念变成能运行的代码 |
| 3 | 运行练习文件，观察输出 | 用 shape、loss、accuracy 判断代码是否合理 |
| 4 | 最后对照 `../answers/answer_28_autoencoder.py` | 查漏补缺，而不是直接抄答案 |

## 核心概念

- 自编码器由 encoder 和 decoder 组成，目标是重建输入。
- 瓶颈层维度较小时，模型被迫学习压缩表示。
- 重建任务通常使用 MSELoss 或 BCE 类损失，取决于数据范围。

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
| `encoder`<br>`decoder` | AutoEncoder 通常写成 `encoder` 和 `decoder` 两个子网络。 |
| `z = encoder(x)`<br>`recon = decoder(z)` | `z = encoder(x)` 是压缩表示，`recon = decoder(z)` 是重建结果。 |
| 概念 / 经验 | 瓶颈层 latent 维度越小，压缩压力越大，重建可能更难。 |
| `nn.MSELoss()` | 连续值重建常用 `nn.MSELoss()`；0-1 图像也可考虑 BCE 类损失。 |
| 概念 / 经验 | decoder 最后一层是否加激活取决于数据范围，例如 0-1 可加 Sigmoid。 |
| 概念 / 经验 | 自编码器不需要标签，是自监督/无监督风格的训练。 |
| 概念 / 经验 | 训练循环和普通回归类似，只是 label 就是输入 x 自己。 |
| 概念 / 经验 | 可以用 latent 做可视化、异常检测或下游特征。 |
| 概念 / 经验 | 如果 recon 很差，尝试加大 latent 维度或模型容量。 |
| 概念 / 经验 | 如果只会复制输入，说明压缩或正则约束可能不够。 |

## 练习前代码骨架

下面不是答案，只是提醒你代码应该从哪里开始。真正实现请写在 lesson 文件里。

```python
def main():
    # TODO: 实现并训练一个小 AutoEncoder。
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

- 生成二维圆环数据。
- 用 encoder 把 2 维压到 1 维，再 decoder 重建到 2 维。
- 训练并打印重建误差。

## 完成标准

- 练习文件能直接运行，没有异常。
- 你能说清楚每个关键张量的 shape。
- 你能解释为什么使用这个 loss、这个 dtype、这个输出维度。
- 训练任务的 loss 或指标有合理变化，而不是完全随机。
- 看答案后，能关掉答案再独立复写一遍。

## 文件位置

- 练习文件：`../lessons/lesson_28_autoencoder.py`
- 答案文件：`../answers/answer_28_autoencoder.py`

建议先独立完成练习文件，再打开答案对照。能不看答案复写一遍，才算真的吃下来了。
