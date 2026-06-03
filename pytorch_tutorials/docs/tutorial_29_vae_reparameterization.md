# 第 29 课：VAE：均值、方差与重参数化

> 这份文档是本节的详细教程和速查表。先读它，再去写 `lesson` 文件；卡住时回来查，不要急着打开答案。

## 学习路径

| 步骤 | 你要做什么 | 目的 |
| --- | --- | --- |
| 1 | 读完本页的“核心概念”和“关键写法” | 先理解整体思路，知道 API 在解决什么问题 |
| 2 | 打开 `../lessons/lesson_29_vae_reparameterization.py` 补全 TODO | 把概念变成能运行的代码 |
| 3 | 运行练习文件，观察输出 | 用 shape、loss、accuracy 判断代码是否合理 |
| 4 | 最后对照 `../answers/answer_29_vae_reparameterization.py` | 查漏补缺，而不是直接抄答案 |

## 核心概念

- VAE 的 encoder 输出潜变量分布参数 mu 和 logvar，而不是单个确定向量。
- 重参数化 z=mu+std*eps 让采样过程仍能反向传播到 encoder。
- VAE loss 通常由重建误差和 KL 散度组成。

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
| `mu`<br>`logvar` | VAE encoder 输出 `mu` 和 `logvar`，表示潜变量高斯分布参数。 |
| `std = torch.exp(0.5 * logvar)` | `std = torch.exp(0.5 * logvar)` 把 log 方差转成标准差。 |
| `eps = torch.randn_like(std)` | `eps = torch.randn_like(std)` 采样标准正态噪声。 |
| `z = mu + std * eps` | `z = mu + std * eps` 是重参数化技巧，使采样过程可反向传播。 |
| 概念 / 经验 | 重建损失衡量 decoder 输出和输入的接近程度。 |
| `-0.5 * mean(1 + logvar - mu^2 - exp(logvar))` | KL loss 让潜变量分布接近标准正态，常写成 `-0.5 * mean(1 + logvar - mu^2 - exp(logvar))`。 |
| `recon_loss + beta * kl` | 总 loss 通常是 `recon_loss + beta * kl`，beta 可调节压缩/生成约束强度。 |
| 概念 / 经验 | logvar 比直接预测 variance 更稳定，也更不容易出现负方差。 |
| 概念 / 经验 | 训练 VAE 时 recon 和 KL 要分别打印，方便判断哪一项主导。 |
| 概念 / 经验 | VAE 训练好后可以从标准正态采样 z，再用 decoder 生成数据。 |

## 练习前代码骨架

下面不是答案，只是提醒你代码应该从哪里开始。真正实现请写在 lesson 文件里。

```python
def reparameterize(mu, logvar):
    # TODO: 实现重参数化采样。
    pass


def main():
    # TODO: 训练一个 tiny VAE。
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

- 实现 reparameterize(mu, logvar)。
- 写一个极小 VAE 处理二维数据。
- 训练时同时计算 recon loss 和 KL loss。

## 完成标准

- 练习文件能直接运行，没有异常。
- 你能说清楚每个关键张量的 shape。
- 你能解释为什么使用这个 loss、这个 dtype、这个输出维度。
- 训练任务的 loss 或指标有合理变化，而不是完全随机。
- 看答案后，能关掉答案再独立复写一遍。

## 文件位置

- 练习文件：`../lessons/lesson_29_vae_reparameterization.py`
- 答案文件：`../answers/answer_29_vae_reparameterization.py`

建议先独立完成练习文件，再打开答案对照。能不看答案复写一遍，才算真的吃下来了。
