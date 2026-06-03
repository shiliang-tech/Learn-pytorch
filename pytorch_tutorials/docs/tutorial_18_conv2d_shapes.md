# 第 18 课：图像张量与 Conv2d 形状

> 这份文档是本节的详细教程和速查表。先读它，再去写 `lesson` 文件；卡住时回来查，不要急着打开答案。

## 学习路径

| 步骤 | 你要做什么 | 目的 |
| --- | --- | --- |
| 1 | 读完本页的“核心概念”和“关键写法” | 先理解整体思路，知道 API 在解决什么问题 |
| 2 | 打开 `../lessons/lesson_18_conv2d_shapes.py` 补全 TODO | 把概念变成能运行的代码 |
| 3 | 运行练习文件，观察输出 | 用 shape、loss、accuracy 判断代码是否合理 |
| 4 | 最后对照 `../answers/answer_18_conv2d_shapes.py` | 查漏补缺，而不是直接抄答案 |

## 核心概念

- PyTorch 图像张量常用 NCHW：batch、channel、height、width。
- Conv2d 会用卷积核在空间维度滑动，输出通道数由 out_channels 决定。
- padding、stride、kernel_size 会共同决定输出高宽。

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
| `[N, C, H, W]`<br>`[N, H, W, C]` | PyTorch 卷积输入是 `[N, C, H, W]`，不是 `[N, H, W, C]`。 |
| `nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)` | `nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)` 是基本写法。 |
| `out_channels` | `out_channels` 决定输出通道数，也就是学多少个卷积核。 |
| `padding=1, kernel_size=3, stride=1` | `padding=1, kernel_size=3, stride=1` 常保持高宽不变。 |
| `nn.MaxPool2d(2)` | `nn.MaxPool2d(2)` 会把高宽各减半，通道数不变。 |
| `nn.AdaptiveAvgPool2d((1, 1))` | `nn.AdaptiveAvgPool2d((1, 1))` 可把任意高宽压到 1x1，便于接分类头。 |
| `(H + 2P - K) / S + 1` | 卷积输出高宽公式大致是 `(H + 2P - K) / S + 1`，向下取整。 |
| `nn.Flatten()` | 进入 Linear 前用 `nn.Flatten()`，并确认 flatten 后维度。 |
| 概念 / 经验 | 图像从 PIL/NumPy 来时常是 HWC，需要转换到 CHW。 |
| 概念 / 经验 | 排查 CNN 第一件事：打印每层输出 shape。 |

## 练习前代码骨架

下面不是答案，只是提醒你代码应该从哪里开始。真正实现请写在 lesson 文件里。

```python
def main():
    # TODO: 创建图像 batch，经过卷积和池化，打印 shape。
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

- 创建形状 [8,1,28,28] 的随机图片 batch。
- 用 Conv2d(1,4,kernel_size=3,padding=1) 处理。
- 接 MaxPool2d(2)，打印每一步 shape。

## 完成标准

- 练习文件能直接运行，没有异常。
- 你能说清楚每个关键张量的 shape。
- 你能解释为什么使用这些 API，以及为什么要打印这些检查项。
- 看答案后，能关掉答案再独立复写一遍。

## 文件位置

- 练习文件：`../lessons/lesson_18_conv2d_shapes.py`
- 答案文件：`../answers/answer_18_conv2d_shapes.py`

建议先独立完成练习文件，再打开答案对照。能不看答案复写一遍，才算真的吃下来了。
