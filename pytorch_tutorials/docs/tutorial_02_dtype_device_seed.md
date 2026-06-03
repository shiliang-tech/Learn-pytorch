# 第 02 课：dtype、device 与随机种子

> 这份文档是本节的详细教程和速查表。先读它，再去写 `lesson` 文件；卡住时回来查，不要急着打开答案。

## 学习路径

| 步骤 | 你要做什么 | 目的 |
| --- | --- | --- |
| 1 | 读完本页的“核心概念”和“关键写法” | 先理解整体思路，知道 API 在解决什么问题 |
| 2 | 打开 `../lessons/lesson_02_dtype_device_seed.py` 补全 TODO | 把概念变成能运行的代码 |
| 3 | 运行练习文件，观察输出 | 用 shape、loss、accuracy 判断代码是否合理 |
| 4 | 最后对照 `../answers/answer_02_dtype_device_seed.py` | 查漏补缺，而不是直接抄答案 |

## 核心概念

- dtype 决定数值类型，常见训练默认用 torch.float32；分类标签常用 torch.long。
- device 决定张量在 CPU 还是 GPU。模型和输入必须在同一个 device 上。
- 随机种子让实验更容易复现，尤其是初始化、随机数据、DataLoader shuffle。

## 理解思路

写 PyTorch 代码时，先不要把注意力放在“背 API”上，而是先问四个问题：

1. 数据是什么形状？例如 `[batch, features]`、`[batch, time, features]` 或 `[N, C, H, W]`。
2. 标签是什么类型？回归通常是浮点，分类通常是整数类别编号或 0/1 浮点标签。
3. 模型输入和输出应该是什么 shape？loss 函数会严格要求它们匹配。
4. 哪些张量需要梯度，哪些只是数据或指标？不要让日志、评估和保存逻辑干扰计算图。

## 关键写法速查

| 写法 | 什么时候用 |
| --- | --- |
| `torch.manual_seed(42)` | `torch.manual_seed(42)` 固定 CPU 随机数，让初始化和随机数据更容易复现。 |
| `torch.cuda.manual_seed_all(42)` | 如果使用 CUDA，也常见写法是 `torch.cuda.manual_seed_all(42)`，用于多 GPU 随机种子。 |
| `dtype`<br>`torch.float32`<br>`torch.long` | `dtype` 决定数值类型；模型输入常用 `torch.float32`，分类标签常用 `torch.long`。 |
| `x.float()`<br>`x.long()`<br>`x.double()` | `x.float()`、`x.long()`、`x.double()` 是常见 dtype 快捷转换。 |
| `x.to(torch.float32)`<br>`x.to(device)`<br>`x.to(device=device, dtype=torch.float32)` | `x.to(torch.float32)` 可以转换 dtype；`x.to(device)` 可以移动设备；`x.to(device=device, dtype=torch.float32)` 可同时处理。 |
| `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` | `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` 是 CPU/GPU 通用代码入口。 |
| `model.to(device)`<br>`batch.to(device)` | 模型和数据必须在同一个 device 上，`model.to(device)` 和 `batch.to(device)` 通常要配套出现。 |
| `torch.randn(2, 3, device=device, dtype=torch.float32)` | `torch.randn(2, 3, device=device, dtype=torch.float32)` 可以创建时直接指定 device 和 dtype。 |
| `x.cpu()`<br>`x.detach().cpu().numpy()` | `x.cpu()` 把张量移回 CPU；转 NumPy 前通常需要 `x.detach().cpu().numpy()`。 |
| `Expected all tensors to be on the same device` | 常见错误 `Expected all tensors to be on the same device` 基本就是模型、输入、标签有东西没搬到同一个设备。 |
| `expected scalar type Long` | 常见错误 `expected scalar type Long` 多半是分类标签 dtype 不对，CrossEntropyLoss 要 long 类别编号。 |

## 练习前代码骨架

下面不是答案，只是提醒你代码应该从哪里开始。真正实现请写在 lesson 文件里。

```python
def main():
    # TODO: 设置随机种子，练习 dtype 与 device 转换。
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

## 本节任务

- 设置 torch.manual_seed(42)。
- 创建一个随机张量，把它转换成 float64，再转换回 float32。
- 选择 cuda 或 cpu 作为 device，并把张量移动过去。

## 完成标准

- 练习文件能直接运行，没有异常。
- 你能说清楚每个关键张量的 shape。
- 你能解释为什么使用这些 API，以及为什么要打印这些检查项。
- 看答案后，能关掉答案再独立复写一遍。

## 文件位置

- 练习文件：`../lessons/lesson_02_dtype_device_seed.py`
- 答案文件：`../answers/answer_02_dtype_device_seed.py`

建议先独立完成练习文件，再打开答案对照。能不看答案复写一遍，才算真的吃下来了。
