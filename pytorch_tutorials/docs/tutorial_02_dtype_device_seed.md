# 第 02 课：dtype、device 与随机种子

> 这一页只讲本节知识点和常用写法。练习题在 `lessons` 目录，答案在 `answers` 目录。

## 核心概念

- dtype 决定数值类型，常见训练默认用 torch.float32；分类标签常用 torch.long。
- device 决定张量在 CPU 还是 GPU。模型和输入必须在同一个 device 上。
- 随机种子让实验更容易复现，尤其是初始化、随机数据、DataLoader shuffle。

## 关键写法详解

### 1. torch.manual_seed(42)

```python
torch.manual_seed(42)
```

说明：

`torch.manual_seed(42)` 固定 CPU 随机数，让初始化和随机数据更容易复现。
- 如果代码要兼容 CPU/GPU，模型、输入、标签和新建临时张量都要放在同一个 device。

### 2. torch.cuda.manual_seed_all(42)

```python
torch.cuda.manual_seed_all(42)
```

说明：

如果使用 CUDA，也常见写法是 `torch.cuda.manual_seed_all(42)`，用于多 GPU 随机种子。
- 如果代码要兼容 CPU/GPU，模型、输入、标签和新建临时张量都要放在同一个 device。

### 3. dtype

```python
dtype
torch.float32
torch.long
```

说明：

`dtype` 决定数值类型；模型输入常用 `torch.float32`，分类标签常用 `torch.long`。
- 同时注意 dtype：模型输入通常是浮点张量，分类标签通常是 `torch.long`，二分类 BCE 标签通常是浮点 0/1。

### 4. x.float()

```python
x.float()
x.long()
x.double()
```

说明：

`x.float()`、`x.long()`、`x.double()` 是常见 dtype 快捷转换。
- 同时注意 dtype：模型输入通常是浮点张量，分类标签通常是 `torch.long`，二分类 BCE 标签通常是浮点 0/1。

### 5. x.to(torch.float32)

```python
x.to(torch.float32)
x.to(device)
x.to(device=device, dtype=torch.float32)
```

说明：

`x.to(torch.float32)` 可以转换 dtype；`x.to(device)` 可以移动设备；`x.to(device=device, dtype=torch.float32)` 可同时处理。
- 同时注意 dtype：模型输入通常是浮点张量，分类标签通常是 `torch.long`，二分类 BCE 标签通常是浮点 0/1。
- 如果代码要兼容 CPU/GPU，模型、输入、标签和新建临时张量都要放在同一个 device。

### 6. 常用写法

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

说明：

`device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` 是 CPU/GPU 通用代码入口。
- 如果代码要兼容 CPU/GPU，模型、输入、标签和新建临时张量都要放在同一个 device。

### 7. model.to(device)

```python
model.to(device)
batch.to(device)
```

说明：

模型和数据必须在同一个 device 上，`model.to(device)` 和 `batch.to(device)` 通常要配套出现。
- 如果代码要兼容 CPU/GPU，模型、输入、标签和新建临时张量都要放在同一个 device。
- 数据管道的重点是第 0 维样本数一致。进入训练循环后，每个 batch 都应该能直接喂给模型。

### 8. 常用写法

```python
torch.randn(2, 3, device=device, dtype=torch.float32)
```

说明：

`torch.randn(2, 3, device=device, dtype=torch.float32)` 可以创建时直接指定 device 和 dtype。
- 同时注意 dtype：模型输入通常是浮点张量，分类标签通常是 `torch.long`，二分类 BCE 标签通常是浮点 0/1。
- 如果代码要兼容 CPU/GPU，模型、输入、标签和新建临时张量都要放在同一个 device。

### 9. x.cpu()

```python
x.cpu()
x.detach().cpu().numpy()
```

说明：

`x.cpu()` 把张量移回 CPU；转 NumPy 前通常需要 `x.detach().cpu().numpy()`。
- 如果代码要兼容 CPU/GPU，模型、输入、标签和新建临时张量都要放在同一个 device。
- 和梯度有关的写法要小心计算图是否被断开。用于打印日志时可以 `.item()`，但参与训练的张量不要过早转成 Python 数字。

### 10. Expected all tensors to be on the same device

```python
Expected all tensors to be on the same device
```

说明：

常见错误 `Expected all tensors to be on the same device` 基本就是模型、输入、标签有东西没搬到同一个设备。
- 如果代码要兼容 CPU/GPU，模型、输入、标签和新建临时张量都要放在同一个 device。

### 11. expected scalar type Long

```python
expected scalar type Long
```

说明：

常见错误 `expected scalar type Long` 多半是分类标签 dtype 不对，CrossEntropyLoss 要 long 类别编号。
- 同时注意 dtype：模型输入通常是浮点张量，分类标签通常是 `torch.long`，二分类 BCE 标签通常是浮点 0/1。
- 训练时关注顺序：先前向得到输出，再算 loss，然后清空旧梯度、反向传播、更新参数。顺序乱了通常不会得到正确训练。
- 分类任务要分清 logits 和概率。大多数 PyTorch loss 直接接收 logits，只有推理或展示结果时才需要 sigmoid/softmax。


## 本节任务

- 设置 torch.manual_seed(42)。
- 创建一个随机张量，把它转换成 float64，再转换回 float32。
- 选择 cuda 或 cpu 作为 device，并把张量移动过去。
