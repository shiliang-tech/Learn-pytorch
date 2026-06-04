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

### 2. torch.cuda.manual_seed_all(42)

```python
torch.cuda.manual_seed_all(42)
```

说明：

如果使用 CUDA，也常见写法是 `torch.cuda.manual_seed_all(42)`，用于多 GPU 随机种子。

### 3. dtype

```python
dtype
torch.float32
torch.long
```

说明：

`dtype` 决定数值类型；模型输入常用 `torch.float32`，分类标签常用 `torch.long`。

### 4. x.float()

```python
x.float()
x.long()
x.double()
```

说明：

`x.float()`、`x.long()`、`x.double()` 是常见 dtype 快捷转换。

### 5. x.to(torch.float32)

```python
x.to(torch.float32)
x.to(device)
x.to(device=device, dtype=torch.float32)
```

说明：

`x.to(torch.float32)` 可以转换 dtype；`x.to(device)` 可以移动设备；`x.to(device=device, dtype=torch.float32)` 可同时处理。

### 6. 常用写法

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

说明：

`device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` 是 CPU/GPU 通用代码入口。

### 7. model.to(device)

```python
model.to(device)
batch.to(device)
```

说明：

模型和数据必须在同一个 device 上，`model.to(device)` 和 `batch.to(device)` 通常要配套出现。

### 8. 常用写法

```python
torch.randn(2, 3, device=device, dtype=torch.float32)
```

说明：

`torch.randn(2, 3, device=device, dtype=torch.float32)` 可以创建时直接指定 device 和 dtype。

### 9. x.cpu()

```python
x.cpu()
x.detach().cpu().numpy()
```

说明：

`x.cpu()` 把张量移回 CPU；转 NumPy 前通常需要 `x.detach().cpu().numpy()`。

### 10. Expected all tensors to be on the same device

```python
Expected all tensors to be on the same device
```

说明：

常见错误 `Expected all tensors to be on the same device` 基本就是模型、输入、标签有东西没搬到同一个设备。

### 11. expected scalar type Long

```python
expected scalar type Long
```

说明：

常见错误 `expected scalar type Long` 多半是分类标签 dtype 不对，CrossEntropyLoss 要 long 类别编号。


## 本节任务

- 设置 torch.manual_seed(42)。
- 创建一个随机张量，把它转换成 float64，再转换回 float32。
- 选择 cuda 或 cpu 作为 device，并把张量移动过去。
