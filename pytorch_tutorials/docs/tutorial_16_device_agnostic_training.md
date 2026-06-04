# 第 16 课：设备无关训练：CPU/GPU 通用代码

> 这一页只讲本节知识点和常用写法。练习题在 `lessons` 目录，答案在 `answers` 目录。

## 核心概念

- 设备无关代码会先选择 device，然后把模型和每个 batch 都移动到这个 device。
- 不要只移动模型或只移动数据，否则会出现 device mismatch 错误。
- 保存模型时通常仍保存 state_dict；加载到 CPU 可使用 map_location='cpu'。

## 关键写法详解

### 1. 常用写法

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

说明：

统一写 `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`。

### 2. model.to(device)

```python
model.to(device)
```

说明：

模型创建后调用 `model.to(device)`。

### 3. xb = xb.to(device); yb = yb.to(device)

```python
xb = xb.to(device); yb = yb.to(device)
```

说明：

每个 batch 进入模型前调用 `xb = xb.to(device); yb = yb.to(device)`。

### 4. torch.zeros_like(x)

```python
torch.zeros_like(x)
device=device
```

说明：

新创建的临时张量也要注意 device，可用 `torch.zeros_like(x)` 或指定 `device=device`。

### 5. .cuda()

```python
.cuda()
```

说明：

不要在通用代码里硬写 `.cuda()`，没有 GPU 的机器会直接失败。

### 6. loss.item()

```python
loss.item()
```

说明：

日志里的 loss 可用 `loss.item()`，不用把整个模型或 batch 搬回 CPU。

### 7. x.detach().cpu().numpy()

```python
x.detach().cpu().numpy()
```

说明：

转 NumPy 前必须在 CPU 上：`x.detach().cpu().numpy()`。

### 8. map_location

```python
map_location
```

说明：

保存 state_dict 时不需要特别处理 device；加载时可用 `map_location` 控制。

### 9. tensor.device

```python
tensor.device
```

说明：

device mismatch 错误几乎都能通过打印 `tensor.device` 定位。

### 10. 概念和经验

说明：

多 GPU、分布式训练是更高阶主题，单机入门先把单 device 写规范。


## 本节任务

- 选择 cuda 或 cpu。
- 把模型、输入和标签都移动到 device。
- 训练一个小分类器并打印当前 device。
