# 第 09 课：TensorDataset 与 DataLoader

> 这一页只讲本节知识点和常用写法。练习题在 `lessons` 目录，答案在 `answers` 目录。

## 核心概念

- Dataset 定义如何取一个样本，DataLoader 负责批量、打乱、多进程读取。
- batch 训练比一次喂全部数据更常见，也更接近真实项目。
- shuffle=True 常用于训练集，验证集和测试集通常不需要 shuffle。

## 关键写法详解

### 1. TensorDataset(x, y)

```python
TensorDataset(x, y)
```

说明：

`TensorDataset(x, y)` 把多个张量按第 0 维对齐，组成样本集。
- 数据管道的重点是第 0 维样本数一致。进入训练循环后，每个 batch 都应该能直接喂给模型。

### 2. DataLoader(dataset, batch_size=32, shuffle=True)

```python
DataLoader(dataset, batch_size=32, shuffle=True)
```

说明：

`DataLoader(dataset, batch_size=32, shuffle=True)` 自动切 batch 并打乱训练数据。
- 数据管道的重点是第 0 维样本数一致。进入训练循环后，每个 batch 都应该能直接喂给模型。

### 3. for xb, yb in loader:

```python
for xb, yb in loader:
```

说明：

`for xb, yb in loader:` 是 mini-batch 训练的基本写法。
- 数据管道的重点是第 0 维样本数一致。进入训练循环后，每个 batch 都应该能直接喂给模型。

### 4. len(dataset)

```python
len(dataset)
```

说明：

Dataset 里的第 0 维通常是样本数，`len(dataset)` 返回样本数量。
- 数据管道的重点是第 0 维样本数一致。进入训练循环后，每个 batch 都应该能直接喂给模型。

### 5. shuffle=True

```python
shuffle=True
```

说明：

`shuffle=True` 常用于训练集；验证/测试集一般设为 False。

### 6. zero_grad -> backward -> step

```python
zero_grad -> backward -> step
```

说明：

每个 batch 内仍要做完整训练三连：`zero_grad -> backward -> step`。
- 训练时关注顺序：先前向得到输出，再算 loss，然后清空旧梯度、反向传播、更新参数。顺序乱了通常不会得到正确训练。
- 和梯度有关的写法要小心计算图是否被断开。用于打印日志时可以 `.item()`，但参与训练的张量不要过早转成 Python 数字。
- 数据管道的重点是第 0 维样本数一致。进入训练循环后，每个 batch 都应该能直接喂给模型。

### 7. xb

```python
xb
yb
```

说明：

如果使用 GPU，要在循环里把每个 `xb`、`yb` 移到 device。
- 如果代码要兼容 CPU/GPU，模型、输入、标签和新建临时张量都要放在同一个 device。

### 8. drop_last=True

```python
drop_last=True
```

说明：

`drop_last=True` 可以丢弃最后一个不满 batch 的小批次，BatchNorm 或固定形状场景可能有用。
- 数据管道的重点是第 0 维样本数一致。进入训练循环后，每个 batch 都应该能直接喂给模型。

### 9. if __name__ == '__main__':

```python
if __name__ == '__main__':
```

说明：

Windows 上自定义多进程 DataLoader 时要注意 `if __name__ == '__main__':` 保护。
- 数据管道的重点是第 0 维样本数一致。进入训练循环后，每个 batch 都应该能直接喂给模型。

### 10. num_workers=0

```python
num_workers=0
```

说明：

刚入门时 `num_workers=0` 最稳，等流程跑通再考虑加速数据读取。


## 本节任务

- 用 TensorDataset 包装 x 和 y。
- 用 DataLoader 每次取 16 条样本。
- 写一个 mini-batch 训练循环拟合线性回归。
