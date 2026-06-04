# 第 17 课：自定义 Dataset

> 这一页只讲本节知识点和常用写法。练习题在 `lessons` 目录，答案在 `answers` 目录。

## 核心概念

- 自定义 Dataset 至少实现 __len__ 和 __getitem__。
- __getitem__ 返回一个样本，可以是张量、标签、字典等结构。
- 把数据生成或预处理封装到 Dataset 后，训练循环会更稳定清晰。

## 关键写法详解

### 1. torch.utils.data.Dataset

```python
torch.utils.data.Dataset
```

说明：

自定义 Dataset 继承 `torch.utils.data.Dataset`。

### 2. __len__(self)

```python
__len__(self)
```

说明：

`__len__(self)` 返回样本数，DataLoader 依赖它判断一轮有多少数据。

### 3. __getitem__(self, idx)

```python
__getitem__(self, idx)
(x, y)
```

说明：

`__getitem__(self, idx)` 返回第 idx 个样本，可以是 `(x, y)`、字典或更多字段。

### 4. __init__

```python
__init__
__getitem__
```

说明：

数据可以在 `__init__` 中提前准备，也可以在 `__getitem__` 中按需读取。

### 5. 概念和经验

说明：

返回的数值最好转换成张量，避免训练循环里到处做类型转换。

### 6. DataLoader

```python
DataLoader
collate_fn
```

说明：

`DataLoader` 会把多个样本自动 collate 成 batch；形状不一致时需要自定义 `collate_fn`。

### 7. 概念和经验

说明：

回归标签通常是 float，分类标签通常是 long。

### 8. __getitem__

```python
__getitem__
```

说明：

数据增强常放在 Dataset 的 `__getitem__` 中。

### 9. 概念和经验

说明：

Dataset 不负责训练，只负责稳定地提供样本。

### 10. loader = DataLoader(dataset, batch_size=4)

```python
loader = DataLoader(dataset, batch_size=4)
```

说明：

先用小数据和 `loader = DataLoader(dataset, batch_size=4)` 打印一个 batch，确认 shape。


## 本节任务

- 写一个 SineDataset，输入 x，标签 y=sin(x)。
- 用 DataLoader 批量读取。
- 训练一个小 MLP 做回归。
