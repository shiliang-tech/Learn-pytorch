# 第 05 课：聚合统计、范数与标准化

> 这一页只讲本节知识点和常用写法。练习题在 `lessons` 目录，答案在 `answers` 目录。

## 核心概念

- mean、sum、max、argmax 这类 reduction 会沿某些维度汇总信息。
- 范数常用来度量向量大小，训练中也会用于正则化、梯度裁剪、相似度计算。
- 标准化让数据均值接近 0、标准差接近 1，通常能让优化更稳定。

## 关键写法详解

### 1. x.mean()

```python
x.mean()
x.mean(dim=0)
```

说明：

`x.mean()` 对所有元素求均值；`x.mean(dim=0)` 沿第 0 维聚合，常得到每列均值。

### 2. x.sum(dim=1)

```python
x.sum(dim=1)
```

说明：

`x.sum(dim=1)` 对每一行求和，常用于统计每个样本的特征总量。

### 3. x.max(dim=1)

```python
x.max(dim=1)
(values, indices)
x.argmax(dim=1)
```

说明：

`x.max(dim=1)` 返回 `(values, indices)`，多分类里更常用 `x.argmax(dim=1)`。

### 4. keepdim=True

```python
keepdim=True
```

说明：

`keepdim=True` 保留被聚合维度，方便后续和原张量广播运算。

### 5. x.std(dim=0)

```python
x.std(dim=0)
(x - mean) / (std + 1e-8)
```

说明：

`x.std(dim=0)` 计算标准差；标准化常写成 `(x - mean) / (std + 1e-8)`。

### 6. torch.linalg.vector_norm(x, dim=1)

```python
torch.linalg.vector_norm(x, dim=1)
```

说明：

`torch.linalg.vector_norm(x, dim=1)` 计算每行向量范数。

### 7. torch.clamp(x, min=0, max=1)

```python
torch.clamp(x, min=0, max=1)
```

说明：

`torch.clamp(x, min=0, max=1)` 可以把数值限制在区间内。

### 8. torch.nan_to_num(x)

```python
torch.nan_to_num(x)
```

说明：

`torch.nan_to_num(x)` 可以把 NaN/Inf 替换成有限数，在排查坏数据时有用。

### 9. dim

```python
dim
```

说明：

`dim` 是 PyTorch 最重要的参数之一；不知道该写几，就先把 shape 按维度编号写在纸上。
- 写这类代码时，第一步先确认张量形状。PyTorch 的很多报错不是公式错了，而是某一维没有对齐。

### 10. .item()

```python
.item()
```

说明：

统计值参与训练时仍然是张量；只打印时再 `.item()`，不要过早把它变成 Python 数字。


## 本节任务

- 创建一个 5x3 的随机数据矩阵。
- 计算每一列的均值和标准差。
- 把每一列标准化，并验证标准化后的均值接近 0。
