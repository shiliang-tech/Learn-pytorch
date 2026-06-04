# 第 04 课：索引、切片与布尔 mask

> 这一页只讲本节知识点和常用写法。练习题在 `lessons` 目录，答案在 `answers` 目录。

## 核心概念

- 索引用来取局部数据，切片常用于 batch、序列、图像区域。
- 布尔 mask 可以筛选满足条件的元素，是处理异常值、条件统计的常用手段。
- 训练中经常会按标签、置信度、padding mask 来选择样本或位置。

## 关键写法详解

### 1. x[i]

```python
x[i]
x[-1]
```

说明：

`x[i]` 取第 i 行或第 i 个元素；索引从 0 开始，`x[-1]` 表示最后一个。

### 2. x[:, 0]

```python
x[:, 0]
x[1, :]
```

说明：

`x[:, 0]` 取所有行的第 0 列；`x[1, :]` 取第 1 行所有列。

### 3. x[:, :3]

```python
x[:, :3]
x[::2]
```

说明：

`x[:, :3]` 取所有行的前 3 列；`x[::2]` 每隔一个取一次。

### 4. x[start:end]

```python
x[start:end]
end
```

说明：

`x[start:end]` 左闭右开，不包含 `end` 位置。

### 5. mask = x > 10

```python
mask = x > 10
x[mask]
```

说明：

`mask = x > 10` 会得到布尔张量；`x[mask]` 会筛出满足条件的元素。

### 6. torch.where(condition, a, b)

```python
torch.where(condition, a, b)
```

说明：

`torch.where(condition, a, b)` 按条件从 a/b 中选值，适合做条件替换。

### 7. x.argmax(dim=1)

```python
x.argmax(dim=1)
```

说明：

`x.argmax(dim=1)` 常用于从分类 logits 得到预测类别。

### 8. x.gather(dim, index)

```python
x.gather(dim, index)
```

说明：

`x.gather(dim, index)` 可以按 index 收集元素，常见于更复杂的分类或序列任务。

### 9. x.scatter_(dim, index, value)

```python
x.scatter_(dim, index, value)
_
```

说明：

`x.scatter_(dim, index, value)` 可以按 index 写入，函数名带 `_` 表示原地修改。

### 10. x[mask] = 0

```python
x[mask] = 0
```

说明：

原地修改如 `x[mask] = 0` 很方便，但在 autograd 计算图里要谨慎，可能破坏梯度需要的中间值。


## 本节任务

- 创建 4x5 的矩阵。
- 取出第 2 行、前 3 列、最后一列。
- 用 mask 找出所有大于 10 的元素，并计算它们的均值。
