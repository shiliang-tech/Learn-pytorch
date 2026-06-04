# 第 03 课：形状变换与广播机制

> 这一页只讲本节知识点和常用写法。练习题在 `lessons` 目录，答案在 `answers` 目录。

## 核心概念

- view/reshape 可以改变张量形状，但元素总数必须一致。
- unsqueeze/squeeze 用于增加或删除长度为 1 的维度。
- 广播让不同形状的张量参与运算，例如 batch 中每一行都加同一个偏置向量。

## 关键写法详解

### 1. x.reshape(new_shape)

```python
x.reshape(new_shape)
(3, 4)
```

说明：

`x.reshape(new_shape)` 改变形状，元素总数必须不变，例如 12 个元素可以变成 `(3, 4)`。

### 2. x.view(...)

```python
x.view(...)
reshape
```

说明：

`x.view(...)` 也能改形状，但要求内存连续；不确定时优先用 `reshape`。

### 3. x.flatten()

```python
x.flatten()
nn.Flatten()
```

说明：

`x.flatten()` 展平成一维；`nn.Flatten()` 常放在 CNN 进入 Linear 前。

### 4. x.unsqueeze(dim)

```python
x.unsqueeze(dim)
[3] -> [1, 3]
[3, 1]
```

说明：

`x.unsqueeze(dim)` 增加长度为 1 的维度，例如 `[3] -> [1, 3]` 或 `[3, 1]`。

### 5. x.squeeze(dim)

```python
x.squeeze(dim)
```

说明：

`x.squeeze(dim)` 删除指定的长度为 1 的维度；不指定 dim 会删掉所有长度为 1 的维度，使用时要小心 batch 维。

### 6. x.permute(0, 2, 3, 1)

```python
x.permute(0, 2, 3, 1)
```

说明：

`x.permute(0, 2, 3, 1)` 按任意顺序重排维度；图像数据经常在 NHWC 和 NCHW 间转换。

### 7. x.transpose(dim0, dim1)

```python
x.transpose(dim0, dim1)
x.T
```

说明：

`x.transpose(dim0, dim1)` 交换两个维度；二维矩阵也可以用 `x.T`。

### 8. 概念和经验

说明：

广播规则从最后一维开始对齐；两个维度相等，或其中一个为 1，才可以广播。

### 9. bias

```python
bias
[features]
[batch, features]
```

说明：

`bias` 形状是 `[features]` 时，可以自动加到 `[batch, features]` 的每一行。

### 10. keepdim=True

```python
keepdim=True
(x - mean) / std
```

说明：

`keepdim=True` 可以保留聚合后的维度，常用于 `(x - mean) / std` 这种标准化写法。

### 11. 概念和经验

说明：

如果广播报错，先打印参与运算的两个 shape，从最后一维往前对齐检查。


## 本节任务

- 把 torch.arange(12) 变成 3x4 矩阵。
- 创建长度为 4 的 bias，并加到矩阵每一行。
- 用 unsqueeze 把结果变成 1x3x4。
