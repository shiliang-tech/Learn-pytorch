# 第 24 课：RNN：序列分类入门

> 这一页只讲本节知识点和常用写法。练习题在 `lessons` 目录，答案在 `answers` 目录。

## 核心概念

- RNN 会按时间步处理序列，隐藏状态携带之前的信息。
- 输入形状常用 [batch, time, features]，需要设置 batch_first=True。
- 序列分类常取最后一个时间步的隐藏状态接分类层。

## 关键写法详解

### 1. 常用写法

```python
nn.RNN(input_size, hidden_size, batch_first=True)
[batch, time, features]
```

说明：

`nn.RNN(input_size, hidden_size, batch_first=True)` 接收 `[batch, time, features]`。

### 2. (out, h_n)

```python
(out, h_n)
out
h_n
```

说明：

RNN 返回 `(out, h_n)`；`out` 是每个时间步输出，`h_n` 是最后隐藏状态。

### 3. out[:, -1, :]

```python
out[:, -1, :]
h_n[-1]
```

说明：

序列分类常用 `out[:, -1, :]` 或 `h_n[-1]` 作为整体表示。

### 4. 形状约定

```python
[batch, time, 1]
```

说明：

输入一维数值序列时，features 维也要保留，例如 `[batch, time, 1]`。

### 5. hidden_size

```python
hidden_size
```

说明：

`hidden_size` 决定隐藏状态维度，也是分类头 Linear 的输入维度。

### 6. 概念和经验

说明：

普通 RNN 容易梯度消失，长序列更常用 LSTM/GRU/Transformer。

### 7. pack_padded_sequence

```python
pack_padded_sequence
```

说明：

变长序列可以用 padding + mask，或 `pack_padded_sequence`。

### 8. 概念和经验

说明：

RNN 训练也使用普通的 optimizer/loss/backward/step。

### 9. 形状约定

```python
[batch, time, features]
```

说明：

序列任务排错重点是 `[batch, time, features]` 维度顺序。

### 10. batch_first=False

```python
batch_first=False
[time, batch, features]
```

说明：

如果 `batch_first=False`，输入顺序会变成 `[time, batch, features]`，初学建议设 True。


## 本节任务

- 生成长度为 8 的一维序列。
- 标签为序列后半段均值是否大于前半段均值。
- 使用 nn.RNN 做二分类。
