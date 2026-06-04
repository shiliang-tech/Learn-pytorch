# 第 27 课：TransformerEncoder：序列建模小例子

> 这一页只讲本节知识点和常用写法。练习题在 `lessons` 目录，答案在 `answers` 目录。

## 核心概念

- TransformerEncoder 用自注意力让序列中每个位置都能看见其他位置。
- 它通常需要 embedding、位置编码或某种位置信息。
- nn.TransformerEncoderLayer 可以快速搭出标准 encoder block。

## 关键写法详解

### 1. 常用写法

```python
nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
```

说明：

`nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)` 创建一个 encoder 层。

### 2. nn.TransformerEncoder(layer, num_layers=...)

```python
nn.TransformerEncoder(layer, num_layers=...)
```

说明：

`nn.TransformerEncoder(layer, num_layers=...)` 堆叠多个 encoder 层。

### 3. nn.Embedding

```python
nn.Embedding
[batch, seq] -> [batch, seq, dim]
```

说明：

输入到 Transformer 前通常先经过 `nn.Embedding`，shape `[batch, seq] -> [batch, seq, dim]`。

### 4. 概念和经验

说明：

Transformer 本身不知道顺序，需要加位置编码或可学习位置参数。

### 5. d_model

```python
d_model
nhead
```

说明：

`d_model` 必须能被 `nhead` 整除。

### 6. 概念和经验

说明：

分类任务可用平均池化、最后位置、或专门的 CLS token 得到序列表示。

### 7. src_key_padding_mask

```python
src_key_padding_mask
```

说明：

padding 序列时要传 `src_key_padding_mask`，避免模型关注 padding。

### 8. 概念和经验

说明：

Transformer 输出 shape 和输入 embedding shape 通常一致。

### 9. 概念和经验

说明：

小数据上 Transformer 不一定比 MLP/RNN 好，先理解机制更重要。

### 10. 概念和经验

说明：

如果训练很慢，可以先减小 dim、层数和序列长度。


## 本节任务

- 构造 token 序列，标签为第一个 token 是否大于最后一个 token。
- 使用 Embedding + TransformerEncoder。
- 取序列表示平均后分类。
