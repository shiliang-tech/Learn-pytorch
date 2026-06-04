# 第 26 课：Attention：Query、Key、Value

> 这一页只讲本节知识点和常用写法。练习题在 `lessons` 目录，答案在 `answers` 目录。

## 核心概念

- Attention 用 query 和 key 的相似度作为权重，再对 value 做加权求和。
- 缩放点积注意力会除以 sqrt(d_k)，避免维度变大时 logits 过大。
- 注意力权重经过 softmax 后，每一行通常和为 1。

## 关键写法详解

### 1. d_k

```python
d_k
d_model
```

说明：

Q/K/V 的最后一维是特征维，常记作 `d_k` 或 `d_model`。

### 2. scores = q @ k.transpose(-2, -1)

```python
scores = q @ k.transpose(-2, -1)
```

说明：

`scores = q @ k.transpose(-2, -1)` 得到每个 query 对每个 key 的相似度。

### 3. 公式写法

```python
scores / math.sqrt(q.size(-1))
```

说明：

`scores / math.sqrt(q.size(-1))` 是缩放点积注意力的标准写法。

### 4. weights = torch.softmax(scores, dim=-1)

```python
weights = torch.softmax(scores, dim=-1)
```

说明：

`weights = torch.softmax(scores, dim=-1)` 得到注意力权重。
- 分类任务要分清 logits 和概率。大多数 PyTorch loss 直接接收 logits，只有推理或展示结果时才需要 sigmoid/softmax。

### 5. output = weights @ v

```python
output = weights @ v
```

说明：

`output = weights @ v` 用权重对 value 加权求和。

### 6. 形状约定

```python
[batch, seq, dim]
```

说明：

输入可为 `[batch, seq, dim]`，输出通常也是 `[batch, seq, dim]`。
- 写这类代码时，第一步先确认张量形状。PyTorch 的很多报错不是公式错了，而是某一维没有对齐。
- 数据管道的重点是第 0 维样本数一致。进入训练循环后，每个 batch 都应该能直接喂给模型。

### 7. -1e9

```python
-1e9
```

说明：

mask 可在 softmax 前把不可见位置填成很小的数，如 `-1e9`。
- 分类任务要分清 logits 和概率。大多数 PyTorch loss 直接接收 logits，只有推理或展示结果时才需要 sigmoid/softmax。

### 8. 概念和经验

说明：

自注意力是 q/k/v 都来自同一个序列；交叉注意力是 q 和 k/v 来自不同序列。

### 9. 概念和经验

说明：

多头注意力会把特征维拆成多个 head 并行计算。

### 10. 概念和经验

说明：

排查 attention 时检查 weights 最后一维求和是否接近 1。


## 本节任务

- 创建 q、k、v 三个张量。
- 计算 scaled dot-product attention。
- 验证 attention weights 最后一维求和为 1。
