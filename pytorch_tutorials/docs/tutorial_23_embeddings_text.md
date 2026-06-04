# 第 23 课：Embedding：把离散 id 变成向量

> 这一页只讲本节知识点和常用写法。练习题在 `lessons` 目录，答案在 `answers` 目录。

## 核心概念

- Embedding 本质是一个可训练查表矩阵，把 token id 映射成稠密向量。
- 文本、类别 id、用户 id、商品 id 都常用 embedding 表示。
- 简单文本分类可以先对词向量求平均，再接线性分类层。

## 关键写法详解

### 1. nn.Embedding(vocab_size, embedding_dim)

```python
nn.Embedding(vocab_size, embedding_dim)
```

说明：

`nn.Embedding(vocab_size, embedding_dim)` 输入 token id，输出 token 向量。

### 2. torch.long

```python
torch.long
```

说明：

Embedding 输入必须是整数张量，dtype 通常是 `torch.long`。

### 3. 形状约定

```python
[batch, seq_len]
[batch, seq_len, dim]
```

说明：

输入 shape `[batch, seq_len]` 经过 Embedding 后变成 `[batch, seq_len, dim]`。

### 4. emb.mean(dim=1)

```python
emb.mean(dim=1)
```

说明：

`emb.mean(dim=1)` 是平均池化，可把序列变成句向量。

### 5. 概念和经验

说明：

有 padding 时不能直接 mean，需要用 mask 排除 padding 位置。

### 6. padding_idx=0

```python
padding_idx=0
```

说明：

`padding_idx=0` 可以让某个 token 作为 padding，并让它的 embedding 不更新。

### 7. Embedding -> pooling -> Linear

```python
Embedding -> pooling -> Linear
```

说明：

文本分类头常写成 `Embedding -> pooling -> Linear`。

### 8. 概念和经验

说明：

词表大小 vocab_size 必须大于最大 token id。

### 9. 概念和经验

说明：

Embedding 也可用于用户 id、商品 id、类别 id，不只用于文本。

### 10. embedding.weight.data

```python
embedding.weight.data
```

说明：

预训练词向量可以赋给 `embedding.weight.data`，但入门先练随机初始化即可。


## 本节任务

- 构造整数 token 序列，标签由某个关键词是否出现决定。
- 使用 nn.Embedding 和 mean pooling。
- 训练一个二分类模型。
