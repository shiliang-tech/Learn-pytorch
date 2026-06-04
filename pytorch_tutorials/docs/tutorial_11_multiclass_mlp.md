# 第 11 课：多分类 MLP 与 CrossEntropyLoss

> 这一页只讲本节知识点和常用写法。练习题在 `lessons` 目录，答案在 `answers` 目录。

## 核心概念

- 多分类模型输出每一类的 logit，shape 通常是 [batch, num_classes]。
- CrossEntropyLoss 内部包含 log_softmax，标签必须是类别索引 long 类型。
- MLP 通过 Linear + 非线性激活堆叠，能学习比线性模型更复杂的边界。

## 关键写法详解

### 1. 形状约定

```python
[batch, num_classes]
```

说明：

多分类输出 shape 是 `[batch, num_classes]`，每一列对应一个类别 logit。

### 2. nn.CrossEntropyLoss()

```python
nn.CrossEntropyLoss()
```

说明：

`nn.CrossEntropyLoss()` 输入 raw logits，不要先 softmax。

### 3. 形状约定

```python
[batch]
torch.long
```

说明：

CrossEntropyLoss 的标签 shape 通常是 `[batch]`，dtype 必须是 `torch.long`。

### 4. pred = logits.argmax(dim=1)

```python
pred = logits.argmax(dim=1)
```

说明：

`pred = logits.argmax(dim=1)` 得到预测类别编号。

### 5. Linear -> ReLU -> Linear

```python
Linear -> ReLU -> Linear
```

说明：

MLP 常见结构是 `Linear -> ReLU -> Linear`，隐藏层可以重复堆叠。

### 6. nn.Sequential

```python
nn.Sequential
nn.Module
```

说明：

`nn.Sequential` 适合顺序结构；需要分支或多输入时自定义 `nn.Module`。

### 7. 概念和经验

说明：

隐藏层宽度如 16、32、64 是常见起点，不必一开始就很大。

### 8. (pred == y).float().mean().item()

```python
(pred == y).float().mean().item()
```

说明：

分类准确率可写成 `(pred == y).float().mean().item()`。

### 9. num_classes-1

```python
num_classes-1
```

说明：

如果 loss 不降，先检查标签是否从 0 到 `num_classes-1`，以及标签 dtype 是否 long。

### 10. torch.softmax(logits, dim=1)

```python
torch.softmax(logits, dim=1)
```

说明：

评估概率时再用 `torch.softmax(logits, dim=1)`。


## 本节任务

- 生成 3 类二维点。
- 搭建 Linear-ReLU-Linear 的 MLP。
- 使用 CrossEntropyLoss 训练并打印准确率。
