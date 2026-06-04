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
- 写这类代码时，第一步先确认张量形状。PyTorch 的很多报错不是公式错了，而是某一维没有对齐。
- 分类任务要分清 logits 和概率。大多数 PyTorch loss 直接接收 logits，只有推理或展示结果时才需要 sigmoid/softmax。
- 数据管道的重点是第 0 维样本数一致。进入训练循环后，每个 batch 都应该能直接喂给模型。

### 2. nn.CrossEntropyLoss()

```python
nn.CrossEntropyLoss()
```

说明：

`nn.CrossEntropyLoss()` 输入 raw logits，不要先 softmax。
- 训练时关注顺序：先前向得到输出，再算 loss，然后清空旧梯度、反向传播、更新参数。顺序乱了通常不会得到正确训练。
- 分类任务要分清 logits 和概率。大多数 PyTorch loss 直接接收 logits，只有推理或展示结果时才需要 sigmoid/softmax。

### 3. 形状约定

```python
[batch]
torch.long
```

说明：

CrossEntropyLoss 的标签 shape 通常是 `[batch]`，dtype 必须是 `torch.long`。
- 写这类代码时，第一步先确认张量形状。PyTorch 的很多报错不是公式错了，而是某一维没有对齐。
- 同时注意 dtype：模型输入通常是浮点张量，分类标签通常是 `torch.long`，二分类 BCE 标签通常是浮点 0/1。
- 训练时关注顺序：先前向得到输出，再算 loss，然后清空旧梯度、反向传播、更新参数。顺序乱了通常不会得到正确训练。
- 分类任务要分清 logits 和概率。大多数 PyTorch loss 直接接收 logits，只有推理或展示结果时才需要 sigmoid/softmax。
- 数据管道的重点是第 0 维样本数一致。进入训练循环后，每个 batch 都应该能直接喂给模型。

### 4. pred = logits.argmax(dim=1)

```python
pred = logits.argmax(dim=1)
```

说明：

`pred = logits.argmax(dim=1)` 得到预测类别编号。
- 分类任务要分清 logits 和概率。大多数 PyTorch loss 直接接收 logits，只有推理或展示结果时才需要 sigmoid/softmax。

### 5. Linear -> ReLU -> Linear

```python
Linear -> ReLU -> Linear
```

说明：

MLP 常见结构是 `Linear -> ReLU -> Linear`，隐藏层可以重复堆叠。
- 涉及模型层时，把每一层都看成一次 shape 变换；不确定时在 forward 中临时打印中间结果 shape。

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
- 同时注意 dtype：模型输入通常是浮点张量，分类标签通常是 `torch.long`，二分类 BCE 标签通常是浮点 0/1。

### 9. num_classes-1

```python
num_classes-1
```

说明：

如果 loss 不降，先检查标签是否从 0 到 `num_classes-1`，以及标签 dtype 是否 long。
- 同时注意 dtype：模型输入通常是浮点张量，分类标签通常是 `torch.long`，二分类 BCE 标签通常是浮点 0/1。
- 训练时关注顺序：先前向得到输出，再算 loss，然后清空旧梯度、反向传播、更新参数。顺序乱了通常不会得到正确训练。

### 10. torch.softmax(logits, dim=1)

```python
torch.softmax(logits, dim=1)
```

说明：

评估概率时再用 `torch.softmax(logits, dim=1)`。
- 分类任务要分清 logits 和概率。大多数 PyTorch loss 直接接收 logits，只有推理或展示结果时才需要 sigmoid/softmax。


## 本节任务

- 生成 3 类二维点。
- 搭建 Linear-ReLU-Linear 的 MLP。
- 使用 CrossEntropyLoss 训练并打印准确率。
