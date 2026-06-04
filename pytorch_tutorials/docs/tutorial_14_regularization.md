# 第 14 课：正则化：weight decay 与 Dropout

> 这一页只讲本节知识点和常用写法。练习题在 `lessons` 目录，答案在 `answers` 目录。

## 核心概念

- 正则化的目标是降低过拟合，让模型不要只记住训练数据。
- weight_decay 相当于惩罚过大的权重，常直接传给 optimizer。
- Dropout 在训练时随机丢弃部分激活，eval 时自动关闭。

## 关键写法详解

### 1. weight_decay=1e-4

```python
weight_decay=1e-4
```

说明：

`weight_decay=1e-4` 可以直接写进 Adam/SGD，给权重加 L2 正则效果。

### 2. nn.Dropout(p=0.5)

```python
nn.Dropout(p=0.5)
```

说明：

`nn.Dropout(p=0.5)` 在训练时随机把一部分激活置 0。

### 3. 概念和经验

说明：

Dropout 放在隐藏层后更常见，通常不放在最终输出 logits 后。
- 分类任务要分清 logits 和概率。大多数 PyTorch loss 直接接收 logits，只有推理或展示结果时才需要 sigmoid/softmax。

### 4. model.train()

```python
model.train()
model.eval()
```

说明：

`model.train()` 时 Dropout 随机；`model.eval()` 时 Dropout 关闭。

### 5. 概念和经验

说明：

过拟合表现通常是 train 指标很好、val 指标明显差。

### 6. 概念和经验

说明：

减小模型、增加数据、数据增强、weight decay、Dropout 都是常见正则化手段。

### 7. 概念和经验

说明：

weight decay 太大可能欠拟合，表现为训练集也学不好。

### 8. 概念和经验

说明：

Dropout 太大也会欠拟合，小模型上不一定需要 Dropout。

### 9. 概念和经验

说明：

正则化不是越多越好，要看验证集表现。

### 10. 概念和经验

说明：

调参时一次只改一两个因素，否则很难判断是谁起作用。


## 本节任务

- 构建包含 Dropout 的 MLP。
- 给 Adam 设置 weight_decay。
- 比较 train/eval 模式下同一个输入的输出是否稳定。
