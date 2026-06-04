# 第 22 课：指标：准确率、精确率、召回率、混淆矩阵

> 这一页只讲本节知识点和常用写法。练习题在 `lessons` 目录，答案在 `answers` 目录。

## 核心概念

- accuracy 适合类别均衡场景，但类别不均衡时可能误导。
- precision 关注预测为正的样本中有多少是真的正类。
- recall 关注真实正类中有多少被找出来，混淆矩阵能展示错误类型。

## 关键写法详解

### 1. tp = ((y_true == 1) & (y_pred == 1)).sum()

```python
tp = ((y_true == 1) & (y_pred == 1)).sum()
```

说明：

`tp = ((y_true == 1) & (y_pred == 1)).sum()` 统计真正例。

### 2. fp

```python
fp
fn
tn
```

说明：

`fp` 是真实为 0 但预测为 1；`fn` 是真实为 1 但预测为 0；`tn` 是真实为 0 且预测为 0。

### 3. 公式写法

```python
accuracy = (tp + tn) / total
```

说明：

`accuracy = (tp + tn) / total` 表示总体预测正确比例。

### 4. 公式写法

```python
precision = tp / (tp + fp)
```

说明：

`precision = tp / (tp + fp)` 表示预测为正的样本里多少是真的正。

### 5. 公式写法

```python
recall = tp / (tp + fn)
```

说明：

`recall = tp / (tp + fn)` 表示真实正样本里多少被找出来。

### 6. 公式写法

```python
f1 = 2 * precision * recall / (precision + recall)
```

说明：

`f1 = 2 * precision * recall / (precision + recall)` 平衡 precision 和 recall。

### 7. 概念和经验

说明：

类别不均衡时 accuracy 可能误导，要同时看 precision/recall/F1。

### 8. 形状约定

```python
[num_classes, num_classes]
```

说明：

多分类混淆矩阵可以建立 `[num_classes, num_classes]` 的计数矩阵。
- 写这类代码时，第一步先确认张量形状。PyTorch 的很多报错不是公式错了，而是某一维没有对齐。

### 9. torch.no_grad()

```python
torch.no_grad()
model.eval()
```

说明：

指标计算通常放在 `torch.no_grad()` 和 `model.eval()` 下。
- 和梯度有关的写法要小心计算图是否被断开。用于打印日志时可以 `.item()`，但参与训练的张量不要过早转成 Python 数字。

### 10. 概念和经验

说明：

实际项目要明确哪个类别是正类，否则 precision/recall 的含义会混乱。


## 本节任务

- 给定 y_true 和 y_pred。
- 计算二分类混淆矩阵 TP/FP/FN/TN。
- 计算 accuracy、precision、recall。
