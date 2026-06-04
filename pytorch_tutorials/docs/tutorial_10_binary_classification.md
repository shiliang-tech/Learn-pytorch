# 第 10 课：二分类：logits、Sigmoid 与 BCEWithLogitsLoss

> 这一页只讲本节知识点和常用写法。练习题在 `lessons` 目录，答案在 `answers` 目录。

## 核心概念

- 二分类模型常输出一个 logit，logit 经过 sigmoid 后变成属于正类的概率。
- BCEWithLogitsLoss 内部包含 sigmoid，比手动 sigmoid 后再 BCE 更数值稳定。
- 预测时一般用 sigmoid(logit)>0.5 得到类别。

## 关键写法详解

### 1. 形状约定

```python
[batch, 1]
```

说明：

二分类可输出一个 logit，shape `[batch, 1]`，标签也整理成 `[batch, 1]` 的 float。

### 2. nn.BCEWithLogitsLoss()

```python
nn.BCEWithLogitsLoss()
```

说明：

`nn.BCEWithLogitsLoss()` 内部已经包含 sigmoid，训练时不要手动 sigmoid。

### 3. probs = torch.sigmoid(logits)

```python
probs = torch.sigmoid(logits)
```

说明：

推理阶段用 `probs = torch.sigmoid(logits)` 得到正类概率。

### 4. pred = probs > 0.5

```python
pred = probs > 0.5
```

说明：

`pred = probs > 0.5` 是默认阈值，也可以按业务需求调阈值。

### 5. .float()

```python
.float()
```

说明：

标签要是 0/1 浮点数；如果是 bool 或 long，通常用 `.float()` 转换。

### 6. ((pred == y.bool()).float().mean())

```python
((pred == y.bool()).float().mean())
```

说明：

准确率可写成 `((pred == y.bool()).float().mean())`。

### 7. CrossEntropyLoss

```python
CrossEntropyLoss
```

说明：

如果输出两个 logits，也可以把二分类当多分类，用 `CrossEntropyLoss` 和 long 标签。

### 8. 概念和经验

说明：

类别极不均衡时，accuracy 可能虚高，要关注 precision、recall 或 AUC。

### 9. pos_weight

```python
pos_weight
```

说明：

`pos_weight` 参数可以让 BCEWithLogitsLoss 更重视正类，适合正负样本不均衡。

### 10. 概念和经验

说明：

logits 不是概率，可以小于 0 或大于 1；只有 sigmoid 后才是 0 到 1。


## 本节任务

- 生成二维点，标签为 x1+x2>0。
- 训练 nn.Linear(2,1) 二分类器。
- 计算训练准确率。
