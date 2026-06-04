# 第 21 课：学习率调度与早停思路

> 这一页只讲本节知识点和常用写法。练习题在 `lessons` 目录，答案在 `answers` 目录。

## 核心概念

- 学习率太大可能震荡，太小可能收敛很慢。
- scheduler 可以在训练过程中调整学习率，例如 StepLR 定期衰减。
- 早停会在验证指标长期不提升时停止训练，避免浪费时间和过拟合。

## 关键写法详解

### 1. StepLR(optimizer, step_size=20, gamma=0.5)

```python
StepLR(optimizer, step_size=20, gamma=0.5)
```

说明：

`StepLR(optimizer, step_size=20, gamma=0.5)` 每 20 轮把学习率乘 0.5。

### 2. scheduler.step()

```python
scheduler.step()
```

说明：

常见调用位置是每个 epoch 训练结束后 `scheduler.step()`。

### 3. scheduler.get_last_lr()

```python
scheduler.get_last_lr()
```

说明：

`scheduler.get_last_lr()` 可以查看当前学习率。

### 4. best_val_loss

```python
best_val_loss
best_val_acc
```

说明：

早停需要记录 `best_val_loss` 或 `best_val_acc`。

### 5. bad_epochs = 0

```python
bad_epochs = 0
bad_epochs += 1
```

说明：

如果指标提升，保存模型并把 `bad_epochs = 0`；否则 `bad_epochs += 1`。

### 6. patience

```python
patience
```

说明：

`patience` 表示允许多少轮不提升。

### 7. 1e-4

```python
1e-4
```

说明：

比较浮点指标时常加一个最小改善阈值，比如 `1e-4`。

### 8. 概念和经验

说明：

val loss 适合早停，val accuracy 也可以，但 accuracy 可能更抖。

### 9. 概念和经验

说明：

ReduceLROnPlateau 是另一种常见调度器，它根据验证指标自动降学习率。

### 10. 概念和经验

说明：

scheduler 和 early stopping 都属于训练策略，不改变模型 forward 结构。


## 本节任务

- 使用 StepLR 每 20 个 epoch 把学习率乘 0.5。
- 记录最好的验证 loss。
- 如果验证 loss 连续若干轮没有提升就提前停止。
