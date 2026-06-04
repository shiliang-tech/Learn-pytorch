# 第 32 课：端到端小项目：从数据到保存模型

> 这一页只讲本节知识点和常用写法。练习题在 `lessons` 目录，答案在 `answers` 目录。

## 核心概念

- 完整项目通常包含：数据准备、模型、训练循环、验证指标、保存最好模型。
- 把 train_one_epoch 和 evaluate 拆成函数，会让代码更容易维护。
- 最终你应该能独立搭起一个小型监督学习项目骨架。

## 关键写法详解

### 1. 概念和经验

说明：

端到端脚本通常包含：配置、数据、模型、loss、optimizer、train、evaluate、save。

### 2. train_one_epoch

```python
train_one_epoch
```

说明：

`train_one_epoch` 负责 train 模式、前向、loss、反向、更新，并返回训练日志。

### 3. evaluate

```python
evaluate
```

说明：

`evaluate` 负责 eval 模式、no_grad、计算 loss 和指标，不更新参数。

### 4. 概念和经验

说明：

主循环里每个 epoch 调一次训练和验证。

### 5. best_val_acc

```python
best_val_acc
best_val_loss
```

说明：

用 `best_val_acc` 或 `best_val_loss` 判断是否保存模型。

### 6. torch.save(model.state_dict(), path)

```python
torch.save(model.state_dict(), path)
```

说明：

`torch.save(model.state_dict(), path)` 保存验证集最好的参数。

### 7. 概念和经验

说明：

实验要固定随机种子，便于比较不同改动。

### 8. 概念和经验

说明：

每次改模型/学习率/batch size 后，都用同一验证集比较。

### 9. 概念和经验

说明：

日志至少打印 epoch、train_loss、val_loss、val_acc。

### 10. 概念和经验

说明：

项目变大后，可以把 dataset、model、train utils 拆成不同文件，但先把单文件流程写熟。


## 本节任务

- 创建一个非线性二分类数据集。
- 写 train_one_epoch 与 evaluate。
- 保存验证准确率最高的模型参数。
