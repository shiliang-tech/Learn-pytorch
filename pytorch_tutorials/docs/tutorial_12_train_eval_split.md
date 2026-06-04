# 第 12 课：训练集、验证集与 eval 模式

> 这一页只讲本节知识点和常用写法。练习题在 `lessons` 目录，答案在 `answers` 目录。

## 核心概念

- 训练集用来更新参数，验证集用来观察泛化效果。
- model.train() 与 model.eval() 会影响 Dropout、BatchNorm 等层的行为。
- 验证阶段通常配合 torch.no_grad()，避免构建计算图，节省显存和时间。

## 关键写法详解

### 1. backward

```python
backward
optimizer.step()
```

说明：

训练集用于 `backward` 和 `optimizer.step()`，验证集只用于评估。
- 训练时关注顺序：先前向得到输出，再算 loss，然后清空旧梯度、反向传播、更新参数。顺序乱了通常不会得到正确训练。

### 2. model.train()

```python
model.train()
```

说明：

`model.train()` 会打开 Dropout，并让 BatchNorm 使用当前 batch 统计。
- 数据管道的重点是第 0 维样本数一致。进入训练循环后，每个 batch 都应该能直接喂给模型。

### 3. model.eval()

```python
model.eval()
```

说明：

`model.eval()` 会关闭 Dropout，并让 BatchNorm 使用 running statistics。
- 数据管道的重点是第 0 维样本数一致。进入训练循环后，每个 batch 都应该能直接喂给模型。

### 4. with torch.no_grad():

```python
with torch.no_grad():
```

说明：

验证代码通常写在 `with torch.no_grad():` 中，避免构建计算图。
- 和梯度有关的写法要小心计算图是否被断开。用于打印日志时可以 `.item()`，但参与训练的张量不要过早转成 Python 数字。

### 5. random_split

```python
random_split
```

说明：

切分数据可先用简单切片；正式项目可用 `random_split` 或 sklearn 的 train_test_split。

### 6. 概念和经验

说明：

每个 epoch 后记录 train loss、val loss、val accuracy，更容易发现过拟合。
- 训练时关注顺序：先前向得到输出，再算 loss，然后清空旧梯度、反向传播、更新参数。顺序乱了通常不会得到正确训练。

### 7. 概念和经验

说明：

训练 loss 降、验证 loss 升，通常是过拟合信号。
- 训练时关注顺序：先前向得到输出，再算 loss，然后清空旧梯度、反向传播、更新参数。顺序乱了通常不会得到正确训练。

### 8. optimizer.step()

```python
optimizer.step()
```

说明：

验证时不要调用 `optimizer.step()`，也不要对验证 loss 做 backward。
- 训练时关注顺序：先前向得到输出，再算 loss，然后清空旧梯度、反向传播、更新参数。顺序乱了通常不会得到正确训练。

### 9. 概念和经验

说明：

如果使用 DataLoader，训练 loader 可 shuffle，验证 loader 不需要 shuffle。
- 数据管道的重点是第 0 维样本数一致。进入训练循环后，每个 batch 都应该能直接喂给模型。

### 10. 概念和经验

说明：

保存模型时通常保存验证集表现最好的版本，而不是最后一轮。


## 本节任务

- 把合成分类数据切成训练集和验证集。
- 训练 MLP，并在每个 epoch 后计算验证准确率。
- 验证时使用 model.eval() 和 torch.no_grad()。
