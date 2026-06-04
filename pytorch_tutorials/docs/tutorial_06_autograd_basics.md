# 第 06 课：自动求导 autograd

> 这一页只讲本节知识点和常用写法。练习题在 `lessons` 目录，答案在 `answers` 目录。

## 核心概念

- requires_grad=True 会让 PyTorch 记录张量参与的计算图。
- loss.backward() 会从标量 loss 反向传播，计算叶子张量的 grad。
- 每次反向传播前通常要清空旧梯度，因为 PyTorch 默认会累加梯度。

## 关键写法详解

### 1. x = torch.tensor(1.0, requires_grad=True)

```python
x = torch.tensor(1.0, requires_grad=True)
```

说明：

`x = torch.tensor(1.0, requires_grad=True)` 会让 PyTorch 记录 x 参与的计算。

### 2. requires_grad=True

```python
requires_grad=True
```

说明：

只有浮点或复数张量能 `requires_grad=True`，整数标签不能求梯度。

### 3. loss.backward()

```python
loss.backward()
gradient
```

说明：

`loss.backward()` 要求 loss 通常是标量；非标量需要传入 `gradient` 参数。

### 4. .grad

```python
.grad
w.grad
```

说明：

叶子张量的梯度保存在 `.grad`，例如 `w.grad`。

### 5. 概念和经验

说明：

PyTorch 默认梯度累加，所以训练循环里必须清空旧梯度。

### 6. with torch.no_grad():

```python
with torch.no_grad():
```

说明：

`with torch.no_grad():` 里面的计算不会被 autograd 记录，常用于手动更新参数和推理。

### 7. x.detach()

```python
x.detach()
```

说明：

`x.detach()` 得到一个不再连接当前计算图的新张量，常用于停止梯度传播。

### 8. loss.item()

```python
loss.item()
.item()
```

说明：

`loss.item()` 只用于日志，不要用 `.item()` 后的 Python 数字继续参与反向传播。

### 9. element 0 of tensors does not require grad

```python
element 0 of tensors does not require grad
```

说明：

如果看到 `element 0 of tensors does not require grad`，说明 loss 和可训练参数之间的计算图断了。

### 10. _

```python
_
```

说明：

如果遇到原地操作报错，检查是否对需要梯度的中间张量用了带 `_` 的方法或切片赋值。


## 本节任务

- 创建一个可求导的标量 w。
- 令 loss=(w-3)^2，调用 backward。
- 打印 w.grad，并解释为什么梯度等于 2*(w-3)。
