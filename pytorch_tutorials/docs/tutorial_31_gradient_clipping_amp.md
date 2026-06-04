# 第 31 课：梯度裁剪与混合精度入口

> 这一页只讲本节知识点和常用写法。练习题在 `lessons` 目录，答案在 `answers` 目录。

## 核心概念

- 梯度裁剪可以限制梯度范数，缓解梯度爆炸，常见于 RNN/Transformer。
- 混合精度能在 GPU 上用更少显存和更快速度训练，CPU 上通常保持普通精度。
- torch.amp 的启用应根据 cuda 是否可用来决定。

## 关键写法详解

### 1. loss.backward()

```python
loss.backward()
optimizer.step()
```

说明：

`loss.backward()` 后、`optimizer.step()` 前进行梯度裁剪。
- 训练时关注顺序：先前向得到输出，再算 loss，然后清空旧梯度、反向传播、更新参数。顺序乱了通常不会得到正确训练。

### 2. 常用写法

```python
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

说明：

`nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` 限制整体梯度范数。
- 和梯度有关的写法要小心计算图是否被断开。用于打印日志时可以 `.item()`，但参与训练的张量不要过早转成 Python 数字。

### 3. 概念和经验

说明：

梯度裁剪常用于 RNN、LSTM、Transformer，缓解梯度爆炸。
- 涉及模型层时，把每一层都看成一次 shape 变换；不确定时在 forward 中临时打印中间结果 shape。

### 4. 常用写法

```python
torch.amp.autocast(device_type='cuda', enabled=True)
```

说明：

`torch.amp.autocast(device_type='cuda', enabled=True)` 会在合适操作上使用混合精度。
- 如果代码要兼容 CPU/GPU，模型、输入、标签和新建临时张量都要放在同一个 device。

### 5. torch.amp.GradScaler('cuda', enabled=use_amp)

```python
torch.amp.GradScaler('cuda', enabled=use_amp)
```

说明：

`torch.amp.GradScaler('cuda', enabled=use_amp)` 用于缩放 loss，减少 float16 下梯度下溢。
- 同时注意 dtype：模型输入通常是浮点张量，分类标签通常是 `torch.long`，二分类 BCE 标签通常是浮点 0/1。
- 如果代码要兼容 CPU/GPU，模型、输入、标签和新建临时张量都要放在同一个 device。
- 训练时关注顺序：先前向得到输出，再算 loss，然后清空旧梯度、反向传播、更新参数。顺序乱了通常不会得到正确训练。
- 和梯度有关的写法要小心计算图是否被断开。用于打印日志时可以 `.item()`，但参与训练的张量不要过早转成 Python 数字。

### 6. 概念和经验

说明：

AMP 典型顺序：autocast 前向 -> scaler.scale(loss).backward() -> scaler.step(opt) -> scaler.update()。
- 训练时关注顺序：先前向得到输出，再算 loss，然后清空旧梯度、反向传播、更新参数。顺序乱了通常不会得到正确训练。

### 7. scaler.unscale_(optimizer)

```python
scaler.unscale_(optimizer)
```

说明：

如果要裁剪梯度，先 `scaler.unscale_(optimizer)`，再 clip。
- 训练时关注顺序：先前向得到输出，再算 loss，然后清空旧梯度、反向传播、更新参数。顺序乱了通常不会得到正确训练。

### 8. enabled=False

```python
enabled=False
```

说明：

CPU 上通常 `enabled=False`，保持普通 float32 训练。
- 同时注意 dtype：模型输入通常是浮点张量，分类标签通常是 `torch.long`，二分类 BCE 标签通常是浮点 0/1。
- 如果代码要兼容 CPU/GPU，模型、输入、标签和新建临时张量都要放在同一个 device。

### 9. 概念和经验

说明：

不是所有模型都适合 AMP；遇到 NaN 时先关 AMP 排查。

### 10. 概念和经验

说明：

即使不用 AMP，梯度裁剪仍然可以单独使用。


## 本节任务

- 训练一个小网络。
- 反向传播后使用 clip_grad_norm_。
- 使用 autocast/GradScaler 写出兼容 CPU 的训练步骤。
