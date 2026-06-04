# 第 15 课：保存与加载模型参数

> 这一页只讲本节知识点和常用写法。练习题在 `lessons` 目录，答案在 `answers` 目录。

## 核心概念

- 推荐保存 model.state_dict()，它只包含参数和 buffer，结构清晰、可迁移。
- 加载时需要先创建相同结构的模型，再 load_state_dict。
- 推理前记得 model.eval()，避免 Dropout 和 BatchNorm 处在训练行为。

## 关键写法详解

### 1. torch.save(model.state_dict(), path)

```python
torch.save(model.state_dict(), path)
```

说明：

`torch.save(model.state_dict(), path)` 是推荐保存方式，只保存参数和 buffer。

### 2. 常用写法

```python
model = MyModel(); model.load_state_dict(torch.load(path))
```

说明：

加载时先创建同结构模型：`model = MyModel(); model.load_state_dict(torch.load(path))`。

### 3. map_location='cpu'

```python
map_location='cpu'
```

说明：

`map_location='cpu'` 可以把 GPU 保存的参数加载到 CPU。
- 如果代码要兼容 CPU/GPU，模型、输入、标签和新建临时张量都要放在同一个 device。

### 4. model.eval()

```python
model.eval()
```

说明：

推理前调用 `model.eval()`，保证 Dropout/BatchNorm 行为正确。
- 数据管道的重点是第 0 维样本数一致。进入训练循环后，每个 batch 都应该能直接喂给模型。

### 5. 概念和经验

说明：

如果还要恢复训练，应同时保存 optimizer state、epoch、best metric。
- 训练时关注顺序：先前向得到输出，再算 loss，然后清空旧梯度、反向传播、更新参数。顺序乱了通常不会得到正确训练。

### 6. 常用写法

```python
{'model': model.state_dict(), 'optimizer': opt.state_dict(), 'epoch': epoch}
```

说明：

checkpoint 常见结构：`{'model': model.state_dict(), 'optimizer': opt.state_dict(), 'epoch': epoch}`。
- 训练时关注顺序：先前向得到输出，再算 loss，然后清空旧梯度、反向传播、更新参数。顺序乱了通常不会得到正确训练。

### 7. strict=False

```python
strict=False
```

说明：

`strict=False` 可以在模型结构略有变化时加载部分参数，但要仔细检查 missing/unexpected keys。

### 8. model

```python
model
```

说明：

只保存整个 `model` 对象不够稳，代码结构变化后更容易加载失败。

### 9. pathlib.Path

```python
pathlib.Path
```

说明：

保存路径用 `pathlib.Path` 更跨平台，也方便创建目录。

### 10. 概念和经验

说明：

加载后可用同一输入比较两个模型输出，确认参数恢复正确。


## 本节任务

- 训练一个小模型。
- 把 state_dict 保存到 pytorch_tutorials/tmp_linear.pt。
- 创建新模型加载参数，验证两者输出一致。
