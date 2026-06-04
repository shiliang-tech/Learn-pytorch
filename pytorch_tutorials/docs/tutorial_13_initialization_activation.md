# 第 13 课：初始化与激活函数

> 这一页只讲本节知识点和常用写法。练习题在 `lessons` 目录，答案在 `answers` 目录。

## 核心概念

- 激活函数引入非线性，否则多层 Linear 仍等价于一个 Linear。
- ReLU 常配合 Kaiming 初始化，Tanh/Sigmoid 更常配合 Xavier 初始化。
- 初始化会影响早期梯度大小，过大或过小都可能让训练不稳定。

## 关键写法详解

### 1. class MLP(nn.Module): __init__

```python
class MLP(nn.Module): __init__
forward
```

说明：

自定义模型写法：`class MLP(nn.Module): __init__` 定义层，`forward` 定义计算。

### 2. super().__init__()

```python
super().__init__()
```

说明：

`super().__init__()` 必须调用，否则 PyTorch 不能正确注册参数和子模块。

### 3. ReLU

```python
ReLU
Tanh
GELU
```

说明：

激活函数如 `ReLU`、`Tanh`、`GELU` 负责引入非线性。

### 4. 概念和经验

说明：

没有激活函数的多层 Linear 仍等价于一个 Linear，表达能力有限。

### 5. nn.init.kaiming_normal_(layer.weight)

```python
nn.init.kaiming_normal_(layer.weight)
```

说明：

`nn.init.kaiming_normal_(layer.weight)` 常配合 ReLU。

### 6. nn.init.xavier_uniform_(layer.weight)

```python
nn.init.xavier_uniform_(layer.weight)
```

说明：

`nn.init.xavier_uniform_(layer.weight)` 常配合 Tanh/Sigmoid 或一般全连接网络。

### 7. nn.init.zeros_(layer.bias)

```python
nn.init.zeros_(layer.bias)
```

说明：

`nn.init.zeros_(layer.bias)` 是常见 bias 初始化。

### 8. for layer in model.modules():

```python
for layer in model.modules():
isinstance(layer, nn.Linear)
```

说明：

遍历模块可用 `for layer in model.modules():`，再用 `isinstance(layer, nn.Linear)` 筛选。

### 9. 概念和经验

说明：

初始化通常在模型创建后、训练开始前做一次。

### 10. 概念和经验

说明：

过深网络还会涉及残差连接、归一化层和更谨慎的初始化策略。


## 本节任务

- 定义一个两层 MLP 类。
- 对 Linear 层使用 kaiming_normal_ 初始化。
- 训练一个非线性二分类任务。
