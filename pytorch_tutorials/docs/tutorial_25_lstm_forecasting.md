# 第 25 课：LSTM：预测正弦序列下一步

> 这一页只讲本节知识点和常用写法。练习题在 `lessons` 目录，答案在 `answers` 目录。

## 核心概念

- LSTM 用门控结构缓解普通 RNN 的长程依赖困难。
- 时间序列预测常用过去若干步作为输入，预测下一步或未来多步。
- 回归任务输出连续值，常用 MSELoss。

## 关键写法详解

### 1. 常用写法

```python
nn.LSTM(input_size, hidden_size, batch_first=True)
```

说明：

`nn.LSTM(input_size, hidden_size, batch_first=True)` 是门控循环网络。
- 涉及模型层时，把每一层都看成一次 shape 变换；不确定时在 forward 中临时打印中间结果 shape。
- 数据管道的重点是第 0 维样本数一致。进入训练循环后，每个 batch 都应该能直接喂给模型。

### 2. (out, (h_n, c_n))

```python
(out, (h_n, c_n))
```

说明：

LSTM 返回 `(out, (h_n, c_n))`，其中 c_n 是 cell state。
- 涉及模型层时，把每一层都看成一次 shape 变换；不确定时在 forward 中临时打印中间结果 shape。

### 3. out[:, -1, :]

```python
out[:, -1, :]
nn.Linear(hidden_size, output_size)
```

说明：

预测下一步常取 `out[:, -1, :]` 接 `nn.Linear(hidden_size, output_size)`。
- 写这类代码时，第一步先确认张量形状。PyTorch 的很多报错不是公式错了，而是某一维没有对齐。
- 涉及模型层时，把每一层都看成一次 shape 变换；不确定时在 forward 中临时打印中间结果 shape。

### 4. x[i:i+window]

```python
x[i:i+window]
x[i+window]
```

说明：

滑动窗口构造：`x[i:i+window]` 作为输入，`x[i+window]` 作为标签。
- 写这类代码时，第一步先确认张量形状。PyTorch 的很多报错不是公式错了，而是某一维没有对齐。

### 5. 形状约定

```python
[samples, window, features]
```

说明：

时间序列输入 shape 常是 `[samples, window, features]`。
- 写这类代码时，第一步先确认张量形状。PyTorch 的很多报错不是公式错了，而是某一维没有对齐。

### 6. 形状约定

```python
[samples, 1]
```

说明：

回归标签 shape 要和模型输出一致，例如 `[samples, 1]`。
- 写这类代码时，第一步先确认张量形状。PyTorch 的很多报错不是公式错了，而是某一维没有对齐。

### 7. nn.MSELoss()

```python
nn.MSELoss()
nn.L1Loss()
```

说明：

损失常用 `nn.MSELoss()` 或 `nn.L1Loss()`。
- 训练时关注顺序：先前向得到输出，再算 loss，然后清空旧梯度、反向传播、更新参数。顺序乱了通常不会得到正确训练。

### 8. 概念和经验

说明：

序列数值通常需要标准化，真实项目里尤其重要。

### 9. 概念和经验

说明：

预测多步未来可以让输出层输出多个值，或递归地一步步预测。

### 10. 概念和经验

说明：

LSTM 也可能梯度爆炸，必要时配合梯度裁剪。
- 涉及模型层时，把每一层都看成一次 shape 变换；不确定时在 forward 中临时打印中间结果 shape。


## 本节任务

- 生成 sin 曲线滑动窗口数据。
- 输入过去 10 步，预测第 11 步。
- 用 nn.LSTM 和 Linear 完成回归。
