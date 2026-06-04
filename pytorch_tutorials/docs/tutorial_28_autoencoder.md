# 第 28 课：AutoEncoder：压缩与重建

> 这一页只讲本节知识点和常用写法。练习题在 `lessons` 目录，答案在 `answers` 目录。

## 核心概念

- 自编码器由 encoder 和 decoder 组成，目标是重建输入。
- 瓶颈层维度较小时，模型被迫学习压缩表示。
- 重建任务通常使用 MSELoss 或 BCE 类损失，取决于数据范围。

## 关键写法详解

### 1. encoder

```python
encoder
decoder
```

说明：

AutoEncoder 通常写成 `encoder` 和 `decoder` 两个子网络。

### 2. z = encoder(x)

```python
z = encoder(x)
recon = decoder(z)
```

说明：

`z = encoder(x)` 是压缩表示，`recon = decoder(z)` 是重建结果。

### 3. 概念和经验

说明：

瓶颈层 latent 维度越小，压缩压力越大，重建可能更难。

### 4. nn.MSELoss()

```python
nn.MSELoss()
```

说明：

连续值重建常用 `nn.MSELoss()`；0-1 图像也可考虑 BCE 类损失。

### 5. 概念和经验

说明：

decoder 最后一层是否加激活取决于数据范围，例如 0-1 可加 Sigmoid。

### 6. 概念和经验

说明：

自编码器不需要标签，是自监督/无监督风格的训练。

### 7. 概念和经验

说明：

训练循环和普通回归类似，只是 label 就是输入 x 自己。

### 8. 概念和经验

说明：

可以用 latent 做可视化、异常检测或下游特征。

### 9. 概念和经验

说明：

如果 recon 很差，尝试加大 latent 维度或模型容量。

### 10. 概念和经验

说明：

如果只会复制输入，说明压缩或正则约束可能不够。


## 本节任务

- 生成二维圆环数据。
- 用 encoder 把 2 维压到 1 维，再 decoder 重建到 2 维。
- 训练并打印重建误差。
