# 第 30 课：GAN：一维分布生成

> 这一页只讲本节知识点和常用写法。练习题在 `lessons` 目录，答案在 `answers` 目录。

## 核心概念

- GAN 包含生成器 G 和判别器 D，二者交替训练。
- D 学会区分真实样本和生成样本，G 学会骗过 D。
- GAN 训练不稳定，本节只做最小可运行例子，理解训练步骤即可。

## 关键写法详解

### 1. 概念和经验

说明：

GAN 有两个模型：Generator 负责造假，Discriminator 负责辨别真假。

### 2. 概念和经验

说明：

训练 D：真实样本标签是 1，生成样本标签是 0。

### 3. fake = G(noise).detach()

```python
fake = G(noise).detach()
```

说明：

训练 D 时 `fake = G(noise).detach()`，避免 D 的 loss 更新到 G。

### 4. D(G(noise))

```python
D(G(noise))
```

说明：

训练 G：希望 `D(G(noise))` 被判断为真，所以目标标签是 1。

### 5. 概念和经验

说明：

D 和 G 通常各有自己的 optimizer。

### 6. BCEWithLogitsLoss

```python
BCEWithLogitsLoss
```

说明：

`BCEWithLogitsLoss` 适合判别器输出 raw logit 的写法。

### 7. 概念和经验

说明：

GAN 训练可能不稳定，loss 不一定像监督学习那样单调下降。

### 8. 概念和经验

说明：

可以观察生成样本的均值、方差或可视化分布判断效果。

### 9. 概念和经验

说明：

D 太强时 G 学不到东西；G/D 学习率和训练次数需要平衡。

### 10. 概念和经验

说明：

真实图像 GAN 会复杂很多，本节先掌握交替训练流程。


## 本节任务

- 真实数据来自均值 2、标准差 0.5 的一维高斯。
- Generator 把噪声映射成一维样本。
- 交替训练 D 和 G，并打印生成样本均值。
