# 第 29 课：VAE：均值、方差与重参数化

> 这一页只讲本节知识点和常用写法。练习题在 `lessons` 目录，答案在 `answers` 目录。

## 核心概念

- VAE 的 encoder 输出潜变量分布参数 mu 和 logvar，而不是单个确定向量。
- 重参数化 z=mu+std*eps 让采样过程仍能反向传播到 encoder。
- VAE loss 通常由重建误差和 KL 散度组成。

## 关键写法详解

### 1. mu

```python
mu
logvar
```

说明：

VAE encoder 输出 `mu` 和 `logvar`，表示潜变量高斯分布参数。

### 2. std = torch.exp(0.5 * logvar)

```python
std = torch.exp(0.5 * logvar)
```

说明：

`std = torch.exp(0.5 * logvar)` 把 log 方差转成标准差。

### 3. eps = torch.randn_like(std)

```python
eps = torch.randn_like(std)
```

说明：

`eps = torch.randn_like(std)` 采样标准正态噪声。

### 4. z = mu + std * eps

```python
z = mu + std * eps
```

说明：

`z = mu + std * eps` 是重参数化技巧，使采样过程可反向传播。

### 5. 概念和经验

说明：

重建损失衡量 decoder 输出和输入的接近程度。

### 6. -0.5 * mean(1 + logvar - mu^2 - exp(logvar))

```python
-0.5 * mean(1 + logvar - mu^2 - exp(logvar))
```

说明：

KL loss 让潜变量分布接近标准正态，常写成 `-0.5 * mean(1 + logvar - mu^2 - exp(logvar))`。

### 7. recon_loss + beta * kl

```python
recon_loss + beta * kl
```

说明：

总 loss 通常是 `recon_loss + beta * kl`，beta 可调节压缩/生成约束强度。

### 8. 概念和经验

说明：

logvar 比直接预测 variance 更稳定，也更不容易出现负方差。

### 9. 概念和经验

说明：

训练 VAE 时 recon 和 KL 要分别打印，方便判断哪一项主导。

### 10. 概念和经验

说明：

VAE 训练好后可以从标准正态采样 z，再用 decoder 生成数据。


## 本节任务

- 实现 reparameterize(mu, logvar)。
- 写一个极小 VAE 处理二维数据。
- 训练时同时计算 recon loss 和 KL loss。
