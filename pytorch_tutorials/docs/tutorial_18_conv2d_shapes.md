# 第 18 课：图像张量与 Conv2d 形状

> 这一页只讲本节知识点和常用写法。练习题在 `lessons` 目录，答案在 `answers` 目录。

## 核心概念

- PyTorch 图像张量常用 NCHW：batch、channel、height、width。
- Conv2d 会用卷积核在空间维度滑动，输出通道数由 out_channels 决定。
- padding、stride、kernel_size 会共同决定输出高宽。

## 关键写法详解

### 1. 形状约定

```python
[N, C, H, W]
[N, H, W, C]
```

说明：

PyTorch 卷积输入是 `[N, C, H, W]`，不是 `[N, H, W, C]`。

### 2. 常用写法

```python
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
```

说明：

`nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)` 是基本写法。

### 3. out_channels

```python
out_channels
```

说明：

`out_channels` 决定输出通道数，也就是学多少个卷积核。

### 4. padding=1, kernel_size=3, stride=1

```python
padding=1, kernel_size=3, stride=1
```

说明：

`padding=1, kernel_size=3, stride=1` 常保持高宽不变。

### 5. nn.MaxPool2d(2)

```python
nn.MaxPool2d(2)
```

说明：

`nn.MaxPool2d(2)` 会把高宽各减半，通道数不变。

### 6. nn.AdaptiveAvgPool2d((1, 1))

```python
nn.AdaptiveAvgPool2d((1, 1))
```

说明：

`nn.AdaptiveAvgPool2d((1, 1))` 可把任意高宽压到 1x1，便于接分类头。

### 7. 公式写法

```python
(H + 2P - K) / S + 1
```

说明：

卷积输出高宽公式大致是 `(H + 2P - K) / S + 1`，向下取整。

### 8. nn.Flatten()

```python
nn.Flatten()
```

说明：

进入 Linear 前用 `nn.Flatten()`，并确认 flatten 后维度。

### 9. 概念和经验

说明：

图像从 PIL/NumPy 来时常是 HWC，需要转换到 CHW。

### 10. 概念和经验

说明：

排查 CNN 第一件事：打印每层输出 shape。


## 本节任务

- 创建形状 [8,1,28,28] 的随机图片 batch。
- 用 Conv2d(1,4,kernel_size=3,padding=1) 处理。
- 接 MaxPool2d(2)，打印每一步 shape。
