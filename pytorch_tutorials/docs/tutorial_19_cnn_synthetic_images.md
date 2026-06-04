# 第 19 课：CNN 小项目：识别条纹方向

> 这一页只讲本节知识点和常用写法。练习题在 `lessons` 目录，答案在 `answers` 目录。

## 核心概念

- CNN 适合处理局部空间模式，例如边缘、纹理、形状。
- 合成数据能帮助你先掌握训练流程，不依赖下载数据集。
- Flatten 前要确认卷积输出形状，避免 Linear 的输入维度写错。

## 关键写法详解

### 1. 常用写法

```python
Conv2d -> ReLU -> Pool -> Conv2d -> ReLU -> Pool -> Flatten -> Linear
```

说明：

典型 CNN 分类器：`Conv2d -> ReLU -> Pool -> Conv2d -> ReLU -> Pool -> Flatten -> Linear`。

### 2. 概念和经验

说明：

小图像任务可以先用 8、16、32 个通道，不必一开始很大。

### 3. 概念和经验

说明：

卷积层学习局部模式，池化层降低空间尺寸并扩大感受野。

### 4. nn.Sequential

```python
nn.Sequential
```

说明：

`nn.Sequential` 很适合写直线型 CNN。

### 5. channels * height * width

```python
channels * height * width
```

说明：

Flatten 后 Linear 输入维度等于 `channels * height * width`。

### 6. 概念和经验

说明：

如果不想手算维度，可以先跑一个 dummy input 打印 shape。

### 7. 概念和经验

说明：

分类输出层维度等于类别数。

### 8. 概念和经验

说明：

CNN 输入一般要归一化到合理范围，例如 0-1 或均值方差标准化。

### 9. 概念和经验

说明：

合成数据适合练流程，但真实图像还要考虑数据增强和 train/val/test 切分。

### 10. 概念和经验

说明：

训练准确率 1.0 不代表泛化好，要看独立验证集。


## 本节任务

- 生成 16x16 灰度图：竖条纹为 0 类，横条纹为 1 类。
- 写一个小 CNN 分类器。
- 训练并打印准确率。
