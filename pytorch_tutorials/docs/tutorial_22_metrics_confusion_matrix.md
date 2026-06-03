# 第 22 课：指标：准确率、精确率、召回率、混淆矩阵

> 这份文档是本节的详细教程和速查表。先读它，再去写 `lesson` 文件；卡住时回来查，不要急着打开答案。

## 学习路径

| 步骤 | 你要做什么 | 目的 |
| --- | --- | --- |
| 1 | 读完本页的“核心概念”和“关键写法” | 先理解整体思路，知道 API 在解决什么问题 |
| 2 | 打开 `../lessons/lesson_22_metrics_confusion_matrix.py` 补全 TODO | 把概念变成能运行的代码 |
| 3 | 运行练习文件，观察输出 | 用 shape、loss、accuracy 判断代码是否合理 |
| 4 | 最后对照 `../answers/answer_22_metrics_confusion_matrix.py` | 查漏补缺，而不是直接抄答案 |

## 核心概念

- accuracy 适合类别均衡场景，但类别不均衡时可能误导。
- precision 关注预测为正的样本中有多少是真的正类。
- recall 关注真实正类中有多少被找出来，混淆矩阵能展示错误类型。

## 理解思路

写 PyTorch 代码时，先不要把注意力放在“背 API”上，而是先问四个问题：

1. 数据是什么形状？例如 `[batch, features]`、`[batch, time, features]` 或 `[N, C, H, W]`。
2. 标签是什么类型？回归通常是浮点，分类通常是整数类别编号或 0/1 浮点标签。
3. 模型输入和输出应该是什么 shape？loss 函数会严格要求它们匹配。
4. 哪些张量需要梯度，哪些只是数据或指标？不要让日志、评估和保存逻辑干扰计算图。

## 关键写法速查

| 写法 | 什么时候用 |
| --- | --- |
| `tp = ((y_true == 1) & (y_pred == 1)).sum()` | `tp = ((y_true == 1) & (y_pred == 1)).sum()` 统计真正例。 |
| `fp`<br>`fn`<br>`tn` | `fp` 是真实为 0 但预测为 1；`fn` 是真实为 1 但预测为 0；`tn` 是真实为 0 且预测为 0。 |
| `accuracy = (tp + tn) / total` | `accuracy = (tp + tn) / total` 表示总体预测正确比例。 |
| `precision = tp / (tp + fp)` | `precision = tp / (tp + fp)` 表示预测为正的样本里多少是真的正。 |
| `recall = tp / (tp + fn)` | `recall = tp / (tp + fn)` 表示真实正样本里多少被找出来。 |
| `f1 = 2 * precision * recall / (precision + recall)` | `f1 = 2 * precision * recall / (precision + recall)` 平衡 precision 和 recall。 |
| 概念 / 经验 | 类别不均衡时 accuracy 可能误导，要同时看 precision/recall/F1。 |
| `[num_classes, num_classes]` | 多分类混淆矩阵可以建立 `[num_classes, num_classes]` 的计数矩阵。 |
| `torch.no_grad()`<br>`model.eval()` | 指标计算通常放在 `torch.no_grad()` 和 `model.eval()` 下。 |
| 概念 / 经验 | 实际项目要明确哪个类别是正类，否则 precision/recall 的含义会混乱。 |

## 练习前代码骨架

下面不是答案，只是提醒你代码应该从哪里开始。真正实现请写在 lesson 文件里。

```python
def main():
    # TODO: 手写二分类指标计算。
    pass


if __name__ == "__main__":
    main()
```

## 写题步骤

1. 先把本节会用到的 import、张量 shape、loss 或模型层写出来。
2. 每完成一步就打印一次关键变量，特别是 `shape`、`dtype`、`device`。
3. 如果是训练任务，先让代码跑通，再观察 loss 是否下降、accuracy 是否合理。
4. 不确定 API 怎么用时，优先回到本页的关键写法，不要急着看答案。

## 常见排错表

| 现象 | 优先检查 |
| --- | --- |
| shape 报错 | 打印参与运算的所有张量 shape，按维度逐个对齐 |
| dtype 报错 | 分类标签通常要 `torch.long`，回归值和模型输入通常要浮点 |
| device 报错 | 模型、输入、标签必须在同一个 device |
| loss 不下降 | 检查是否执行了 `zero_grad()`、`backward()`、`step()`，学习率是否过大或过小 |
| 指标奇怪 | 确认标签含义、输出 shape、阈值或 `argmax(dim=...)` 是否写对 |

## 本节任务

- 给定 y_true 和 y_pred。
- 计算二分类混淆矩阵 TP/FP/FN/TN。
- 计算 accuracy、precision、recall。

## 完成标准

- 练习文件能直接运行，没有异常。
- 你能说清楚每个关键张量的 shape。
- 你能解释为什么使用这些 API，以及为什么要打印这些检查项。
- 看答案后，能关掉答案再独立复写一遍。

## 文件位置

- 练习文件：`../lessons/lesson_22_metrics_confusion_matrix.py`
- 答案文件：`../answers/answer_22_metrics_confusion_matrix.py`

建议先独立完成练习文件，再打开答案对照。能不看答案复写一遍，才算真的吃下来了。
