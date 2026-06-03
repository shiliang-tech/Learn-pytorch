# 第 26 课：Attention：Query、Key、Value

> 这份文档是本节的详细教程和速查表。先读它，再去写 `lesson` 文件；卡住时回来查，不要急着打开答案。

## 学习路径

| 步骤 | 你要做什么 | 目的 |
| --- | --- | --- |
| 1 | 读完本页的“核心概念”和“关键写法” | 先理解整体思路，知道 API 在解决什么问题 |
| 2 | 打开 `../lessons/lesson_26_attention_basics.py` 补全 TODO | 把概念变成能运行的代码 |
| 3 | 运行练习文件，观察输出 | 用 shape、loss、accuracy 判断代码是否合理 |
| 4 | 最后对照 `../answers/answer_26_attention_basics.py` | 查漏补缺，而不是直接抄答案 |

## 核心概念

- Attention 用 query 和 key 的相似度作为权重，再对 value 做加权求和。
- 缩放点积注意力会除以 sqrt(d_k)，避免维度变大时 logits 过大。
- 注意力权重经过 softmax 后，每一行通常和为 1。

## 理解思路

写 PyTorch 代码时，先不要把注意力放在“背 API”上，而是先问四个问题：

1. 数据是什么形状？例如 `[batch, features]`、`[batch, time, features]` 或 `[N, C, H, W]`。
2. 标签是什么类型？回归通常是浮点，分类通常是整数类别编号或 0/1 浮点标签。
3. 模型输入和输出应该是什么 shape？loss 函数会严格要求它们匹配。
4. 哪些张量需要梯度，哪些只是数据或指标？不要让日志、评估和保存逻辑干扰计算图。

## 关键写法速查

| 写法 | 什么时候用 |
| --- | --- |
| `d_k`<br>`d_model` | Q/K/V 的最后一维是特征维，常记作 `d_k` 或 `d_model`。 |
| `scores = q @ k.transpose(-2, -1)` | `scores = q @ k.transpose(-2, -1)` 得到每个 query 对每个 key 的相似度。 |
| `scores / math.sqrt(q.size(-1))` | `scores / math.sqrt(q.size(-1))` 是缩放点积注意力的标准写法。 |
| `weights = torch.softmax(scores, dim=-1)` | `weights = torch.softmax(scores, dim=-1)` 得到注意力权重。 |
| `output = weights @ v` | `output = weights @ v` 用权重对 value 加权求和。 |
| `[batch, seq, dim]`<br>`[batch, seq, dim]` | 输入可为 `[batch, seq, dim]`，输出通常也是 `[batch, seq, dim]`。 |
| `-1e9` | mask 可在 softmax 前把不可见位置填成很小的数，如 `-1e9`。 |
| 概念 / 经验 | 自注意力是 q/k/v 都来自同一个序列；交叉注意力是 q 和 k/v 来自不同序列。 |
| 概念 / 经验 | 多头注意力会把特征维拆成多个 head 并行计算。 |
| 概念 / 经验 | 排查 attention 时检查 weights 最后一维求和是否接近 1。 |

## 练习前代码骨架

下面不是答案，只是提醒你代码应该从哪里开始。真正实现请写在 lesson 文件里。

```python
def scaled_dot_product_attention(q, k, v):
    # TODO: 返回 output 和 weights。
    pass


def main():
    # TODO: 构造输入并测试函数。
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

- 创建 q、k、v 三个张量。
- 计算 scaled dot-product attention。
- 验证 attention weights 最后一维求和为 1。

## 完成标准

- 练习文件能直接运行，没有异常。
- 你能说清楚每个关键张量的 shape。
- 你能解释为什么使用这些 API，以及为什么要打印这些检查项。
- 看答案后，能关掉答案再独立复写一遍。

## 文件位置

- 练习文件：`../lessons/lesson_26_attention_basics.py`
- 答案文件：`../answers/answer_26_attention_basics.py`

建议先独立完成练习文件，再打开答案对照。能不看答案复写一遍，才算真的吃下来了。
