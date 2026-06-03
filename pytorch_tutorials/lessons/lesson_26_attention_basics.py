# 第 26 课：Attention：Query、Key、Value
#
# 核心讲解
# 这里保留最短版本，详细教程请先读：
# ../docs/tutorial_26_attention_basics.md
#
# 1. Attention 用 query 和 key 的相似度作为权重，再对 value 做加权求和。
# 2. 缩放点积注意力会除以 sqrt(d_k)，避免维度变大时 logits 过大。
# 3. 注意力权重经过 softmax 后，每一行通常和为 1。
#
# 操作路线
# 写本节代码时，可以按这个顺序推进：
# 1. 创建 q、k、v 三个张量。
# 2. 计算 scaled dot-product attention。
# 3. 验证 attention weights 最后一维求和为 1。
#
# 小检查
# - 先把关键张量 print 出来，确认数值和 shape 符合预期。
# - 如果报错，优先检查 shape、dtype、device 这三件事。
# - 如果训练结果不动，优先检查是否 zero_grad、backward、step 都写了。
#
# 学习目标
# - 能用自己的话解释本节核心概念。
# - 能不看答案写出一个可以运行的最小 PyTorch 程序。
# - 能通过输出的 shape、loss 或 accuracy 判断代码是否基本正确。
#
# 练习任务
# - 创建 q、k、v 三个张量。
# - 计算 scaled dot-product attention。
# - 验证 attention weights 最后一维求和为 1。
#
# 做题要求
# - 尽量先不看 answers 目录。
# - 代码需要能直接运行。
# - 打印关键 shape、loss 或 accuracy，确认自己真的跑通了。
#
# ===== 练习区：从这里开始写代码 =====
def scaled_dot_product_attention(q, k, v):
    # TODO: 返回 output 和 weights。
    pass


def main():
    # TODO: 构造输入并测试函数。
    pass


if __name__ == "__main__":
    main()
