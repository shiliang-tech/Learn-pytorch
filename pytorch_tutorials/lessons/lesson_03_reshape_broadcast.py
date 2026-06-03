# 第 03 课：形状变换与广播机制
#
# 核心讲解
# 这里保留最短版本，详细教程请先读：
# ../docs/tutorial_03_reshape_broadcast.md
#
# 1. view/reshape 可以改变张量形状，但元素总数必须一致。
# 2. unsqueeze/squeeze 用于增加或删除长度为 1 的维度。
# 3. 广播让不同形状的张量参与运算，例如 batch 中每一行都加同一个偏置向量。
#
# 操作路线
# 写本节代码时，可以按这个顺序推进：
# 1. 把 torch.arange(12) 变成 3x4 矩阵。
# 2. 创建长度为 4 的 bias，并加到矩阵每一行。
# 3. 用 unsqueeze 把结果变成 1x3x4。
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
# - 把 torch.arange(12) 变成 3x4 矩阵。
# - 创建长度为 4 的 bias，并加到矩阵每一行。
# - 用 unsqueeze 把结果变成 1x3x4。
#
# 做题要求
# - 尽量先不看 answers 目录。
# - 代码需要能直接运行。
# - 打印关键 shape、loss 或 accuracy，确认自己真的跑通了。
#
# ===== 练习区：从这里开始写代码 =====
def main():
    # TODO: 练习 reshape、广播和 unsqueeze。
    pass


if __name__ == "__main__":
    main()
