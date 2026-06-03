# 第 07 课：手写梯度下降：拟合 y=2x+1
#
# 核心讲解
# 这里保留最短版本，详细教程请先读：
# ../docs/tutorial_07_manual_gradient_descent.md
#
# 1. 训练的基本循环是：前向计算、计算损失、反向传播、更新参数、清空梯度。
# 2. 梯度下降用参数减去 learning_rate * gradient，让 loss 往下降方向移动。
# 3. 先手写一次更新过程，可以更清楚 nn.Module 和 optimizer 后面在替你做什么。
#
# 操作路线
# 写本节代码时，可以按这个顺序推进：
# 1. 生成 x 与 y=2x+1 的训练数据。
# 2. 用两个 requires_grad 参数 w、b 手动训练。
# 3. 每轮用 no_grad 更新参数，并把梯度清零。
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
# - 生成 x 与 y=2x+1 的训练数据。
# - 用两个 requires_grad 参数 w、b 手动训练。
# - 每轮用 no_grad 更新参数，并把梯度清零。
#
# 做题要求
# - 尽量先不看 answers 目录。
# - 代码需要能直接运行。
# - 打印关键 shape、loss 或 accuracy，确认自己真的跑通了。
#
# ===== 练习区：从这里开始写代码 =====
def main():
    # TODO: 手写一个 100 轮的梯度下降线性回归。
    pass


if __name__ == "__main__":
    main()
