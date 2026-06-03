# 第 08 课：nn.Module 版线性回归
#
# 核心讲解
# 这里保留最短版本，详细教程请先读：
# ../docs/tutorial_08_nn_linear_regression.md
#
# 1. nn.Module 是 PyTorch 组织模型参数和前向计算的标准方式。
# 2. nn.Linear(in_features, out_features) 实现 y=xW^T+b。
# 3. optimizer 负责根据梯度更新参数，常见有 SGD、Adam。
#
# 操作路线
# 写本节代码时，可以按这个顺序推进：
# 1. 用 nn.Linear(1, 1) 拟合 y=-3x+0.5。
# 2. 使用 MSELoss 和 SGD。
# 3. 训练结束后打印 weight、bias 和 loss。
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
# - 用 nn.Linear(1, 1) 拟合 y=-3x+0.5。
# - 使用 MSELoss 和 SGD。
# - 训练结束后打印 weight、bias 和 loss。
#
# 做题要求
# - 尽量先不看 answers 目录。
# - 代码需要能直接运行。
# - 打印关键 shape、loss 或 accuracy，确认自己真的跑通了。
#
# ===== 练习区：从这里开始写代码 =====
def main():
    # TODO: 使用 nn.Linear、MSELoss、SGD 完成训练。
    pass


if __name__ == "__main__":
    main()
